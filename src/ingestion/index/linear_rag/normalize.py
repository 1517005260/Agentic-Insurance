"""Entity surface cleanup, junk filtering, canonical-form normalization.

Three pure functions in pipeline order: ``cleanup`` → ``is_junk`` →
``canonical_form``. ``normalize_for_hash`` chains them and returns ``None``
when the surface should be dropped.

Design principle: **semantic typing is the NER layer's job, structural
cleanup is ours**. GLiNER's open-set prompt list controls which spans
ever reach this module (we don't ask it for ``money`` / ``date`` /
``quantity`` labels), so by the time a surface arrives here the only
remaining junk shapes are PaddleOCR LaTeX residues, HTML fragments,
Unicode weirdness, OCR currency-residue tokens (``rmb3``, ``aud750``),
inline CSS attributes (``word-wrap``), and runaway sentence-fragment
spans the model occasionally emits at low threshold. ``is_junk`` codifies
those four structural failure modes alongside the basic length+letter
test.

Multilingual: language-agnostic Unicode-property classes (``\\p{L}``,
``\\p{Han}``) handle CN / EN / Traditional / mixed. OpenCC folds
Traditional → Simplified for cross-script entity dedup.
"""
import html
import unicodedata
from typing import Optional

import ftfy
import regex  # third-party, supports \p{...} Unicode property classes

# OpenCC is loaded lazily; the dictionary takes ~50ms to build and is only
# needed when the surface contains Traditional characters.
_OPENCC_T2S = None


def _opencc_t2s():
    global _OPENCC_T2S
    if _OPENCC_T2S is None:
        from opencc import OpenCC

        _OPENCC_T2S = OpenCC("t2s")
    return _OPENCC_T2S


# ----------------------------------------------------------------- regex

# LaTeX wrappers around real text — match `$...$` only when the content
# carries a LaTeX-distinctive char (``{ } \ ^``). That keeps real currency
# usage like ``$100`` intact while sweeping ``${^{18}}$``, ``$\dagger$`` etc.
_LATEX_WRAPPED_RE = regex.compile(r"\$[^$]*?[{}\\^][^$]*?\$")

# HTML / XML tag fragments anywhere in the surface.
_HTML_TAG_RE = regex.compile(r"</?[A-Za-z][^>]*>")

# Whitespace collapse.
_WS_RE = regex.compile(r"\s+")

# Any letter character — Latin, CJK, Cyrillic, etc. Unicode property class
# ``\p{L}`` covers all script-independent letters.
_HAS_LETTER_RE = regex.compile(r"\p{L}")

# Han (CJK) ideographs — used for language detection and to gate OpenCC.
_HAS_HAN_RE = regex.compile(r"\p{Han}")

# Leading-article strip (English only).
_LEADING_ARTICLE_RE = regex.compile(r"^(the|a|an)\s+", regex.IGNORECASE)

# Trailing whitespace + sentence-ending punctuation that frequently
# leaks into an entity surface when the NER splits a Chinese sentence
# mid-clause (`保单。` / `,并选择一` style fragments). Newlines and
# full-width spaces are NOT listed here because ``_WS_RE`` (``\s+``)
# already collapses them to a normal space earlier in ``cleanup``.
# Half-width ``!?`` and full-width ``！？`` cover Chinese terminal
# punctuation beyond the period. ``$`` is included because GLiNER
# occasionally emits spans like ``"lifetime annuity option $"`` where
# a stray LaTeX dollar opener leaks past the wrapped-LaTeX scrub.
_TRAILING_PUNCT_RE = regex.compile(r"[。\.,，;；:：、!?！？\$ \t]+$")

# A single dangling opening bracket at end of string — half-width `(`
# or full-width `（`, with optional whitespace before it. spaCy
# occasionally cuts a token like ``万通危疾加护保(优越版)`` exactly at
# the opening bracket, leaving us ``万通危疾加护保(`` which never
# canonicalises to the same entity as the well-bounded mention.
_TRAILING_OPEN_BRACKET_RE = regex.compile(r"\s*[(（]\s*$")

# Structural junk patterns observed in OCR + open-set NER output:
#
# * ``_OCR_CURRENCY_RESIDUE_RE`` — ``rmb3`` / ``aud750`` / ``usd500``.
#   PaddleOCR drops the space between currency code and amount when
#   the gap is narrow on the page; the resulting alphanum token has a
#   letter so passes the basic ``is_junk`` rule, but is semantically a
#   MONEY mention NER's prompt list excluded. Restricted to a known
#   ISO 4217 + common 3-letter alias allowlist so genuine product
#   codes like ``"BISP5"``, ``"GMB1"``, ``"PHPS2"`` are not collateral
#   damage. Add codes here only after seeing them produce noise.
# * ``_CSS_ATTR_DENYLIST`` — explicit lookup of inline-style attribute
#   tokens we've seen survive OCR + leak into entity output. Replaces
#   the previous broad ``[a-z]+(-[a-z]+)+`` pattern that mis-killed
#   ``risk-free`` / ``non-guaranteed``-style hyphenated terms.
# * ``han_fragment_max_chars`` — drops Chinese spans whose length
#   exceeds 15 Han characters AND contain no bracket. GLiNER on
#   ``"本公司保留絕對的酌情權決定復歸紅利的現金價值"`` (24 chars) at
#   threshold 0.3 returns the whole assertive clause as an entity;
#   bounded SKU surfaces with brackets like
#   ``"富饒萬家儲蓄保險計劃（5年繳付）(BISP5)"`` legitimately exceed 15
#   chars and are kept (the bracket signals a real product code, not
#   a sentence fragment).
_ISO_CURRENCY_ALLOWLIST = frozenset({
    "usd", "eur", "gbp", "jpy", "cny", "rmb", "hkd", "aud", "cad",
    "chf", "sgd", "mop", "nzd", "twd", "krw", "thb", "myr", "idr",
    "inr", "php", "vnd", "rub", "brl", "mxn", "zar",
})
_OCR_CURRENCY_RESIDUE_RE = regex.compile(r"^([a-z]{2,5})\d+$", regex.IGNORECASE)
_CSS_ATTR_DENYLIST = frozenset({
    "word-wrap", "word-break", "break-word", "break-all",
    "text-align", "text-decoration", "text-transform", "text-overflow",
    "white-space", "vertical-align", "line-height", "letter-spacing",
    "font-family", "font-weight", "font-size", "font-style",
    "background-color", "background-image", "border-color",
    "border-style", "border-width", "border-radius",
    "margin-top", "margin-bottom", "margin-left", "margin-right",
    "padding-top", "padding-bottom", "padding-left", "padding-right",
    "z-index", "box-sizing", "box-shadow", "overflow-x", "overflow-y",
})
# Han-fragment length cutoff. Empirically 15 works for insurance
# product names (longest legit surface in the benchmark: "富饒萬家儲蓄保險計劃"
# = 10 Han chars); legal and patent corpora have longer legitimate
# spans ("中华人民共和国证券法第一百四十二条" = 18) so this MUST be
# domain-configurable. Callers pass the active value via the
# ``han_fragment_max_chars`` keyword to ``is_junk`` / ``normalize_for_hash``;
# the constant here is only the conservative default for callers that
# don't have a ``LinearRAGConfig`` in scope.
_HAN_FRAGMENT_MAX_CHARS_DEFAULT = 15
_BRACKET_RE = regex.compile(r"[(（\[【]")


# --------------------------------------------------------------- functions

def cleanup(surface: str) -> str:
    """Light, language-neutral cleanup.

    * ``ftfy.fix_text`` repairs Unicode (mojibake, NFC inconsistencies).
    * ``html.unescape`` decodes ``&amp;`` style entities.
    * Strip LaTeX-wrapped fragments like ``${^{2}}$`` (PaddleOCR's
      superscript artifact), keeping anything outside the wrappers.
    * Strip HTML tag fragments leaked from OCR'd tables.
    * Collapse whitespace.
    * Trim a sentence-ending punctuation cluster + dangling opening
      bracket — both are common spaCy boundary artefacts on
      bracket-heavy CJK product names (``"万通危疾加护保("`` →
      ``"万通危疾加护保"``). Run after whitespace collapse so the
      regexes can rely on a clean trailing edge.
    """
    if not surface:
        return ""
    s = ftfy.fix_text(surface)
    s = html.unescape(s)
    s = _LATEX_WRAPPED_RE.sub(" ", s)
    s = _HTML_TAG_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    # Bracket repair in three alternating passes, each idempotent:
    #   1. trim trailing punct          (`保单。`        → `保单`)
    #   2. trim a dangling open bracket (`万通危疾加护保(` → `万通危疾加护保`)
    #   3. trim trailing punct again    (covers cases where step 2
    #      uncovered punct underneath, e.g. `保单。(` → `保单。` → `保单`)
    s = _TRAILING_PUNCT_RE.sub("", s)
    s = _TRAILING_OPEN_BRACKET_RE.sub("", s)
    s = _TRAILING_PUNCT_RE.sub("", s)
    return s.strip()


def is_junk(
    surface: str,
    *,
    han_fragment_max_chars: int = _HAN_FRAGMENT_MAX_CHARS_DEFAULT,
) -> bool:
    """Return True if the surface should never enter the entity universe.

    Five structural rules (in order):

    1. Must be at least 2 chars long AND contain at least one letter
       character (``\\p{L}`` — Latin / CJK / Cyrillic / etc.).
    2. Reject OCR currency residues (``rmb3``, ``aud750``, ``usd500``).
    3. Reject inline CSS attribute names (``word-wrap``, ``break-word``).
    4. Reject Han-character spans that look like sentence fragments:
       length > ``han_fragment_max_chars`` Han chars AND no bracket.
       The cutoff is domain-configurable via
       ``LinearRAGConfig.junk_max_han_chars`` — insurance product
       names top out at ~10 Han chars (15 default), but legal clauses
       and patent technique names legitimately reach 18-25.

    Domain semantic typing (``MONEY``, ``PERCENT``, ``DATE``) is
    controlled upstream by the GLiNER label prompt list, not here.
    """
    if not surface or len(surface) < 2:
        return True
    if not _HAS_LETTER_RE.search(surface):
        return True
    # OCR currency residue: only kill when the alpha prefix is a real
    # ISO/common currency code. ``BISP5`` / ``GMB1`` / ``PHPS2`` are
    # product codes and must survive.
    cur_match = _OCR_CURRENCY_RESIDUE_RE.match(surface)
    if cur_match and cur_match.group(1).lower() in _ISO_CURRENCY_ALLOWLIST:
        return True
    if surface.lower() in _CSS_ATTR_DENYLIST:
        return True
    han_chars = _HAS_HAN_RE.findall(surface)
    if (
        len(han_chars) > han_fragment_max_chars
        and not _BRACKET_RE.search(surface)
    ):
        return True
    return False


def detect_lang(text: str) -> str:
    """Crude language signal: any Han ideograph → ``"zh"``, else ``"en"``."""
    return "zh" if _HAS_HAN_RE.search(text) else "en"


def canonical_form(
    surface: str,
    lang: Optional[str] = None,
    *,
    fold_traditional: bool = True,
) -> str:
    """Return the canonical hash key for a surface.

    Pipeline:

    1. ``unicodedata.normalize("NFKC", …)`` — full/half-width unification, ligatures.
    2. For any string with Han characters: optional Traditional → Simplified
       folding via OpenCC.
    3. ``lower()`` — no-op on CJK, useful for embedded Latin in mixed surfaces
       (``"盛利II"`` → ``"盛利ii"``).
    4. For non-CJK strings: strip a leading article (``the``/``a``/``an``).
    5. Whitespace collapse.
    """
    if not surface:
        return ""
    if lang is None:
        lang = detect_lang(surface)
    s = unicodedata.normalize("NFKC", surface)
    if fold_traditional and _HAS_HAN_RE.search(s):
        try:
            s = _opencc_t2s().convert(s)
        except Exception:
            pass  # OpenCC failures shouldn't block normalization
    s = s.lower()
    if lang == "en":
        s = _LEADING_ARTICLE_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def normalize_for_hash(
    raw_surface: str,
    *,
    fold_traditional: bool = True,
    han_fragment_max_chars: int = _HAN_FRAGMENT_MAX_CHARS_DEFAULT,
) -> Optional[str]:
    """Cleanup → junk filter → canonical form.

    Returns ``None`` when the surface should not enter the entity universe.
    ``han_fragment_max_chars`` propagates to ``is_junk`` so the caller's
    ``LinearRAGConfig`` controls the per-domain length cutoff for the
    Chinese sentence-fragment rule.
    """
    cleaned = cleanup(raw_surface)
    if is_junk(cleaned, han_fragment_max_chars=han_fragment_max_chars):
        return None
    canonical = canonical_form(cleaned, fold_traditional=fold_traditional)
    if not canonical or is_junk(canonical, han_fragment_max_chars=han_fragment_max_chars):
        return None
    return canonical
