"""Entity surface cleanup, junk filtering, canonical-form normalization.

Three pure functions in pipeline order: ``cleanup`` → ``is_junk`` →
``canonical_form``. ``normalize_for_hash`` chains them and returns ``None``
when the surface should be dropped.

Design principle: **semantic typing is spaCy's job, structural cleanup is
ours**. The SpacyNER layer (`ner.py`) drops MONEY / PERCENT / QUANTITY /
DATE / TIME / CARDINAL / ORDINAL by ``ent.label_``. This module doesn't
duplicate that — it only handles things spaCy can't reliably catch:
PaddleOCR LaTeX residues, HTML fragments, Unicode weirdness, and surfaces
made entirely of punctuation/digits that somehow slipped past the label
filter. The ``is_junk`` rule is therefore reduced to one line per check:
length ≥ 2 and at least one Letter (Latin OR CJK).

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
# leaks into an entity surface when spaCy splits a Chinese sentence
# mid-clause (`保单。` / `,并选择一` style fragments). Newlines and
# full-width spaces are NOT listed here because ``_WS_RE`` (``\s+``)
# already collapses them to a normal space earlier in ``cleanup``.
# Half-width ``!?`` and full-width ``！？`` cover Chinese terminal
# punctuation beyond the period.
_TRAILING_PUNCT_RE = regex.compile(r"[。\.,，;；:：、!?！？ \t]+$")

# A single dangling opening bracket at end of string — half-width `(`
# or full-width `（`, with optional whitespace before it. spaCy
# occasionally cuts a token like ``万通危疾加护保(优越版)`` exactly at
# the opening bracket, leaving us ``万通危疾加护保(`` which never
# canonicalises to the same entity as the well-bounded mention.
_TRAILING_OPEN_BRACKET_RE = regex.compile(r"\s*[(（]\s*$")


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


def is_junk(surface: str) -> bool:
    """Return True if the surface should never enter the entity universe.

    Single rule: must be at least 2 chars long AND contain at least one
    letter character (Unicode ``\\p{L}``, which covers Latin / CJK /
    Cyrillic / etc.). Anything that's pure digit / punct / symbol / empty
    is junk; anything else delegates to spaCy's ``ent.label_`` filter
    upstream for semantic typing (``MONEY``, ``PERCENT``, etc.).
    """
    if not surface or len(surface) < 2:
        return True
    if not _HAS_LETTER_RE.search(surface):
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
) -> Optional[str]:
    """Cleanup → junk filter → canonical form.

    Returns ``None`` when the surface should not enter the entity universe.
    """
    cleaned = cleanup(raw_surface)
    if is_junk(cleaned):
        return None
    canonical = canonical_form(cleaned, fold_traditional=fold_traditional)
    if not canonical or is_junk(canonical):
        return None
    return canonical
