"""GLiNER-backed NER with open-set labels.

Replaces the prior spaCy en/zh ``_trf`` + HanLP MTL stack. GLiNER multi-v2.1
takes a runtime list of natural-language label descriptions
(``["product", "term", "law", ...]``) and returns spans in one model
forward, so changing domains is a prompt-list change — never a domain
dictionary. The closed OntoNotes / MSRA tagsets the previous models used
have no PRODUCT / TERM / CONCEPT classes, which on insurance / legal /
medical text caused the ~50% recall floor we measured pre-change.

Two structural responsibilities preserved from the old SpacyNER:

1. **Catalog mention split** — spaCy occasionally captured a Chinese
   product list (``A、B、C``) as a single span. The same can happen with
   GLiNER on the same input shape, so :func:`split_catalog_mentions`
   stays as a NER-side post-processor (mention boundary repair, not
   domain knowledge).
2. **Sentence-level NER** — the algorithm needs both
   ``passage → entities`` and ``sentence → entities`` maps. We drive
   sentence segmentation through pysbd (see ``_sentence.py``) so that
   stays correct under arbitrary mixed zh/en input.
"""

from collections import defaultdict
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import regex

from config.shared import shared_gliner
from ingestion.index._sentence import split_sentences
from ingestion.index.linear_rag.stopword_filter import StopwordFilter


# ---------------------------------------------------------- input scrub

# Drop ``<img ...>`` / ``<img />`` outright (no inner text to recover).
_IMG_RE = regex.compile(r"<img\b[^>]*?/?>", regex.IGNORECASE)
# Drop opening / closing ``<script>`` ``<style>`` tags AND their bodies.
# We never want the inner text of a stylesheet entering the entity
# universe (``word-wrap`` / ``break-word`` etc surfaced as entities in
# the previous run).
_SCRIPT_STYLE_RE = regex.compile(
    r"<(script|style)\b[^>]*?>.*?</\1>",
    regex.IGNORECASE | regex.DOTALL,
)
# LaTeX-superscript pattern PaddleOCR spits out for footnote markers,
# e.g. ``$ ^{1} $``. Strip wholesale before NER so they don't bleed
# into surrounding spans.
_LATEX_SUP_RE = regex.compile(r"\$\s*\^\{[^}]*\}\s*\$")
# Markdown table row marker ``|---|---|`` and trailing pipe sequences;
# turning them into spaces lets the per-row content read as one line.
_MD_TABLE_RULE_RE = regex.compile(r"^\s*\|?\s*[:\- ]+\|[:\-| ]+\s*$", regex.MULTILINE)
# Generic ``<tag ...>`` open tag that is NOT a ``<table>``/``<tr>``/``<td>``
# — those have row-level structural meaning we want to preserve as
# newlines (handled separately below). Closing tags ``</...>`` are also
# stripped here.
_HTML_OPEN_TAG_RE = regex.compile(
    r"</?(?!(?:table|thead|tbody|tr|td|th)\b)[A-Za-z][A-Za-z0-9-]*\b[^>]*?>"
)
# Row separators inside an HTML table — ``<tr>`` opens a new line,
# ``</tr>`` closes the line, ``<td>`` / ``<th>`` cell boundaries become
# spaces. We do this BEFORE the generic-tag scrub so the table
# structure survives.
_HTML_TR_OPEN_RE = regex.compile(r"<tr\b[^>]*?>", regex.IGNORECASE)
_HTML_TR_CLOSE_RE = regex.compile(r"</tr\s*>", regex.IGNORECASE)
_HTML_CELL_RE = regex.compile(r"</?(?:td|th)\b[^>]*?>", regex.IGNORECASE)
_HTML_TABLE_BOUNDARY_RE = regex.compile(
    r"</?(?:table|thead|tbody)\b[^>]*?>", regex.IGNORECASE
)
# Markdown table cell separator: a ``|`` is treated as a separator
# only when the line containing it has at least 2 pipes AND starts /
# ends with a pipe (modulo whitespace) — the canonical markdown table
# row shape. A standalone ``|`` in body prose / regex literals /
# pseudocode is left intact. Lines that look like table rows have
# their pipes flattened to spaces; everything else passes through.
_MD_TABLE_ROW_RE = regex.compile(
    r"^\s*\|.*\|.*$",
    regex.MULTILINE,
)


def _flatten_md_table_pipes(text: str) -> str:
    """Replace ``|`` with space only on lines that look like markdown table rows."""
    def _row_repl(m: "regex.Match[str]") -> str:
        return m.group(0).replace("|", " ")
    return _MD_TABLE_ROW_RE.sub(_row_repl, text)
# Final whitespace collapse (multiple spaces / blank lines → one space
# / one newline). Keep newlines so pysbd treats each table row as its
# own line; pysbd happens to split on punctuation, not newlines, but
# downstream ``split_catalog_mentions`` benefits from the line break.
_MULTI_BLANK_RE = regex.compile(r"\n{2,}")
_MULTI_SPACE_RE = regex.compile(r"[ \t]+")


def preclean_for_ner(text: str) -> str:
    """Strip OCR markup so each table row reads as one sentence.

    Applied to passage text on the way into the NER model — not to the
    ``passage_store`` which keeps the original markdown so evidence-side
    rendering stays faithful. Targeting four structural OCR shapes the
    benchmark surfaced as noise sources:

    1. ``<img />`` and ``<script>`` / ``<style>`` blocks (drop entirely).
    2. ``<table>``: ``<tr>`` becomes a newline, ``<td>``/``<th>`` becomes
       a space, so each row reads as one sentence — exactly the shape
       NER models handle well.
    3. Markdown tables (``|---|---|`` rule rows + ``|`` cell separators):
       the rule line is dropped, the pipes become spaces.
    4. LaTeX superscript footnote markers (``$ ^{1} $``).

    All other HTML tags are stripped but their inner text is preserved.
    """
    if not text:
        return text
    s = text
    s = _SCRIPT_STYLE_RE.sub(" ", s)
    s = _IMG_RE.sub(" ", s)
    # HTML table → row-per-line plain text. Order matters: insert row
    # newlines, then cell-boundary spaces, then drop table-level
    # boundaries.
    s = _HTML_TR_OPEN_RE.sub("\n", s)
    s = _HTML_TR_CLOSE_RE.sub("\n", s)
    s = _HTML_CELL_RE.sub(" ", s)
    s = _HTML_TABLE_BOUNDARY_RE.sub(" ", s)
    # Generic tags after table structure.
    s = _HTML_OPEN_TAG_RE.sub(" ", s)
    # Markdown tables: drop the rule row first (so pipe flatten below
    # doesn't leave bare hyphens), then flatten cell pipes only on
    # lines that look like real table rows.
    s = _MD_TABLE_RULE_RE.sub("", s)
    s = _flatten_md_table_pipes(s)
    # LaTeX superscripts.
    s = _LATEX_SUP_RE.sub(" ", s)
    # Whitespace collapse last.
    s = _MULTI_SPACE_RE.sub(" ", s)
    s = _MULTI_BLANK_RE.sub("\n", s)
    return s.strip()


# GLiNER occasionally emits whole sentences (or multi-sentence runs)
# as a single "term" / "concept" span — the mT5 backbone has no
# structural bias toward NP-shape outputs. Two structural shapes are
# safely rejectable without any domain wordlist:
#
# 1. Excessive raw length without a bracket. Real product / clause /
#    regulation names rarely exceed 80 characters; surfaces that do AND
#    carry no ``(``/``（``/``[`` bracket (SKU markers, version tags) are
#    almost always sentence-level spans. Bracketed surfaces of any
#    length are kept ("万通危疾加护保(优越版)(BISP5)" type).
# 2. Interior sentence-ending punctuation. A genuine entity surface
#    never spans across ``。``/``！``/``？``/``；``/``;`` — these are
#    hard sentence boundaries. (ASCII ``.`` is intentionally OUT of
#    this set: product codes / version strings like ``v1.0`` and
#    abbreviations like ``Co.`` use it as an internal character.)
#
# Han-character-count rejection lives in ``normalize.is_junk`` (gated
# by ``LinearRAGConfig.junk_max_han_chars``) — that's the third channel,
# enforced after this layer because canonical_form may compress length.
_MISBOUND_INTERIOR_PUNCT_RE = regex.compile(r"[。！？；;]")
_BRACKET_OR_CODE_RE = regex.compile(r"[(（\[【]")


# Han ideographs — when present we additionally run a POS-aware
# sentence-shape check (R3 below). jieba.posseg is imported lazily so
# the dictionary load (~5MB, 700ms warmup) is paid once on first call,
# not at import time.
_HAN_RE = regex.compile(r"\p{Han}")
_JIEBA_POSSEG = None


def _pos_tag(text: str):
    """Lazy jieba.posseg.cut. Returns list of (word, flag)."""
    global _JIEBA_POSSEG
    if _JIEBA_POSSEG is None:
        import jieba.posseg as pseg
        _JIEBA_POSSEG = pseg
        # Force dictionary build now so the first real call is fast.
        list(pseg.cut("warmup"))
    return list(_JIEBA_POSSEG.cut(text))


def _is_chinese_sentence_shape(text: str) -> bool:
    """POS-aware sentence-fragment detector for Chinese spans.

    Real entity surfaces (product / clause / role names) are noun
    phrases — POS pattern is mostly noun (n*) with optional 形容词 /
    量词 / 限定词. GLiNER's misbound output is sentence-shaped:
    contains a verb AND at least one functional token (介词 p /
    助词 u / 副词 d / 连词 c) that ties the verb to its arguments.
    Pure noun phrases rarely carry both signals together.

    Empirical fit on a 13-fragment / 50-entity benchmark: precision
    0.69, recall 0.85; the 4-5 misses are jieba dictionary noise
    ('身故'/'载有' mistagged as n/b instead of v) which a different
    segmenter (pkuseg) did not improve on this corpus.
    """
    tags = [f for _, f in _pos_tag(text)]
    has_verb = any(t.startswith("v") for t in tags)
    has_func = any(t.startswith(("p", "u", "d", "c")) for t in tags)
    return has_verb and has_func


def is_misbound_span(text: str, max_span_chars: int) -> bool:
    """Return True if the GLiNER span has the structural shape of a sentence.

    Three language-aware rules — any one match rejects the span:

    * R1 — character length above ``max_span_chars`` AND no bracket.
      Bracketed surfaces of any length are kept (real product/SKU
      markers legitimately extend the name).
    * R2 — any hard sentence-ending punctuation (``。``/``！``/``？``/
      ``；``/``;``) appearing inside the span. ASCII ``.`` is excluded
      because it appears inside product codes / abbreviations.
    * R3 — Chinese-only POS-aware sentence detector (see
      :func:`_is_chinese_sentence_shape`). Catches 11-15 char clauses
      below the ``junk_max_han_chars`` cutoff that R1 / R2 cannot see.
    """
    if not text:
        return True
    s = text.strip()
    if not s:
        return True
    if len(s) > max_span_chars and not _BRACKET_OR_CODE_RE.search(s):
        return True
    if _MISBOUND_INTERIOR_PUNCT_RE.search(s):
        return True
    if _HAN_RE.search(s) and _is_chinese_sentence_shape(s):
        return True
    return False


# Pure-punctuation list separators commonly used in Chinese product
# catalog OCR output. Splitting an entity surface on these is safe:
# Chinese registration rules (PRC 企业名称登记管理规定) disallow these
# punctuation characters inside organisation names, so they are almost
# always enumeration glyphs.
#
# Conjunction words (或 / 及 / 与) were intentionally REMOVED after a
# review surfaced realistic counter-examples like "保险及再保险公司" /
# "联通及电信合作社" — those would be incorrectly split. The catalog
# OCR symptom we're targeting (`A(code)、B(code)、C(code)`) is fully
# covered by the punctuation set; conjunction-glued chains are rare
# enough to leave as composite surfaces (the ``is_composite_surface``
# gate downstream prevents them from polluting alias clusters).
_CATALOG_LIST_SEP_RE = regex.compile(r"[、；;•｜|，]+")


def split_catalog_mentions(text: str) -> List[str]:
    """Fan out a catalog/list span into one surface per mention.

    GLiNER (like spaCy before it) occasionally captures a Chinese product
    list (``A、B、C`` or ``A；B；C``) as a single entity span. Split on the
    safe set of pure-punctuation list separators so each sub-mention
    enters the canonicaliser independently. Surfaces with no separator
    come back as a single-element list, so callers can iterate uniformly.

    This function lives on the **NER side** of the pipeline (mention
    boundary repair) — distinct from ``normalize.cleanup`` which only
    handles surface hygiene (HTML, LaTeX, dangling brackets, trailing
    punctuation).
    """
    stripped = text.strip() if text else ""
    if not stripped:
        return []
    parts = _CATALOG_LIST_SEP_RE.split(text)
    out = [p.strip() for p in parts if p and p.strip()]
    return out or [stripped]


# Default open-set label list. Mirrors what the empirical study against
# 4 real insurance documents showed to give the best mix of recall and
# precision: domain-specific surfaces (product names, codes, regulatory
# terms) AND noise-control labels (currency, person role) so we don't
# accidentally absorb pronouns / measurements / dates into the entity
# universe. English wording is intentional — GLiNER's mT5 backbone
# tokenises English label tokens more stably than Chinese ones.
DEFAULT_LABELS: Tuple[str, ...] = (
    "product",
    "term",
    "concept",
    "organization",
    "code",
    "law",
    "regulation",
    "person role",
)


class GLiNERAdapter:
    """GLiNER-backed NER façade compatible with the prior SpacyNER call shape.

    Same public surface as the old SpacyNER (``batch_ner``,
    ``question_ner``, ``extract_entities_sentences``) so all existing
    callsites in ``LinearRAG`` and ``GraphPPRChannel`` swap in without
    touching their orchestration logic.

    Per-passage flow:

    1. ``pysbd`` splits the passage into sentences.
    2. One batched ``model.batch_predict_entities`` call scores all
       sentences against the runtime label list.
    3. Each entity span goes through ``split_catalog_mentions`` so a
       single span like ``"A、B、C"`` fans out to three surfaces.
    4. Results are aggregated into the same ``(passage_to_entities,
       sentence_to_entities)`` dict shape ``LinearRAG`` already
       consumes.
    """

    def __init__(
        self,
        model_id: str,
        labels: Optional[Sequence[str]] = None,
        threshold: float = 0.3,
        batch_size: int = 16,
        max_span_chars: int = 80,
        noise_labels: Optional[Sequence[str]] = None,
        calibration_enabled: bool = False,
        temperature: float = 1.0,
        label_thresholds: Optional[Dict[str, float]] = None,
        stopword_languages: Optional[Sequence[str]] = None,
        stopword_confidence_floor: float = 0.95,
    ):
        # Pulled from the process-wide cache so ingest workers and the
        # lifespan-pinned PPR channel share a single resident copy of
        # the GLiNER weights (see config.shared.shared_gliner).
        self._model = shared_gliner(model_id)
        self.labels: List[str] = list(labels) if labels else list(DEFAULT_LABELS)
        self.threshold = float(threshold)
        self.batch_size = int(batch_size)
        self.max_span_chars = int(max_span_chars)
        # Decoy / noise-sink labels: members of ``labels`` that GLiNER
        # is asked to classify INTO (e.g. "pronoun" / "date" / "number")
        # so junk routes there instead of contaminating real types. The
        # model does the linguistic classification; we just discard the
        # spans it tags with a sink label. ``labels`` must still contain
        # them — they have to be scored to attract their surfaces.
        self.noise_labels: set = set(noise_labels or [])
        # Confidence calibration via temperature scaling. When enabled,
        # the raw GLiNER score is divided by ``temperature`` before the
        # threshold gate (T > 1 tightens; T = 1 is a no-op). When
        # disabled (default), GLiNER's internal threshold is used
        # directly — matching current behaviour exactly.
        self.calibration_enabled: bool = bool(calibration_enabled)
        self.temperature: float = float(temperature) if temperature > 0 else 1.0
        # Label-conditional thresholds. When non-empty, each label's floor
        # is overridden independently; unspecified labels use self.threshold.
        # Calibration is label-stratified: the open-set ``concept`` slot
        # in particular fires on a lot of generic noun phrases and needs
        # a tighter floor than the global default to keep precision up.
        # Empty dict = inert (no change from the global threshold).
        self.label_thresholds: Dict[str, float] = dict(label_thresholds) if label_thresholds else {}
        # Multilingual stopword admission filter — drops NER surfaces
        # whose lowercased form is a closed-class function word in any
        # configured language and whose GLiNER score is below the
        # confidence floor. See ``stopword_filter.StopwordFilter`` for
        # the rule and the rationale.
        self._stopword_filter = StopwordFilter(
            languages=stopword_languages,
            confidence_floor=stopword_confidence_floor,
        )

    # ----------------------------------------------- batch passage NER ----

    def batch_ner(
        self,
        hash_id_to_passage: Dict[str, str],
        max_workers: int,  # kept for SpacyNER-compatible signature; unused
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Run NER over a passage dict; return two maps GLM consumers want.

        Returns ``(passage_hash_id_to_entities, sentence_to_entities)``.
        Entity surfaces are deduplicated per passage and per sentence
        (preserving first-occurrence order).

        Sentence segmentation runs first across all passages so we can
        feed one large batched forward to the model rather than one call
        per passage. ``max_workers`` is ignored — the bottleneck used to
        be NumPy/CPU spaCy forward; on GPU FP16 GLiNER, single-process
        batched inference is faster than thread-fanned single-passage
        calls (no GIL release benefit on a CUDA forward).
        """
        # Build a flat list of (passage_hash, sentence_text) pairs so we
        # can run one big batched forward and re-associate at the end.
        # Empty passages are kept as keys so the caller's diff logic
        # (set(passage_hash_ids) - cached) stays consistent.
        # ``preclean_for_ner`` strips HTML / table / LaTeX markup BEFORE
        # sentence splitting so each table row reads as one sentence —
        # the shape GLiNER's mT5 backbone handles best.
        sentence_index: List[Tuple[str, str]] = []
        passage_hash_id_to_sentences: Dict[str, List[str]] = {}
        for hash_id, passage in hash_id_to_passage.items():
            cleaned = preclean_for_ner(passage)
            sents = split_sentences(cleaned)
            passage_hash_id_to_sentences[hash_id] = sents
            for s in sents:
                sentence_index.append((hash_id, s))

        if not sentence_index:
            return {h: [] for h in hash_id_to_passage}, {}

        all_sentences = [s for _, s in sentence_index]
        # GLiNER ``inference`` returns a list-of-lists of
        # ``{text, label, score, start, end}`` dicts, parallel to
        # inputs. Passing ``batch_size`` explicitly folds the
        # per-sentence forwards into GPU-batched matmuls; the
        # ``inference`` entrypoint will otherwise default to one
        # forward per sentence and tank GPU utilisation.
        #
        # When confidence calibration is enabled, we run at threshold=0
        # to get all raw scores, then apply the calibrated effective
        # threshold ``gliner_threshold * temperature`` manually. This
        # is equivalent to temperature-scaling the score and then
        # comparing against the original threshold (score/T >= thr
        # ⟺ score >= thr * T), but avoids re-entering model internals.
        #
        # Label-specific thresholds also require running at threshold=0
        # and filtering manually, since GLiNER's inference applies a
        # single global floor.
        needs_raw_run = (
            (self.calibration_enabled and self.temperature != 1.0)
            or bool(self.label_thresholds)
        )
        if needs_raw_run:
            calib_thr = (
                self.threshold * self.temperature
                if self.calibration_enabled and self.temperature != 1.0
                else self.threshold
            )
            all_results_raw: List[List[Dict[str, Any]]] = self._model.inference(
                all_sentences,
                self.labels,
                threshold=0.0,
                batch_size=self.batch_size,
                flat_ner=True,
            )
            # Per-span threshold: label_thresholds[label] if present,
            # else calib_thr (which equals self.threshold when T=1).
            def _span_threshold(label: str) -> float:
                return self.label_thresholds.get(label, calib_thr)

            all_results: List[List[Dict[str, Any]]] = [
                [sp for sp in spans
                 if sp.get("score", 0.0) >= _span_threshold(sp.get("label", ""))]
                for spans in all_results_raw
            ]
        else:
            all_results = self._model.inference(
                all_sentences,
                self.labels,
                threshold=self.threshold,
                batch_size=self.batch_size,
                flat_ner=True,
            )

        passage_to_entities: Dict[str, List[str]] = {
            h: [] for h in hash_id_to_passage
        }
        passage_to_seen: Dict[str, set] = {h: set() for h in hash_id_to_passage}
        sentence_to_entities: Dict[str, List[str]] = defaultdict(list)
        sentence_seen: Dict[str, set] = defaultdict(set)

        for (hash_id, sent_text), spans in zip(sentence_index, all_results):
            if not spans:
                continue
            for span in spans:
                if span.get("label") in self.noise_labels:
                    continue
                raw = span.get("text") or ""
                if not raw:
                    continue
                if is_misbound_span(raw, self.max_span_chars):
                    continue
                if self._stopword_filter.is_blocked(raw, span.get("score", 0.0)):
                    continue
                for piece in split_catalog_mentions(raw):
                    if piece not in sentence_seen[sent_text]:
                        sentence_seen[sent_text].add(piece)
                        sentence_to_entities[sent_text].append(piece)
                    if piece not in passage_to_seen[hash_id]:
                        passage_to_seen[hash_id].add(piece)
                        passage_to_entities[hash_id].append(piece)

        return passage_to_entities, dict(sentence_to_entities)

    # ----------------------------------------------- query-side NER ----

    def question_ner(self, question: str) -> set:
        """Return a set of lowercased entity surfaces from the question.

        Used by ``GraphPPRChannel._seed_entities`` to find PPR seeds.
        Same return contract as the prior SpacyNER (set of lowercase
        strings) so the caller is unchanged.
        """
        if not question or not question.strip():
            return set()
        sents = split_sentences(question) or [question]
        needs_raw = (
            (self.calibration_enabled and self.temperature != 1.0)
            or bool(self.label_thresholds)
        )
        if needs_raw:
            calib_thr = (
                self.threshold * self.temperature
                if self.calibration_enabled and self.temperature != 1.0
                else self.threshold
            )
            raw_spans = self._model.inference(
                sents, self.labels, threshold=0.0, batch_size=self.batch_size, flat_ner=True,
            )
            spans = [
                [sp for sp in ss
                 if sp.get("score", 0.0) >= self.label_thresholds.get(sp.get("label", ""), calib_thr)]
                for ss in raw_spans
            ]
        else:
            spans = self._model.inference(
                sents, self.labels, threshold=self.threshold, batch_size=self.batch_size, flat_ner=True,
            )
        out: set = set()
        for sent_spans in spans:
            for span in sent_spans:
                if span.get("label") in self.noise_labels:
                    continue
                raw = (span.get("text") or "").strip()
                if not raw or is_misbound_span(raw, self.max_span_chars):
                    continue
                if self._stopword_filter.is_blocked(raw, span.get("score", 0.0)):
                    continue
                out.add(raw.lower())
        return out

    # ------------------------------------- per-passage helper (test/api) ----

    def extract_entities_sentences(
        self, text: str, passage_hash_id: str
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Single-passage convenience that mirrors the SpacyNER signature.

        Returns ``({passage_hash_id: [entities]}, {sent_text: [entities]})``.
        Internally just routes through :meth:`batch_ner`.
        """
        passage_map, sentence_map = self.batch_ner({passage_hash_id: text}, 1)
        return passage_map, sentence_map
