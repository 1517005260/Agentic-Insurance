"""spaCy-backed NER with per-passage language routing.

Two responsibilities:

1. **Semantic-type filter** — drop OntoNotes labels that almost always
   carry numeric / temporal / measurement content (CARDINAL / ORDINAL /
   PERCENT / MONEY / QUANTITY / DATE / TIME).
2. **Language routing** — each passage is auto-detected as Chinese vs
   non-Chinese (via langdetect). Chinese passages go through
   ``zh_core_web_trf``, everything else through ``en_core_web_trf``. So
   a corpus of mixed EN / Simplified / Traditional zh gets per-passage
   pipeline selection without the caller setting anything.

Anything that slips through the label filter is caught downstream by the
structural ``normalize.is_junk`` check.
"""

from collections import defaultdict
from typing import Any, Dict, FrozenSet, Iterable, List, Optional

import regex

from config.shared import shared_spacy


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

    spaCy occasionally captures a Chinese product list (``A、B、C`` or
    ``A；B；C``) as a single entity span. Split on the safe set of
    pure-punctuation list separators so each sub-mention enters the
    canonicaliser independently. Surfaces with no separator come back
    as a single-element list, so callers can iterate uniformly.

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


# OntoNotes labels that almost always carry numeric / temporal /
# measurement content rather than a reference-able entity. Dropping these
# at NER time is **domain-neutral** — they correspond to what one would
# normally call "facts" rather than "things".
DEFAULT_DROP_LABELS: FrozenSet[str] = frozenset(
    {
        "CARDINAL",   # 42, three, 1.5
        "ORDINAL",    # 1st, second
        "PERCENT",    # 25%
        "MONEY",      # USD500, $1.5M
        "QUANTITY",   # 5 km, 3kg
        "DATE",       # 2024, last Tuesday
        "TIME",       # 4 pm, 03:00
    }
)


def _detect_lang(text: str) -> str:
    """Return ``"zh"`` if text contains a Han ideograph, ``"en"`` otherwise.

    Any-Han-char → zh works once passage text is the document body alone
    (no metadata prefix). Stray Han characters inside an English passage
    are rare in practice; if they show up the trf-en pipeline still
    handles them gracefully (transformer tokenizers don't crash on OOV
    scripts), the worst case is a misroute on a single passage.
    """
    import regex

    if regex.search(r"\p{Han}", text or ""):
        return "zh"
    return "en"


class SpacyNER:
    """spaCy NER wrapper with configurable label filtering and per-passage
    language routing.

    ``spacy_model`` is the path to the EN pipeline. ``zh_spacy_model`` is
    the optional ZH path; if unset the routing falls back to EN for every
    passage.
    """

    def __init__(
        self,
        spacy_model: str,
        zh_spacy_model: Optional[str] = None,
        drop_labels: Optional[Iterable[str]] = None,
    ):
        # Pipelines come from the process-wide cache (``config.shared``)
        # so a parent that already pre-warmed an EN/ZH trf pipeline at
        # lifespan and a per-ingest LinearRAG share the same Language
        # object instead of paying ~1.5 GB per duplicate ``spacy.load``.
        # Concurrent reads via ``nlp.pipe`` are documented thread-safe;
        # mutation paths are serialised by ``INGEST_LOCK`` upstream.
        self._pipelines: Dict[str, Any] = {"en": shared_spacy(spacy_model)}
        if zh_spacy_model:
            self._pipelines["zh"] = shared_spacy(zh_spacy_model)
        self.drop_labels: FrozenSet[str] = (
            frozenset(drop_labels) if drop_labels is not None else DEFAULT_DROP_LABELS
        )

    @property
    def spacy_model(self):
        """The EN pipeline. Prefer :meth:`pipeline_for` when the language matters."""
        return self._pipelines["en"]

    def pipeline_for(self, lang: str):
        """Return the pipeline for ``lang``, falling back to ``en``."""
        return self._pipelines.get(lang, self._pipelines["en"])

    def batch_ner(self, hash_id_to_passage, max_workers):
        # Group passages by detected language so each pipeline runs once
        # over a contiguous batch (spaCy's pipe is much faster than per-text
        # calls, especially with the trf component).
        by_lang: Dict[str, List[tuple]] = defaultdict(list)
        for hash_id, passage in hash_id_to_passage.items():
            by_lang[_detect_lang(passage)].append((hash_id, passage))

        passage_hash_id_to_entities: Dict[str, List[str]] = {}
        sentence_to_entities: Dict[str, List[str]] = defaultdict(list)

        for lang, items in by_lang.items():
            nlp = self.pipeline_for(lang)
            texts = [t for _, t in items]
            batch_size = max(1, len(texts) // max(1, max_workers))
            for (hash_id, _), doc in zip(items, nlp.pipe(texts, batch_size=batch_size)):
                single_passage, single_sentence = self.extract_entities_sentences(
                    doc, hash_id
                )
                passage_hash_id_to_entities.update(single_passage)
                for sent, ents in single_sentence.items():
                    for e in ents:
                        if e not in sentence_to_entities[sent]:
                            sentence_to_entities[sent].append(e)
        return passage_hash_id_to_entities, sentence_to_entities

    def extract_entities_sentences(self, doc, passage_hash_id):
        sentence_to_entities = defaultdict(list)
        unique_entities = set()
        passage_hash_id_to_entities = {}
        for ent in doc.ents:
            if ent.label_ in self.drop_labels:
                continue
            sent_text = ent.sent.text
            # Fan out catalog/list spans into one surface per mention
            # so each piece deduplicates through the canonicaliser.
            # Surfaces with no separator come back as a single-element
            # list, so the loop body stays uniform.
            for piece in split_catalog_mentions(ent.text):
                if piece not in sentence_to_entities[sent_text]:
                    sentence_to_entities[sent_text].append(piece)
                unique_entities.add(piece)
        passage_hash_id_to_entities[passage_hash_id] = list(unique_entities)
        return passage_hash_id_to_entities, sentence_to_entities

    def question_ner(self, question: str):
        nlp = self.pipeline_for(_detect_lang(question))
        doc = nlp(question)
        question_entities = set()
        for ent in doc.ents:
            if ent.label_ in self.drop_labels:
                continue
            question_entities.add(ent.text.lower())
        return question_entities
