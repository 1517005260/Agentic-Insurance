"""spaCy-backed NER with per-passage language routing.

Two responsibilities:

1. **Semantic-type filter** — drop OntoNotes labels that almost always
   carry numeric / temporal / measurement content (CARDINAL / ORDINAL /
   PERCENT / MONEY / QUANTITY / DATE / TIME). This replaces a stack of
   hand-written regex patterns with the NER's own classifier.
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

import spacy


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
    """Return ``"zh"`` if text is Chinese, ``"en"`` otherwise.

    First a fast Han-ideograph check (any CJK char → zh); when there are
    no Han chars, fall back to langdetect, which classifies as ``"en"`` /
    ``"de"`` / etc. — we treat anything non-zh as ``"en"`` because
    ``en_core_web_trf`` is the most reasonable shared fallback.
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
    passage (equivalent to the older single-pipeline behavior).
    """

    def __init__(
        self,
        spacy_model: str,
        zh_spacy_model: Optional[str] = None,
        drop_labels: Optional[Iterable[str]] = None,
    ):
        self._pipelines: Dict[str, Any] = {"en": spacy.load(spacy_model)}
        if zh_spacy_model:
            self._pipelines["zh"] = spacy.load(zh_spacy_model)
        self.drop_labels: FrozenSet[str] = (
            frozenset(drop_labels) if drop_labels is not None else DEFAULT_DROP_LABELS
        )

    @property
    def spacy_model(self):
        """Backwards-compat shim — defaults to the EN pipeline. Prefer
        :meth:`pipeline_for` when the language matters."""
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
            ent_text = ent.text
            if ent_text not in sentence_to_entities[sent_text]:
                sentence_to_entities[sent_text].append(ent_text)
            unique_entities.add(ent_text)
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
