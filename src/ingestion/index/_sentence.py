"""Sentence segmentation via spaCy's sentencizer.

A blank pipeline (no model file, no torch) plus the rule-based ``sentencizer``
component is enough for English / Chinese / multilingual splitting and is
fast on CPU. We cache one pipeline per language code.
"""

from typing import Dict, List

import spacy
from spacy.language import Language

_NLP_CACHE: Dict[str, Language] = {}


def _get_nlp(lang: str) -> Language:
    if lang not in _NLP_CACHE:
        nlp = spacy.blank(lang)
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        _NLP_CACHE[lang] = nlp
    return _NLP_CACHE[lang]


def split_sentences(text: str, lang: str = "xx") -> List[str]:
    """Return non-empty stripped sentences. ``lang="xx"`` is multilingual."""
    if not text:
        return []
    doc = _get_nlp(lang)(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]
