"""Sentence segmentation via pysbd.

Pure-Python rule-based sentence boundary disambiguation, 22 languages
incl. zh/en. Drives the lightweight passage→sentence split path used by
``text_dense`` / ``maintenance``.
~320 KB package, no model files, sub-millisecond per call.
"""

from typing import Dict, List

import pysbd
import regex

_HAN_RE = regex.compile(r"\p{Han}")
_SEG_CACHE: Dict[str, pysbd.Segmenter] = {}


def _segmenter_for(text: str) -> pysbd.Segmenter:
    """Pick zh segmenter for any-Han input, else en. Cached per language."""
    lang = "zh" if _HAN_RE.search(text) else "en"
    seg = _SEG_CACHE.get(lang)
    if seg is None:
        seg = pysbd.Segmenter(language=lang, clean=False)
        _SEG_CACHE[lang] = seg
    return seg


def split_sentences(text: str, lang: str = "xx") -> List[str]:
    """Return non-empty stripped sentences. ``lang`` is ignored — the
    language is auto-detected per call by a Han-character scan.
    """
    if not text:
        return []
    seg = _segmenter_for(text)
    return [s.strip() for s in seg.segment(text) if s and s.strip()]
