"""Multilingual stopword-based admission filter for NER spans.

Source data: ``stopwords-iso`` (https://github.com/stopwords-iso/stopwords-iso,
MIT-licensed, 58 languages, ~200 KB JSON). Bundled at
``_stopwords/stopwords-iso.json`` next to this module; loaded once per
process.

Why a stopword override on top of GLiNER:

* GLiNER multi-v2.1 was trained on academic / news / mixed text where the
  first-person plural "we" is overwhelmingly the paper author and gets
  routed to ``person``; same for "I" / "he" / "you" / "they". Adding a
  ``function word`` / ``pronoun`` decoy label to the inference prompt is
  not sufficient: GLiNER's score head does not emit competing scores for
  closed-class function words on these surfaces (measured empirically on
  pilot10 — 'we'=person 0.712, no noise-label competitor at any score).
* The fix is linguistic, not statistical: closed-class function words /
  pronouns / determiners / prepositions are NOT entities in any language,
  and a multilingual stopword lexicon is the authoritative source. Per
  the project's no-hardcoded-blacklist rule, we consult an external
  curated lexicon rather than maintaining one in-tree.

Admission rule (single gate):

    drop  iff   surface.strip().lower() in stopword_set
                AND  score < confidence_floor

The confidence floor preserves the rare case where a stopword-shaped
surface is in fact a high-confidence named entity (e.g. "May" the month
or surname under English; "Will" as the modal verb collides with rare
given-name "Will"). Default floor 0.95 — calibrated such that GLiNER's
hub-pronoun emissions (0.3-0.9 on 'we'/'he'/'you'/'they' and the
Chinese equivalents) all fall below it while extreme-confidence named
entities pass through.
"""

import json
from pathlib import Path
from typing import FrozenSet, Iterable, Optional

_STOPWORDS_JSON = Path(__file__).parent / "_stopwords" / "stopwords-iso.json"

# Process-wide cache: (languages-tuple) -> frozenset[str]. Avoids re-reading
# the JSON on every GLiNERAdapter instantiation (one per LinearRAG per
# storage path; pilot runs can spawn dozens).
_CACHE: dict = {}


def load_stopwords(languages: Iterable[str]) -> FrozenSet[str]:
    """Return the union of lowercased stopwords across the requested ISO-639-1
    language codes (e.g. ``("en", "zh")``).

    Unknown codes are silently skipped (stopwords-iso uses the empty set for
    unsupported languages). Returns an empty frozenset when ``languages`` is
    empty — the StopwordFilter then short-circuits as a no-op.
    """
    key = tuple(sorted({lang.lower() for lang in languages if lang}))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    if not key:
        result: FrozenSet[str] = frozenset()
    else:
        with _STOPWORDS_JSON.open("r", encoding="utf-8") as f:
            corpus = json.load(f)
        merged: set = set()
        for lang in key:
            words = corpus.get(lang)
            if not words:
                continue
            merged.update(w.strip().lower() for w in words if w and w.strip())
        result = frozenset(merged)
    _CACHE[key] = result
    return result


class StopwordFilter:
    """Block GLiNER spans whose surface is a multilingual stopword unless the
    model is extremely confident.

    Disabled (no-op) when ``languages`` is empty or
    ``confidence_floor < 0``.
    """

    def __init__(
        self,
        languages: Optional[Iterable[str]] = None,
        confidence_floor: float = 0.95,
    ):
        self.languages = tuple(languages or ())
        self.confidence_floor = float(confidence_floor)
        self._stopwords: FrozenSet[str] = (
            load_stopwords(self.languages)
            if self.languages and self.confidence_floor >= 0.0
            else frozenset()
        )

    @property
    def enabled(self) -> bool:
        return bool(self._stopwords)

    def is_blocked(self, surface: str, score: float) -> bool:
        """Return True iff the span should be dropped at admission."""
        if not self._stopwords:
            return False
        if score >= self.confidence_floor:
            return False
        return surface.strip().lower() in self._stopwords
