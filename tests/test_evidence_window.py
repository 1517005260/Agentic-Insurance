"""Offline tests for the query-centered evidence window.

``query_window`` returns the best query-matching sentence plus its
neighbors, capped — richer than the single-sentence ``query_snippet`` so a
small model can answer in place. These run without embeddings: a stubbed
encoder pins the center sentence, and a cap test exercises the trim path.
"""
import numpy as np

from agentic.tools.acquisition._preview import query_window, split_sentences


class _StubEmbed:
    """Encoder that maps text to a 2-D one-hot by a keyword, so the
    cosine-argmax in ``query_window`` deterministically selects the
    sentence containing the query keyword."""

    def encode(self, text, *, is_query=False):
        def vec(s: str) -> np.ndarray:
            return np.array([1.0, 0.0]) if "target" in s.lower() else np.array([0.0, 1.0])

        if isinstance(text, str):
            return vec(text)
        return np.stack([vec(t) for t in text], axis=0)


_PAGE = (
    "The opening sentence sets the scene here. "
    "A second filler sentence adds more context now. "
    "This is the target sentence with the answer inside. "
    "A trailing sentence follows the target closely. "
    "And a final unrelated closing remark ends it."
)


def test_window_returns_more_than_one_sentence():
    out = query_window(_PAGE, "target", _StubEmbed(), window_sentences=1, max_chars=600)
    # ±1 around the center → the center plus its two neighbors.
    assert out.count(".") >= 2
    assert "target sentence with the answer" in out
    # Neighbors on both sides are included.
    assert "second filler sentence" in out
    assert "trailing sentence follows" in out


def test_window_respects_max_chars():
    sents = split_sentences(_PAGE)
    out = query_window(_PAGE, "target", _StubEmbed(), window_sentences=2, max_chars=80)
    assert len(out) <= 80
    # The first in-window sentence is never cut mid-way: the window starts
    # at sentences[0] (center=2, radius=2) and that sentence is intact.
    assert out.startswith(sents[0])
    # The cap was actually exercised (tail trimmed with an ellipsis).
    assert out.endswith("…")


def test_window_lexical_fallback_when_no_embeddings():
    class _Dead:
        def encode(self, *a, **k):
            raise RuntimeError("no embeddings")

    out = query_window(_PAGE, "target", _Dead(), max_chars=120)
    # Falls back to the leading max_chars of the page.
    assert out
    assert len(out) <= 120
    assert out.startswith("The opening sentence")


def test_window_empty_page():
    assert query_window("", "target", _StubEmbed()) == ""
