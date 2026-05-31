"""Question-conditioned preview snippet, shared across acquisition tools.

Previews a page by the single sentence that maximises cosine
similarity to the query embedding, rather than the first N characters
of page Markdown. The agent can then tell from the snippet whether a
candidate page actually addresses the question, instead of seeing a
title / header that happens to lead the page.

Two operating modes:

* **Fast path** — caller supplies pre-fetched ``cached_sentences``
  (list of ``(sentence_text, sentence_embedding)`` from an index like
  ``GraphPPRChannel.passage_sentence_embs(passage_hash)``). No on-the-
  fly encoding; dominated by one matrix-vector dot product.
* **Slow path** — when no cached sentences are passed, the helper
  splits ``page_text`` on sentence boundaries and encodes each
  candidate sentence via ``embed_client``. Used by tools that don't
  index sentences themselves (e.g. ``semantic_search``).

Both paths fall back to ``make_snippet`` (first N chars) on any
failure so the agent always gets *some* preview text.
"""
import re
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from model_client import EmbeddingClient


# Sentence boundary: end-of-sentence punctuation OR a blank line.
# Conservative — never splits mid-acronym (e.g. "Dr.") because we
# require following whitespace.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n\n+")

# Snippets shorter than this are usually fragments (table cells, list
# bullets) and don't carry enough context for the agent to discriminate.
_MIN_SENT_CHARS = 20


def split_sentences(page_text: str) -> List[str]:
    """Public splitter so callers building a reverse index use the same rule."""
    if not page_text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(page_text) if len(s.strip()) >= _MIN_SENT_CHARS]


def query_snippet(
    page_text: str,
    query: str,
    embed_client: "EmbeddingClient",
    *,
    max_chars: int = 240,
    cached_query_emb: Optional[np.ndarray] = None,
    cached_sentences: Optional[Sequence[Tuple[str, np.ndarray]]] = None,
) -> str:
    """Top-cosine sentence on ``page_text`` relative to ``query``.

    Args:
      page_text: The full page Markdown — used for fallback snippets
        and (slow path) for sentence splitting.
      query: Free-text user query. Encoded if no cache supplied.
      embed_client: Encoder for the slow path. Ignored when both
        ``cached_query_emb`` and ``cached_sentences`` are provided.
      max_chars: Hard cap on returned snippet length.
      cached_query_emb: Pre-encoded query (saves one encode call per
        candidate page when callers can amortise).
      cached_sentences: ``[(sent_text, sent_emb), ...]`` for the page
        — fast-path hook for tools with a pre-built sentence index
        (e.g. ``GraphPPRChannel.passage_sentence_embs``). Pass
        ``None`` to encode on the fly.

    Returns:
      The top-cosine sentence truncated to ``max_chars``. Falls back
      to ``make_snippet(page_text, max_chars)`` (first N chars) on
      any failure path — encoding errors, empty sentence split,
      missing embeddings.
    """
    from agentic.tools.acquisition._common import make_snippet

    if not page_text:
        return ""

    q_emb = cached_query_emb
    if q_emb is None and query:
        try:
            q_emb = embed_client.encode(query, is_query=True)
            if q_emb.ndim == 2:
                q_emb = q_emb[0]
        except Exception:
            return make_snippet(page_text, max_chars)
    if q_emb is None:
        return make_snippet(page_text, max_chars)

    if cached_sentences:
        out = _top_cos_snippet(cached_sentences, q_emb, max_chars)
        if out:
            return out
        # Fall through to slow path if cache returned nothing usable.

    sents = split_sentences(page_text)
    if not sents:
        return make_snippet(page_text, max_chars)
    try:
        s_embs = embed_client.encode(sents, is_query=False)
        if s_embs.ndim == 1:
            s_embs = s_embs.reshape(1, -1)
    except Exception:
        return make_snippet(page_text, max_chars)
    pairs = list(zip(sents, s_embs))
    return _top_cos_snippet(pairs, q_emb, max_chars) or make_snippet(page_text, max_chars)


def _top_cos_snippet(
    sentence_embs: Sequence[Tuple[str, np.ndarray]],
    q_emb: np.ndarray,
    max_chars: int,
) -> Optional[str]:
    if not sentence_embs:
        return None
    try:
        texts = [t for t, _ in sentence_embs]
        mat = np.stack([np.asarray(e) for _, e in sentence_embs], axis=0).astype(np.float64)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        q = np.asarray(q_emb).astype(np.float64)
        q /= np.linalg.norm(q) + 1e-9
        sims = mat @ q
        best = int(np.argmax(sims))
    except Exception:
        return None
    return texts[best][:max_chars]
