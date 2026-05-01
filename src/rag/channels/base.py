"""Shared types and helpers for retrieval channels."""

from collections import defaultdict
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Dict, Iterable, List, Sequence

from rag.preprocess import QueryContext


@dataclass
class ChannelHit:
    """One page returned by a channel.

    ``score`` is the within-channel page score (already aggregated). RRF
    only uses the rank order; absolute magnitude is informational.
    """

    file_id: str
    page_id: str
    score: float
    evidence: List[Any] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (self.file_id, self.page_id)


@dataclass
class RawHit:
    """One sub-path observation that contributes to a page-level score."""

    file_id: str
    page_id: str
    score: float
    evidence: Any = None


def aggregate_per_page(
    raw_hits: Iterable[RawHit],
    top_k: int,
) -> List[ChannelHit]:
    """Apply ``Σ s_i / sqrt(N + 1)`` per (file_id, page_id), return top-K.

    Favors precise high-score over broad low-score: 5 strong hits at 0.8 each
    (sum=4.0, /√6≈1.63) beats 20 weak hits at 0.3 each (sum=6.0, /√21≈1.31).
    """
    grouped: Dict[tuple[str, str], List[RawHit]] = defaultdict(list)
    for hit in raw_hits:
        grouped[(hit.file_id, hit.page_id)].append(hit)

    out: List[ChannelHit] = []
    for (file_id, page_id), hits in grouped.items():
        n = len(hits)
        page_score = sum(h.score for h in hits) / sqrt(n + 1)
        out.append(
            ChannelHit(
                file_id=file_id,
                page_id=page_id,
                score=page_score,
                evidence=[h.evidence for h in hits if h.evidence is not None],
            )
        )
    out.sort(key=lambda h: h.score, reverse=True)
    return out[:top_k]


class BaseChannel:
    """Subclasses implement :meth:`retrieve`. ``name`` is used in trace logs."""

    name: str = "channel"

    def retrieve(self, ctx: QueryContext) -> List[ChannelHit]:
        raise NotImplementedError
