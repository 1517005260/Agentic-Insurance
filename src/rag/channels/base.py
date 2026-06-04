"""Shared types and helpers for retrieval channels."""

from collections import defaultdict
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Callable, Dict, Hashable, Iterable, List, Sequence, Tuple

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


def aggregate_by_key(
    raw_hits: Iterable[RawHit],
    key_fn: Callable[[RawHit], Hashable],
    top_k: int,
    representative_page: Callable[[List[RawHit]], Tuple[str, str]] | None = None,
) -> List[ChannelHit]:
    """Apply ``Σ s_i / sqrt(N + 1)`` per ``key_fn(hit)``, return top-K.

    Favors precise high-score over broad low-score: 5 strong hits at 0.8
    each (sum=4.0, /√6≈1.63) beats 20 weak hits at 0.3 each
    (sum=6.0, /√21≈1.31). The denominator ``√(N+1)`` (not ``N``) keeps
    a single-hit page's score equal to its raw score divided by √2,
    preventing N=1 from gaming the rank.

    ``representative_page`` chooses which (file_id, page_id) labels the
    output ChannelHit when ``key_fn`` collapses multiple pages (e.g. doc-
    level aggregation). Defaults to the highest-scoring hit in the group.
    """
    grouped: Dict[Hashable, List[RawHit]] = defaultdict(list)
    for hit in raw_hits:
        grouped[key_fn(hit)].append(hit)

    if representative_page is None:
        def representative_page(hits: List[RawHit]) -> Tuple[str, str]:
            best = max(hits, key=lambda h: h.score)
            return (best.file_id, best.page_id)

    out: List[ChannelHit] = []
    for _key, hits in grouped.items():
        n = len(hits)
        agg_score = sum(h.score for h in hits) / sqrt(n + 1)
        file_id, page_id = representative_page(hits)
        out.append(
            ChannelHit(
                file_id=file_id,
                page_id=page_id,
                score=agg_score,
                evidence=[h.evidence for h in hits if h.evidence is not None],
            )
        )
    out.sort(key=lambda h: h.score, reverse=True)
    return out[:top_k]


def aggregate_per_page(
    raw_hits: Iterable[RawHit],
    top_k: int,
) -> List[ChannelHit]:
    """``Σ s_i / sqrt(N + 1)`` aggregated at (file_id, page_id) granularity."""
    return aggregate_by_key(
        raw_hits,
        key_fn=lambda h: (h.file_id, h.page_id),
        top_k=top_k,
    )


def aggregate_per_doc(
    raw_hits: Iterable[RawHit],
    top_k: int,
) -> List[ChannelHit]:
    """``Σ s_i / sqrt(N + 1)`` aggregated at file_id granularity.

    Used as a cross-doc duplicate guard: when several pages from the
    same document each score independently, this collapses them into
    one doc-level ChannelHit and surfaces the strongest contributing
    page as the representative. Prevents the "two near-duplicate tables
    in different docs both ranking high" failure mode where the agent
    can't tell which is the gold doc.
    """
    return aggregate_by_key(
        raw_hits,
        key_fn=lambda h: h.file_id,
        top_k=top_k,
    )


def reciprocal_rank_fusion(
    rank_lists: Iterable[Sequence[Hashable]],
    *,
    k: int = 60,
) -> Dict[Hashable, float]:
    """Fuse several ranked lists of item keys into one RRF score map.

    Each input is an ordered sequence of hashable keys (best first). The
    score of a key is ``Σ 1 / (k + rank)`` over the lists it appears in
    (rank is 1-based). ``k=60`` is the published convention (Cormack et
    al., SIGIR 2009): unsupervised, weight-free, robust over a range of
    ``k`` — so fusing heterogeneous criteria (each as its own rank list)
    needs no learned weights and no per-corpus calibration.

    Returns ``{key: rrf_score}``; the caller sorts by score desc. An item
    missing from a list simply contributes nothing from that list.
    """
    scores: Dict[Hashable, float] = defaultdict(float)
    for rank_list in rank_lists:
        for rank, key in enumerate(rank_list, start=1):
            scores[key] += 1.0 / (k + rank)
    return dict(scores)


class BaseChannel:
    """Subclasses implement :meth:`retrieve`. ``name`` is used in trace logs."""

    name: str = "channel"

    def retrieve(self, ctx: QueryContext) -> List[ChannelHit]:
        raise NotImplementedError
