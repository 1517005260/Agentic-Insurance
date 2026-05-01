"""Cross-channel reciprocal-rank fusion.

RRF treats each channel's ranking as evidence; a page's fused score is
``Σ_channel 1 / (k + rank)``. ``k=60`` is the value Cormack et al. settled
on and the one Microsoft Azure AI Search / OpenSearch / Elastic use as
their default. The algorithm is robust over a wide range of ``k``; the
single-knob simplicity is the point.
"""

from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple

from rag.channels.base import ChannelHit


FusedHit = Tuple[str, str, float]  # (file_id, page_id, rrf_score)


def rrf(
    channels: Sequence[Iterable[ChannelHit]],
    k: int = 60,
    top_m: int = 30,
) -> List[FusedHit]:
    """Reciprocal-rank fuse all channels and return the top-``top_m``.

    Uses 1-based ranks (``rank=1`` is the best hit) so ``1/(k+1)`` is the
    largest contribution. Only ``(file_id, page_id)`` matters — ties keep
    the first-seen channel's evidence.
    """
    fused: dict[Tuple[str, str], float] = defaultdict(float)
    for hits in channels:
        for rank, hit in enumerate(hits, start=1):
            fused[hit.key] += 1.0 / (k + rank)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [(fid, pid, score) for (fid, pid), score in ranked[:top_m]]
