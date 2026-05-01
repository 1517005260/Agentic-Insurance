"""Retrieval channels — uniform ``ChannelHit`` interface for cross-channel RRF.

Each channel reads a :class:`rag.preprocess.QueryContext` and returns a list
of :class:`ChannelHit` ranked by ``score`` desc. The cross-channel RRF in
``rag.aggregate`` ignores absolute scores and only uses rank-within-channel.

Within a channel, multiple sub-paths (sub-queries / sub-patterns) are
aggregated per page via ``base.aggregate_per_page``: ``Σ s_i / sqrt(N + 1)``.
"""

from rag.channels.base import BaseChannel, ChannelHit, aggregate_per_page
from rag.channels.bm25 import BM25Channel
from rag.channels.graph_ppr import GraphPPRChannel
from rag.channels.regex_scan import RegexChannel
from rag.channels.semantic import SemanticChannel

__all__ = [
    "BaseChannel",
    "ChannelHit",
    "aggregate_per_page",
    "SemanticChannel",
    "BM25Channel",
    "GraphPPRChannel",
    "RegexChannel",
]
