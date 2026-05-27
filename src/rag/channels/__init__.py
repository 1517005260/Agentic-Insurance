"""Retrieval channels — uniform ``ChannelHit`` interface for cross-channel RRF.

Each channel reads a :class:`rag.preprocess.QueryContext` and returns a list
of :class:`ChannelHit` ranked by ``score`` desc. The cross-channel RRF in
``rag.aggregate`` ignores absolute scores and only uses rank-within-channel.

Within a channel, multiple sub-paths (sub-queries / sub-patterns) are
aggregated via ``base.aggregate_by_key``: ``Σ s_i / sqrt(N + 1)``. The
default page-level aggregator is :func:`base.aggregate_per_page`; the
sibling :func:`base.aggregate_per_doc` collapses to document-level
(used as a cross-doc duplicate guard wherever the same entity surfaces
in multiple files).
"""

from rag.channels.base import (
    BaseChannel,
    ChannelHit,
    RawHit,
    aggregate_by_key,
    aggregate_per_doc,
    aggregate_per_page,
)
from rag.channels.bm25 import BM25Channel
from rag.channels.graph_ppr import GraphPPRChannel
from rag.channels.regex_scan import RegexChannel
from rag.channels.semantic import SemanticChannel

__all__ = [
    "BaseChannel",
    "ChannelHit",
    "RawHit",
    "aggregate_by_key",
    "aggregate_per_doc",
    "aggregate_per_page",
    "SemanticChannel",
    "BM25Channel",
    "GraphPPRChannel",
    "RegexChannel",
]
