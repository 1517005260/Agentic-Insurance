"""Query-time RAG pipeline configuration.

Knobs for the four-channel retrieval flow. Storage paths and API endpoints
come from :mod:`config.settings`; this struct only carries algorithmic /
capacity parameters.
"""

from dataclasses import dataclass


@dataclass
class RAGConfig:
    # ---------- per-channel retrieval ----------
    # Top-K each retrieval sub-path emits before within-channel aggregation.
    semantic_topk_per_subpath: int = 30
    bm25_topk_per_query: int = 30
    ppr_topk: int = 30
    regex_topk_per_pattern: int = 200  # raw matches before page-aggregation
    semantic_channel_topk: int = 30
    bm25_channel_topk: int = 30
    regex_channel_topk: int = 30
    # Saturate per-page unique regex matches (TF cap).
    regex_dedup_cap: int = 5

    # ---------- cross-channel fusion ----------
    rrf_k: int = 60
    rrf_top_m: int = 30  # candidates passed to reranker

    # ---------- rerank ----------
    rerank_top_n: int = 8
    rerank_doc_max_chars: int = 6000  # ≈ 1.5K tokens, page text gets truncated

    # ---------- answer ----------
    # Reasoning models (deepseek-vN, o-series, qwen-think, …) spend most of
    # their completion budget on hidden reasoning tokens before emitting a
    # single visible character; if ``answer_max_tokens`` is too low the
    # final answer truncates mid-sentence (or comes back empty) with
    # ``finish_reason="length"``. 8K total = ~6K reasoning headroom + ~2K
    # visible content, which fits all current reasoning models we use.
    answer_max_tokens: int = 8000

    # ---------- PPR ----------
    ppr_damping: float = 0.5
    ppr_max_iterations: int = 3
    ppr_top_k_sentence: int = 1
    ppr_passage_ratio: float = 1.5
    ppr_passage_node_weight: float = 0.05
    ppr_iteration_threshold: float = 0.5

    # ---------- PPR — three-layer abstraction ----------
    # The three-layer design has physical entities + alias edges +
    # logical clusters. The retrieval-time projection knob picks which
    # graph PPR walks on:
    #
    # * ``ppr_seed_cluster_spread`` (default True): keep PPR on the
    #   physical graph but, when seeding from query entity X, also
    #   activate every alias-cluster sibling of X with a sqrt-damped
    #   share of the reset mass. Cheap (~1 ms), preserves the physical
    #   walk identity (P4 surface attribution intact), recovers the
    #   most common alias miss.
    #
    # * ``ppr_on_logical`` (default False): collapse alias-connected
    #   components into super-nodes and run PPR on the quotient
    #   graph. Closer to HippoRAG's canonical-entity PPR.  Loses
    #   per-physical-surface state identity (P4 needs back-projection
    #   through cluster membership), but storage layer (the physical
    #   graph + alias edges) remains untouched so P1 / P2 repair
    #   properties hold.
    #
    # The two are not mutually exclusive: with ``ppr_on_logical=True``
    # the spread knob is irrelevant (seeds already collapse onto
    # the same supernode).
    ppr_seed_cluster_spread: bool = True
    ppr_on_logical: bool = False

    # ---------- concurrency ----------
    # Cap concurrent HTTP / CPU work across the whole query. The 4 channels
    # plus their internal sub-paths run inside this pool.
    max_workers: int = 8
