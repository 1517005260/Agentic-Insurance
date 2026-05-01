"""Query-time RAG pipeline configuration.

Knobs for the four-channel retrieval flow defined in ``docs/rag_pipeline.md``.
Storage paths and API endpoints come from :mod:`config.settings`; this struct
only carries algorithmic / capacity parameters.
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
    # Saturate per-page unique regex matches (TF cap) — see docs §4.4.
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

    # ---------- PPR (sourced from projects/LinearRAG/src/config.py) ----------
    ppr_damping: float = 0.5
    ppr_max_iterations: int = 3
    ppr_top_k_sentence: int = 1
    ppr_passage_ratio: float = 1.5
    ppr_passage_node_weight: float = 0.05
    ppr_iteration_threshold: float = 0.5

    # ---------- concurrency ----------
    # Cap concurrent HTTP / CPU work across the whole query. The 4 channels
    # plus their internal sub-paths run inside this pool.
    max_workers: int = 8
