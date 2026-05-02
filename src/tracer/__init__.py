"""Per-run trace recorder for both pipelines.

Both ``rag.RAGPipeline`` and ``agentic.BaseAgent`` accept an optional
``Tracer`` and call into a :class:`TraceSession` at well-defined points.
Each session writes a folder under
``STORAGE_PATH/<flavor>/<YYYY-MM-DD>/<run_id>/`` containing structured
JSON that is replayable for postmortem analysis without needing to
re-run the LLM.

Layout per session:

    local_storage/<flavor>/2026-05-01/153022_a1b2c3d4/
        query.json            ← original question + start metadata
        preprocess.json       ← (rag) HyDE / rewrite / regex specs
        channels/
            semantic.json     ← (rag) per-channel hits + scores
            bm25.json
            ...
        fused.json            ← (rag) RRF top-M
        candidates.json       ← (rag) pages loaded for rerank
        rerank.json           ← (rag) reranker top-N output
        trajectory.jsonl      ← (agentic) one line per tool call
        llm_calls.jsonl       ← per LLM round-trip (both flavors)
        final.json            ← answer + summary

The two flavors share the same TraceSession surface; each consumer just
ignores the methods that don't apply to it. Keeping one type avoids
``isinstance(..., AgenticSession)`` noise at the call site.
"""

from tracer.base import TraceSession, Tracer

__all__ = ["TraceSession", "Tracer"]
