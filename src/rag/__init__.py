"""Single-shot RAG query pipeline.

Four parallel retrieval channels (semantic / BM25 / graph PPR / regex) →
RRF fusion → Qwen3 rerank → LLM answer.

The ``agentic/`` package is a superset that adds tool-calling loops on top
of the same primitives; this package is the single-pass version.
"""

from rag.pipeline import AnswerResult, RAGPipeline, answer_query

__all__ = ["RAGPipeline", "AnswerResult", "answer_query"]
