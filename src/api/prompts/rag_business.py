"""Re-export of the business RAG system prompt.

The config-center schema imports this prompt without dragging the web
layer into the algorithm-side import graph, so the definition lives
next to the base / proof / graph system prompts and is re-exported here
under ``api.prompts.rag_business``.
"""
from agentic.agent.prompts.rag_business import RAG_BUSINESS_SYSTEM_PROMPT

__all__ = ["RAG_BUSINESS_SYSTEM_PROMPT"]
