"""Re-export shim — prompt body now lives in the algorithm-layer prompt tree.

The config-center schema needs to import the business RAG system prompt
without dragging the web layer into the algorithm-side import graph,
so the actual definition lives next to the base / proof / graph system
prompts. This shim preserves the old import path for any caller that
still does ``from api.prompts.rag_business import RAG_BUSINESS_SYSTEM_PROMPT``.
"""
from agentic.agent.prompts.rag_business import RAG_BUSINESS_SYSTEM_PROMPT

__all__ = ["RAG_BUSINESS_SYSTEM_PROMPT"]
