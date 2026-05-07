"""Registered config keys + algorithm-layer default sources.

Every entry's ``default`` is imported live from the place that owns the
constant — ``RAGConfig`` for retrieval knobs, the prompt modules for
system prompts, ``CitationItem._PREVIEW_CHARS`` for the citation cap.
The factory ints (``max_loops`` / ``max_token_budget`` per agent kind)
have no good single import site, so we duplicate them here and let
:mod:`tests.test_config_defaults_in_sync` assert they stay in lockstep
with the factory signatures.

That static cross-check is the cost of avoiding three almost-identical
constants exported from the factory module, which would muddy the
single-source-of-truth contract for every other key.
"""
from typing import List

from agentic.agent.prompts import (
    GRAPH_SYSTEM_PROMPT,
    PROOF_SYSTEM_PROMPT,
    RAG_BUSINESS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from config.rag import RAGConfig

from config.config_store.entry import ConfigEntry


# Pull RAG defaults from a live RAGConfig instance — RAGConfig is the
# canonical home; if a teammate retunes a default there, the schema
# follows automatically and tests would catch any genuine drift.
_RAG_DEFAULTS = RAGConfig()


# ----------------------------- agent factory defaults (mirrored, see note above)

# Mirror the kwarg defaults from ``agentic.agent.factory``:
# * ``build_default_agent``: max_loops=12, max_token_budget=128_000
# * ``build_proof_agent``:   max_loops=16, max_token_budget=128_000
# * ``build_graph_agent``:   max_loops=8,  max_token_budget=64_000
#
# We use the factory defaults rather than the BaseAgent / ProofAgent
# constructor defaults because the lifespan calls the factory, so the
# factory ints are the values that actually take effect today.
_BASE_AGENT_DEFAULT_MAX_LOOPS = 12
_BASE_AGENT_DEFAULT_MAX_TOKEN_BUDGET = 128_000
_PROOF_AGENT_DEFAULT_MAX_LOOPS = 16
_PROOF_AGENT_DEFAULT_MAX_TOKEN_BUDGET = 128_000
_GRAPH_AGENT_DEFAULT_MAX_LOOPS = 8
_GRAPH_AGENT_DEFAULT_MAX_TOKEN_BUDGET = 64_000


# --------------------------------------------------- citation preview default

# Imported here rather than from ``api.services.citation`` to keep the
# algorithm-layer config package free of web-layer imports. The web
# module re-exports the same constant, and the in-sync test pins them
# together.
_CITATION_PREVIEW_CHARS_DEFAULT = 240


CONFIG_ENTRIES: List[ConfigEntry] = [
    # ---------- rag.* ----------
    ConfigEntry(
        key="rag.rrf_k",
        type="int",
        default=_RAG_DEFAULTS.rrf_k,
        min=10,
        max=200,
        group="rag",
        description="RRF damping constant; lower → higher rank weight.",
    ),
    ConfigEntry(
        key="rag.rrf_top_m",
        type="int",
        default=_RAG_DEFAULTS.rrf_top_m,
        min=5,
        max=100,
        group="rag",
        description="Candidates passed from RRF fusion to the reranker.",
    ),
    ConfigEntry(
        key="rag.rerank_top_n",
        type="int",
        default=_RAG_DEFAULTS.rerank_top_n,
        min=3,
        max=30,
        group="rag",
        description="Pages handed to the answer-stage LLM after rerank.",
    ),
    ConfigEntry(
        key="rag.answer_max_tokens",
        type="int",
        default=_RAG_DEFAULTS.answer_max_tokens,
        min=1024,
        max=32_000,
        group="rag",
        description="Max tokens the answer-stage LLM may emit (visible + reasoning).",
    ),
    # ---------- agent.* ----------
    ConfigEntry(
        key="agent.base.max_loops",
        type="int",
        default=_BASE_AGENT_DEFAULT_MAX_LOOPS,
        min=4,
        max=32,
        group="agent.base",
        description="Hard ceiling on tool-calling loops for the base agent.",
    ),
    ConfigEntry(
        key="agent.base.max_token_budget",
        type="int",
        default=_BASE_AGENT_DEFAULT_MAX_TOKEN_BUDGET,
        min=32_000,
        max=256_000,
        group="agent.base",
        description="Token-budget force-final-answer threshold for the base agent.",
    ),
    ConfigEntry(
        key="agent.proof.max_loops",
        type="int",
        default=_PROOF_AGENT_DEFAULT_MAX_LOOPS,
        min=4,
        max=32,
        group="agent.proof",
        description="Hard ceiling on tool-calling loops for the proof agent.",
    ),
    ConfigEntry(
        key="agent.proof.max_token_budget",
        type="int",
        default=_PROOF_AGENT_DEFAULT_MAX_TOKEN_BUDGET,
        min=32_000,
        max=256_000,
        group="agent.proof",
        description="Token-budget early-exit threshold for the proof agent.",
    ),
    ConfigEntry(
        key="agent.graph.max_loops",
        type="int",
        default=_GRAPH_AGENT_DEFAULT_MAX_LOOPS,
        min=4,
        max=24,
        group="agent.graph",
        description="Hard ceiling on tool-calling loops for the graph agent.",
    ),
    ConfigEntry(
        key="agent.graph.max_token_budget",
        type="int",
        default=_GRAPH_AGENT_DEFAULT_MAX_TOKEN_BUDGET,
        min=32_000,
        max=128_000,
        group="agent.graph",
        description="Token-budget force-final threshold for the graph agent.",
    ),
    # ---------- prompt.* ----------
    ConfigEntry(
        key="prompt.rag_business",
        type="str",
        default=RAG_BUSINESS_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt the web RAG runner injects into rag.answer.",
    ),
    ConfigEntry(
        key="prompt.base_agent",
        type="str",
        default=SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for the base acquisition agent.",
    ),
    ConfigEntry(
        key="prompt.proof_agent",
        type="str",
        default=PROOF_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for the typed-closure proof agent.",
    ),
    ConfigEntry(
        key="prompt.graph_agent",
        type="str",
        default=GRAPH_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for the knowledge-graph agent.",
    ),
    # ---------- citation.* ----------
    ConfigEntry(
        key="citation.preview_chars",
        type="int",
        default=_CITATION_PREVIEW_CHARS_DEFAULT,
        min=80,
        max=1000,
        group="citation",
        description="Max characters of page text shown inline with each citation.",
    ),
]


CONFIG_ENTRIES_BY_KEY: dict[str, ConfigEntry] = {e.key: e for e in CONFIG_ENTRIES}
