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
    CLAIM_CHECK_SYSTEM_PROMPT,
    COMPARE_SYSTEM_PROMPT,
    EXCLUSION_AUDIT_SYSTEM_PROMPT,
    FRAUD_PPR_SYSTEM_PROMPT,
    GRAPH_SYSTEM_PROMPT,
    POLICY_CALC_SYSTEM_PROMPT,
    PROOF_SYSTEM_PROMPT,
    RAG_BUSINESS_SYSTEM_PROMPT,
    RECOMMEND_SYSTEM_PROMPT,
    REGULATION_SUMMARIZER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    WEB_AGENT_SYSTEM_PROMPT,
    WEB_RAG_SYSTEM_PROMPT,
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
# The web agent has only two tools (web_search / web_fetch); each
# search returns ~5 hits and each fetch returns up to 8 KB of cleaned
# text, so the loop converges fast — 8 loops is plenty and 64 K
# tokens covers a typical multi-source synthesis without going past
# the LLM's effective context.
_WEB_AGENT_DEFAULT_MAX_LOOPS = 8
_WEB_AGENT_DEFAULT_MAX_TOKEN_BUDGET = 64_000


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
    # ---------- agent.web.* ----------
    ConfigEntry(
        key="agent.web.max_loops",
        type="int",
        default=_WEB_AGENT_DEFAULT_MAX_LOOPS,
        min=4,
        max=24,
        group="agent.web",
        description="Hard ceiling on tool-calling loops for the web agent.",
    ),
    ConfigEntry(
        key="agent.web.max_token_budget",
        type="int",
        default=_WEB_AGENT_DEFAULT_MAX_TOKEN_BUDGET,
        min=32_000,
        max=128_000,
        group="agent.web",
        description="Token-budget force-final threshold for the web agent.",
    ),
    # ---------- tavily.* ----------
    ConfigEntry(
        key="tavily.max_results",
        type="int",
        default=5,
        min=3,
        max=20,
        group="tavily",
        description="Default number of Tavily search results returned to the agent / regulation runner.",
    ),
    ConfigEntry(
        key="tavily.search_depth",
        type="str",
        default="basic",
        max_length=16,
        min_length=1,
        group="tavily",
        description='Tavily depth: "basic" (fast) or "advanced" (thorough crawl, more credits).',
    ),
    ConfigEntry(
        key="tavily.include_domains_hk",
        type="str",
        default="ia.org.hk,hkma.gov.hk,sfc.hk,gld.gov.hk",
        max_length=2000,
        min_length=0,
        group="tavily",
        description="Comma-separated HK regulatory domains injected when jurisdiction=hk.",
    ),
    ConfigEntry(
        key="tavily.include_domains_cn",
        type="str",
        default="nfra.gov.cn,csrc.gov.cn,gov.cn,pbc.gov.cn",
        max_length=2000,
        min_length=0,
        group="tavily",
        description="Comma-separated mainland-China regulatory domains injected when jurisdiction=cn.",
    ),
    # ---------- prompt.* (web + workbench prompts) ----------
    ConfigEntry(
        key="prompt.web_rag",
        type="str",
        default=WEB_RAG_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for single-call web RAG (chat web mode).",
    ),
    ConfigEntry(
        key="prompt.web_agent",
        type="str",
        default=WEB_AGENT_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for the web agent (chat web+agent mode).",
    ),
    ConfigEntry(
        key="prompt.regulation",
        type="str",
        default=REGULATION_SUMMARIZER_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for the regulation-search workbench (compliance-strict).",
    ),
    ConfigEntry(
        key="prompt.compare",
        type="str",
        default=COMPARE_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for the multi-product comparison workbench.",
    ),
    ConfigEntry(
        key="prompt.exclusion_audit",
        type="str",
        default=EXCLUSION_AUDIT_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for the underwriting / exclusion audit workbench (ProofAgent forall).",
    ),
    ConfigEntry(
        key="prompt.recommend",
        type="str",
        default=RECOMMEND_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for the product-recommendation workbench (BaseAgent over open corpus).",
    ),
    ConfigEntry(
        key="prompt.claim_check",
        type="str",
        default=CLAIM_CHECK_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for the claim-coverage workbench (BaseAgent → 3-section schema).",
    ),
    ConfigEntry(
        key="prompt.policy_calc",
        type="str",
        default=POLICY_CALC_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for the policy-calculation workbench (BaseAgent + code_run).",
    ),
    ConfigEntry(
        key="prompt.fraud_ppr",
        type="str",
        default=FRAUD_PPR_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for the fraud-PPR analysis (single LLM call over a precomputed PPR subgraph).",
    ),
]


CONFIG_ENTRIES_BY_KEY: dict[str, ConfigEntry] = {e.key: e for e in CONFIG_ENTRIES}
