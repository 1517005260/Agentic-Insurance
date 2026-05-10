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
    RISK_PREDICT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    WEB_AGENT_SYSTEM_PROMPT,
    WEB_RAG_SYSTEM_PROMPT,
)
from config.linear_rag import LinearRAGConfig
from config.rag import RAGConfig

from config.config_store.entry import ConfigEntry


_LINEAR_RAG_DEFAULTS = LinearRAGConfig()


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
    # ---------- ingest.* ----------
    ConfigEntry(
        key="ingest.parallel_workers",
        type="int",
        default=1,
        min=1,
        max=4,
        group="ingest",
        description=(
            "Number of files whose parse stage may run concurrently. "
            "Index-write stage is always serial (faiss / graph stores "
            "are not safe under concurrent writes). **Default 1** for "
            "memory safety: paddle response cache (~50 MB/file) + spaCy "
            "zh-trf NER (~2 GB resident) + LinearRAG igraph snapshots "
            "push an 8 GB host past OOM at parallel_workers=2 on real "
            "60-page PDFs (verified: backend SIGKILL at swap exhaustion). "
            "Bump to 2 only on hosts with ≥12 GB RAM, 4 only on ≥24 GB. "
            "Process restart required after change (semaphore caps at "
            "boot — see _get_parse_sem)."
        ),
    ),
    # ---------- chat.* ----------
    ConfigEntry(
        key="chat.history_turns",
        type="int",
        default=6,
        min=0,
        max=20,
        group="chat",
        description=(
            "Number of prior (user, assistant) turns fed back into the next "
            "request when a chat session is active. 0 disables multi-turn "
            "(every request acts stateless). RAG path uses this to seed "
            "rewrite + answer; agent path stitches prior final answers in "
            "front of the new query."
        ),
    ),
    # ---------- linear_rag.* (LinearRAG literal-substring backfill) ----------
    # Mirror constants in src/config/linear_rag.py +
    # src/rag/channels/graph_ppr.py:_build_gazetteer (the same defaults
    # are used both at ingest-time backfill AND at query-time PPR
    # gazetteer construction; admins tune one knob, both honour it via
    # GraphPPRChannel constructor and ingest pipeline reading from
    # config store).
    ConfigEntry(
        key="linear_rag.literal_backfill_enabled",
        type="bool",
        default=_LINEAR_RAG_DEFAULTS.literal_backfill_enabled,
        group="linear_rag",
        description=(
            "Enable literal-substring backfill at ingest time. When True, "
            "after spaCy NER, sweep every passage against the union of "
            "discovered entity surfaces and add missing entity↔passage "
            "edges (KAG-style 'domain mount'; covers the contextual-NER "
            "miss rate, ~48% on insurance corpus)."
        ),
    ),
    ConfigEntry(
        key="linear_rag.literal_backfill_min_chars",
        type="int",
        default=_LINEAR_RAG_DEFAULTS.literal_backfill_min_chars,
        min=1,
        max=16,
        group="linear_rag",
        description=(
            "Minimum surface character length for literal backfill. "
            "Drops noise like 'us' / 'irs'. Same default applies to the "
            "query-time PPR gazetteer (graph_ppr channel)."
        ),
    ),
    ConfigEntry(
        key="linear_rag.literal_backfill_multi_word_only",
        type="bool",
        default=_LINEAR_RAG_DEFAULTS.literal_backfill_multi_word_only,
        group="linear_rag",
        description=(
            "Require multi-word surfaces only for literal backfill. "
            "Drops single-word ambiguities like 'axa' / 'company'. "
            "Same default applies to the query-time PPR gazetteer."
        ),
    ),
    # ---------- graph_explore.* (entity_lookup tool runtime) ----------
    ConfigEntry(
        key="graph_explore.entity_lookup_min_sim",
        type="float",
        default=0.6,
        min=0.3,
        max=0.95,
        group="graph_explore",
        description=(
            "Cosine similarity floor for the graph_explore entity_lookup "
            "tool. The disambiguator's 0.85 (precision-tuned for adding "
            "alias edges) is too strict at query time; 0.4 surfaces too "
            "much noise; 0.6 is the empirical sweet spot."
        ),
    ),
    ConfigEntry(
        key="graph_explore.entity_lookup_gradient",
        type="float",
        default=0.5,
        min=0.1,
        max=1.0,
        group="graph_explore",
        description=(
            "Gradient g for gradient_topk_candidates() — controls how "
            "fast similarity scores below the top hit are penalised. "
            "Larger g → flatter top-k (more candidates pass); smaller "
            "g → sharper cutoff (only the top match passes)."
        ),
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
        description="Default number of Tavily search results returned to the chat web mode + web agent.",
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
        description=(
            "System prompt for the needs-analysis workbench (BaseAgent). "
            "Two modes: open-corpus top-3 when no held policies are "
            "supplied; gap analysis + complementary picks when "
            "held_policies_file_ids are provided in the request."
        ),
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
        description=(
            "System prompt for the Policy-Review hidden-risk tab "
            "(single LLM call over a precomputed PPR subgraph). Key "
            "name retained for backward compat with persisted overrides; "
            "current default surfaces semantically adjacent clauses, "
            "not fraud judgments."
        ),
    ),
    ConfigEntry(
        key="prompt.risk_predict",
        type="str",
        default=RISK_PREDICT_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description=(
            "System prompt for the proactive pre-issuance risk-prediction "
            "workbench (GraphAgent driving graph_explore PPR → neighbors → "
            "read flow). Output is a forward-looking risk forecast tied to "
            "the customer profile, not a reactive claim/exclusion judgment."
        ),
    ),
]


CONFIG_ENTRIES_BY_KEY: dict[str, ConfigEntry] = {e.key: e for e in CONFIG_ENTRIES}
