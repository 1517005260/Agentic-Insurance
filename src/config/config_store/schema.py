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
    BASE_SYSTEM_PROMPT,
    CLAIM_CHECK_SYSTEM_PROMPT,
    COMPARE_SYSTEM_PROMPT,
    EVIDENCE_FS_SYSTEM_PROMPT,
    FRAUD_PPR_SYSTEM_PROMPT,
    POLICY_CALC_SYSTEM_PROMPT,
    RAG_BUSINESS_SYSTEM_PROMPT,
    RECOMMEND_SYSTEM_PROMPT,
    RISK_PREDICT_SYSTEM_PROMPT,
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
# * ``build_base_agent``:  max_loops=24, max_token_budget=128_000
# * ``build_graph_agent``: max_loops=24, max_token_budget=128_000
#
# We use the factory defaults rather than the BaseAgent constructor
# defaults because the lifespan calls the factory, so the factory ints
# are the values that actually take effect today.
#
# Uniform generous caps: give every agent room to fully explore /
# navigate (multi-hop) up to a large-context reader's window, instead
# of terminating early against a tight loop / token cap. A tight budget
# made the graph agent choke on deep questions (force-answer on a
# half-built chain → abstain); widening it recovered accuracy. The caps
# now exist only as a runaway backstop; the token budget (≈ the model
# window) is the real guard.
_BASE_AGENT_DEFAULT_MAX_LOOPS = 24
_BASE_AGENT_DEFAULT_MAX_TOKEN_BUDGET = 128_000
_GRAPH_AGENT_DEFAULT_MAX_LOOPS = 24
# 128 k = a large-context generator's window (Claude / GPT-5-class /
# deepseek-v4-flash) so navigation terminates on a real stop condition,
# not the token cap. When the deployment runs against a smaller-context
# generator like vLLM-served Qwen3-8B (40960 context − 16384 reserved
# output ≈ 24576 effective input), lower this to ~20 000 via admin
# override.
_GRAPH_AGENT_DEFAULT_MAX_TOKEN_BUDGET = 128_000


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
        key="agent.graph.max_loops",
        type="int",
        default=_GRAPH_AGENT_DEFAULT_MAX_LOOPS,
        min=4,
        max=32,
        group="agent.graph",
        description="Hard ceiling on tool-calling loops for the graph agent.",
    ),
    ConfigEntry(
        key="agent.graph.max_token_budget",
        type="int",
        default=_GRAPH_AGENT_DEFAULT_MAX_TOKEN_BUDGET,
        min=12_000,
        max=256_000,
        group="agent.graph",
        description=(
            "Token-budget force-final threshold for the graph agent. "
            "Default 128 000 sized for large-context generators. Lower "
            "to ~20 000 when running against vLLM-served Qwen3-8B "
            "(40960 context − 16384 reserved output ≈ 24576 effective "
            "input − 4 k headroom)."
        ),
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
        default=BASE_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for the base shell agent.",
    ),
    ConfigEntry(
        key="prompt.graph_agent",
        type="str",
        default=EVIDENCE_FS_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for the EvidenceFS graph agent.",
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
            "memory safety: paddle response cache (~50 MB/file) + the "
            "GLiNER NER model / torch runtime + LinearRAG igraph "
            "snapshots can push an 8 GB host past OOM at "
            "parallel_workers=2 on real 60-page PDFs. "
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
    # ---------- linear_rag.* (query-time PPR gazetteer surface filters) ----------
    # Mirror constants in src/config/linear_rag.py. These two knobs bound
    # the surface set of the query-time PPR gazetteer built by
    # ``src/rag/channels/graph_ppr.py:_build_gazetteer``; admins tune one
    # knob and GraphPPRChannel honours it via its constructor reading from
    # the config store.
    ConfigEntry(
        key="linear_rag.literal_backfill_min_chars",
        type="int",
        default=_LINEAR_RAG_DEFAULTS.literal_backfill_min_chars,
        min=1,
        max=16,
        group="linear_rag",
        description=(
            "Minimum surface character length for the query-time PPR "
            "gazetteer (graph_ppr channel). Drops noise like 'us' / 'irs'."
        ),
    ),
    ConfigEntry(
        key="linear_rag.literal_backfill_multi_word_only",
        type="bool",
        default=_LINEAR_RAG_DEFAULTS.literal_backfill_multi_word_only,
        group="linear_rag",
        description=(
            "Require multi-word surfaces only for the query-time PPR "
            "gazetteer (graph_ppr channel). Drops single-word ambiguities "
            "like 'axa' / 'company'."
        ),
    ),
    # ---------- linear_rag.gliner_* (open-set NER) ----------
    # Mirror constants in src/config/linear_rag.py. Changing the label
    # list at runtime swaps the NER prompt — the right knob to turn
    # when adapting to a new domain (medical / legal / patent / …)
    # without writing a domain dictionary.
    ConfigEntry(
        key="linear_rag.gliner_model_id",
        type="str",
        default=_LINEAR_RAG_DEFAULTS.gliner_model_id,
        group="linear_rag",
        description=(
            "HuggingFace repo id for the GLiNER NER model. Weights live "
            "in the standard HF cache (~/.cache/huggingface/hub/). "
            "Default 'urchade/gliner_multiv2.1' is the multilingual "
            "checkpoint validated on insurance corpus."
        ),
    ),
    ConfigEntry(
        key="linear_rag.gliner_labels",
        type="list_str",
        default=list(_LINEAR_RAG_DEFAULTS.gliner_labels),
        min_length=1,
        max_length=32,
        group="linear_rag",
        description=(
            "Open-set NER label prompt. Use English label tokens — the "
            "mT5 backbone tokenises them more stably than Chinese. "
            "Domain swap = label swap (e.g. ['disease','drug','procedure'] "
            "for medical)."
        ),
    ),
    ConfigEntry(
        key="linear_rag.gliner_noise_labels",
        type="list_str",
        default=list(_LINEAR_RAG_DEFAULTS.gliner_noise_labels),
        min_length=0,
        max_length=16,
        group="linear_rag",
        description=(
            "Decoy / noise-sink subset of gliner_labels. GLiNER scores "
            "these (junk like pronouns / bare dates / numbers attaches "
            "to them) and the pipeline discards spans tagged with them. "
            "Model-native noise control, not a surface filter. Members "
            "must also appear in gliner_labels. Empty = inert."
        ),
    ),
    ConfigEntry(
        key="linear_rag.gliner_threshold",
        type="float",
        default=_LINEAR_RAG_DEFAULTS.gliner_threshold,
        min=0.0,
        max=1.0,
        group="linear_rag",
        description=(
            "Score floor for emitted GLiNER spans. 0.3 trades off recall "
            "against noise; lower values surface long sentence-fragment spans."
        ),
    ),
    ConfigEntry(
        key="linear_rag.gliner_label_thresholds",
        type="dict_str_float",
        default=dict(_LINEAR_RAG_DEFAULTS.gliner_label_thresholds),
        group="linear_rag",
        description=(
            "Per-label GLiNER score thresholds (label-conditional calibration). "
            "Overrides gliner_threshold for the named labels; unspecified labels "
            "use gliner_threshold. The default {'concept': 0.5} tightens the "
            "noisiest open-set slot, trimming concept over-generation with "
            "little page-recall loss vs the global 0.3 floor. "
            "Empty dict = inert (all labels use gliner_threshold)."
        ),
    ),
    ConfigEntry(
        key="linear_rag.junk_max_han_chars",
        type="int",
        default=_LINEAR_RAG_DEFAULTS.junk_max_han_chars,
        min=6,
        max=40,
        group="linear_rag",
        description=(
            "Max Han-character length for an unbraced entity surface. "
            "Surfaces above this are rejected as sentence-fragment leakage. "
            "Insurance product names top out at ~10 (default 12); legal / "
            "patent corpora typically need 20-25."
        ),
    ),
    ConfigEntry(
        key="linear_rag.ner_max_span_chars",
        type="int",
        default=_LINEAR_RAG_DEFAULTS.ner_max_span_chars,
        min=20,
        max=200,
        group="linear_rag",
        description=(
            "Raw-character cap on GLiNER output spans. Bracketed surfaces "
            "(SKU markers, version tags) are kept regardless of length; "
            "non-bracketed spans above this are rejected as sentence-shape "
            "noise. Defensive ceiling — measured longest legitimate "
            "insurance / legal surface sits at ~50 chars."
        ),
    ),
    # ---------- linear_rag.graphml_flush_every (build persistence cadence) ----------
    ConfigEntry(
        key="linear_rag.graphml_flush_every",
        type="int",
        default=_LINEAR_RAG_DEFAULTS.graphml_flush_every,
        min=1,
        max=1000,
        group="linear_rag",
        description=(
            "How often LinearRAG.index() persists LinearRAG.graphml, in "
            "index() calls. 1 (default) = every doc — bit-identical to "
            "the per-file API builder (fresh instance per file). A "
            "persistent bulk driver (GraphIndexBuilder reuse_graph=True) "
            "sets this >1 so the O(V+E) graphml round-trip is amortised "
            "across docs instead of O(N²); such a driver must force a "
            "final flush_graphml() at end and before checkpoints."
        ),
    ),
    # ---------- linear_rag.evidence_fs_enabled (EvidenceFS emission) ----------
    ConfigEntry(
        key="linear_rag.evidence_fs_enabled",
        type="bool",
        default=_LINEAR_RAG_DEFAULTS.evidence_fs_enabled,
        group="linear_rag",
        description=(
            "Emit EvidenceFS — the shell-operable evidence filesystem — at the "
            "end of every LinearRAG.flush_all(), compiled from the exact-offset "
            "segmentation of each document's combined.md. On by default; turn "
            "off for text-benchmark / bulk paths with no combined.md corpus."
        ),
    ),
    # ---------- graph_explore.* (chain_entity tool runtime) ----------
    ConfigEntry(
        key="graph_explore.entity_lookup_min_sim",
        type="float",
        default=0.6,
        min=0.3,
        max=0.95,
        group="graph_explore",
        description=(
            "Input-resolution guard, not a retrieval-scoring threshold: a "
            "deliberately-named chain_entity `focus` anchor whose best "
            "embedding match is below this is flagged low-confidence and not "
            "used to seed the walk (prevents anchoring on a garbage match)."
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
            "Reserved: unused by the current 2-mode entity-lookup tool. "
            "Retained so persisted configs and the admin key still bind."
        ),
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
        key="prompt.compare",
        type="str",
        default=COMPARE_SYSTEM_PROMPT,
        max_length=8000,
        min_length=1,
        group="prompt",
        description="System prompt for the multi-product comparison workbench.",
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
            "name matches persisted overrides; current default surfaces "
            "semantically adjacent clauses, not fraud judgments."
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
