"""Static cross-check between schema defaults and their algorithm sources.

Most of the registered config keys take their ``default`` from a live
import (``RAGConfig().rrf_k``, ``SYSTEM_PROMPT``, …) — those will never
drift. The agent factory's ``max_loops`` / ``max_token_budget`` ints
have no good single import site, so :mod:`config.config_store.schema`
mirrors them; this test pins the mirror to the factory signatures so a
factory change without a schema bump fails CI loudly.
"""
import inspect

from agentic.agent.factory import (
    build_default_agent,
    build_graph_agent,
    build_proof_agent,
    build_web_agent,
)
from agentic.agent.prompts import (
    CLAIM_CHECK_SYSTEM_PROMPT,
    COMPARE_SYSTEM_PROMPT,
    EXCLUSION_AUDIT_SYSTEM_PROMPT,
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
from api.services.citation import DEFAULT_PREVIEW_CHARS
from config.config_store.schema import CONFIG_ENTRIES_BY_KEY
from config.rag import RAGConfig


def _factory_default(fn, name: str):
    return inspect.signature(fn).parameters[name].default


def test_rag_defaults_match_RAGConfig():
    cfg = RAGConfig()
    for key, attr in [
        ("rag.rrf_k", "rrf_k"),
        ("rag.rrf_top_m", "rrf_top_m"),
        ("rag.rerank_top_n", "rerank_top_n"),
        ("rag.answer_max_tokens", "answer_max_tokens"),
    ]:
        assert CONFIG_ENTRIES_BY_KEY[key].default == getattr(cfg, attr), key


def test_agent_factory_defaults_mirrored_in_schema():
    expectations = [
        ("agent.base.max_loops", build_default_agent, "max_loops"),
        ("agent.base.max_token_budget", build_default_agent, "max_token_budget"),
        ("agent.proof.max_loops", build_proof_agent, "max_loops"),
        ("agent.proof.max_token_budget", build_proof_agent, "max_token_budget"),
        ("agent.graph.max_loops", build_graph_agent, "max_loops"),
        ("agent.graph.max_token_budget", build_graph_agent, "max_token_budget"),
        ("agent.web.max_loops", build_web_agent, "max_loops"),
        ("agent.web.max_token_budget", build_web_agent, "max_token_budget"),
    ]
    for key, fn, param in expectations:
        assert (
            CONFIG_ENTRIES_BY_KEY[key].default == _factory_default(fn, param)
        ), f"{key} drifted from {fn.__name__}.{param}"


def test_prompt_defaults_match_module_constants():
    assert CONFIG_ENTRIES_BY_KEY["prompt.rag_business"].default is RAG_BUSINESS_SYSTEM_PROMPT
    assert CONFIG_ENTRIES_BY_KEY["prompt.base_agent"].default is SYSTEM_PROMPT
    assert CONFIG_ENTRIES_BY_KEY["prompt.proof_agent"].default is PROOF_SYSTEM_PROMPT
    assert CONFIG_ENTRIES_BY_KEY["prompt.graph_agent"].default is GRAPH_SYSTEM_PROMPT
    assert CONFIG_ENTRIES_BY_KEY["prompt.web_rag"].default is WEB_RAG_SYSTEM_PROMPT
    assert CONFIG_ENTRIES_BY_KEY["prompt.web_agent"].default is WEB_AGENT_SYSTEM_PROMPT
    assert CONFIG_ENTRIES_BY_KEY["prompt.regulation"].default is REGULATION_SUMMARIZER_SYSTEM_PROMPT
    assert CONFIG_ENTRIES_BY_KEY["prompt.compare"].default is COMPARE_SYSTEM_PROMPT
    assert CONFIG_ENTRIES_BY_KEY["prompt.exclusion_audit"].default is EXCLUSION_AUDIT_SYSTEM_PROMPT
    assert CONFIG_ENTRIES_BY_KEY["prompt.recommend"].default is RECOMMEND_SYSTEM_PROMPT
    assert CONFIG_ENTRIES_BY_KEY["prompt.claim_check"].default is CLAIM_CHECK_SYSTEM_PROMPT
    assert CONFIG_ENTRIES_BY_KEY["prompt.policy_calc"].default is POLICY_CALC_SYSTEM_PROMPT


def test_citation_default_matches_module_constant():
    assert CONFIG_ENTRIES_BY_KEY["citation.preview_chars"].default == DEFAULT_PREVIEW_CHARS


def test_tavily_defaults():
    assert CONFIG_ENTRIES_BY_KEY["tavily.max_results"].default == 5
    assert CONFIG_ENTRIES_BY_KEY["tavily.search_depth"].default == "basic"
    assert "ia.org.hk" in CONFIG_ENTRIES_BY_KEY["tavily.include_domains_hk"].default
    assert "nfra.gov.cn" in CONFIG_ENTRIES_BY_KEY["tavily.include_domains_cn"].default


def test_entry_count_is_29():
    # rag/rerank/agent core (15) + agent.web (2) + tavily (4) + prompt (8) = 29.
    assert len(CONFIG_ENTRIES_BY_KEY) == 29
