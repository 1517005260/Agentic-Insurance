"""Agent prompts assembled from labeled blocks so individual sections can be
swapped without rewriting the whole text."""

from agentic.agent.prompts.system import (
    RESPONSE_GUIDELINES,
    SCOPE_CONVENTIONS,
    STRATEGY_GUIDE,
    SYSTEM_PROMPT,
    TOOL_OVERVIEW,
    build_system_prompt,
)

from agentic.agent.prompts.proof_system import (
    PROOF_SYSTEM_PROMPT,
    build_proof_system_prompt,
)

from agentic.agent.prompts.graph_system import (
    GRAPH_SYSTEM_PROMPT,
    build_graph_system_prompt,
)

from agentic.agent.prompts.rag_business import RAG_BUSINESS_SYSTEM_PROMPT

from agentic.agent.prompts.web_system import (
    WEB_AGENT_SYSTEM_PROMPT,
    WEB_RAG_SYSTEM_PROMPT,
)

from agentic.agent.prompts.insurance import (
    CLAIM_CHECK_SYSTEM_PROMPT,
    COMPARE_SYSTEM_PROMPT,
    EXCLUSION_AUDIT_SYSTEM_PROMPT,
    FRAUD_PPR_SYSTEM_PROMPT,
    POLICY_CALC_SYSTEM_PROMPT,
    RECOMMEND_SYSTEM_PROMPT,
    RISK_PREDICT_SYSTEM_PROMPT,
)

__all__ = [
    "SYSTEM_PROMPT",
    "TOOL_OVERVIEW",
    "SCOPE_CONVENTIONS",
    "STRATEGY_GUIDE",
    "RESPONSE_GUIDELINES",
    "build_system_prompt",
    "PROOF_SYSTEM_PROMPT",
    "build_proof_system_prompt",
    "GRAPH_SYSTEM_PROMPT",
    "build_graph_system_prompt",
    "RAG_BUSINESS_SYSTEM_PROMPT",
    "WEB_AGENT_SYSTEM_PROMPT",
    "WEB_RAG_SYSTEM_PROMPT",
    "CLAIM_CHECK_SYSTEM_PROMPT",
    "COMPARE_SYSTEM_PROMPT",
    "EXCLUSION_AUDIT_SYSTEM_PROMPT",
    "FRAUD_PPR_SYSTEM_PROMPT",
    "POLICY_CALC_SYSTEM_PROMPT",
    "RECOMMEND_SYSTEM_PROMPT",
    "RISK_PREDICT_SYSTEM_PROMPT",
]
