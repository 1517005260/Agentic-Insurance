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

__all__ = [
    "SYSTEM_PROMPT",
    "TOOL_OVERVIEW",
    "SCOPE_CONVENTIONS",
    "STRATEGY_GUIDE",
    "RESPONSE_GUIDELINES",
    "build_system_prompt",
    "PROOF_SYSTEM_PROMPT",
    "build_proof_system_prompt",
]
