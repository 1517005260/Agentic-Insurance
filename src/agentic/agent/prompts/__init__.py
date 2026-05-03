"""Agent prompts.

The system prompt is composed from labeled blocks so individual sections
can be swapped without rewriting the whole text. Public surface:

* :data:`SYSTEM_PROMPT` — assembled default for the acquisition-only
  baseline (Phase 3 BaseAgent).
* :data:`PROOF_SYSTEM_PROMPT` — Phase 4 ProofAgent prompt: same
  acquisition guidance plus the proof-gate contract.
* :func:`build_system_prompt` / :func:`build_proof_system_prompt` —
  reassemble the blocks with optional appendices.
"""

from agentic.agent.prompts.proof_system import (
    PROOF_SYSTEM_PROMPT,
    build_proof_system_prompt,
)
from agentic.agent.prompts.system import (
    RESPONSE_GUIDELINES,
    SCOPE_CONVENTIONS,
    STRATEGY_GUIDE,
    SYSTEM_PROMPT,
    TOOL_OVERVIEW,
    build_system_prompt,
)

__all__ = [
    "SYSTEM_PROMPT",
    "PROOF_SYSTEM_PROMPT",
    "TOOL_OVERVIEW",
    "SCOPE_CONVENTIONS",
    "STRATEGY_GUIDE",
    "RESPONSE_GUIDELINES",
    "build_system_prompt",
    "build_proof_system_prompt",
]
