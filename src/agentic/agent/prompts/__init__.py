"""Agent prompts.

The system prompt is composed from labeled blocks so individual sections
can be swapped without rewriting the whole text. Public surface:

* :data:`SYSTEM_PROMPT` — already-assembled default for the acquisition-
  only agent loop. Wire this into ``BaseAgent(system_prompt=...)``.
* :func:`build_system_prompt` — same blocks, but lets a caller add an
  extra appendix (e.g. a benchmark- or business-mode footer).

Proof-state guidance is intentionally absent — the obligation / evidence
tools land in a later phase, and we will re-export a different system
prompt then. Until that phase, the agent runs as a free-form acquisition
loop with explicit citation requirements.
"""

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
    "TOOL_OVERVIEW",
    "SCOPE_CONVENTIONS",
    "STRATEGY_GUIDE",
    "RESPONSE_GUIDELINES",
    "build_system_prompt",
]
