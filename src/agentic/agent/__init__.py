"""Agent loops and factories.

* :class:`BaseAgent` — Phase 3 acquisition-only baseline. Builds via
  :func:`build_default_agent`.
* :class:`ProofAgent` — Phase 4 gate-controlled agent that goes through
  the proof-obligation gate. Builds via :func:`build_proof_agent`.
"""

from agentic.agent.base import BaseAgent
from agentic.agent.factory import build_default_agent, build_proof_agent
from agentic.agent.proof_agent import ProofAgent

__all__ = ["BaseAgent", "ProofAgent", "build_default_agent", "build_proof_agent"]
