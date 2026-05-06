"""Agent loop, tool registry, and proof-gate algorithms."""
from agentic.agent.base import BaseAgent
from agentic.agent.factory import build_default_agent, build_proof_agent
from agentic.agent.proof_agent import ProofAgent, ProofRunResult
from agentic.core.config import Config
from agentic.core.context import AgentContext
from agentic.tools.base import BaseTool
from agentic.tools.registry import ToolRegistry

__all__ = [
    "BaseAgent",
    "ProofAgent",
    "ProofRunResult",
    "build_default_agent",
    "build_proof_agent",
    "Config",
    "AgentContext",
    "BaseTool",
    "ToolRegistry",
    "__version__",
]
