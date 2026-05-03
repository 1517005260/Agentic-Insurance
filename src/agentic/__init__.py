"""Agent loop, tools, and (future) algorithm code.

Engineering scaffolding lives in sibling top-level packages:
``model_client`` (HTTP clients), ``storage`` (page assets / evidence stores),
``paddle_ocr`` (PDF ingestion), ``config`` (env + constants).
"""
from agentic.agent.base import BaseAgent
from agentic.agent.proof_agent import ProofAgent
from agentic.agent.factory import build_default_agent, build_proof_agent
from agentic.core.config import Config
from agentic.core.context import AgentContext
from agentic.tools.base import BaseTool
from agentic.tools.registry import ToolRegistry

__all__ = [
    "BaseAgent",
    "ProofAgent",
    "build_default_agent",
    "build_proof_agent",
    "Config",
    "AgentContext",
    "BaseTool",
    "ToolRegistry",
    "__version__",
]
