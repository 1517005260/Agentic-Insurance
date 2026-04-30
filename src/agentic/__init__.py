"""Agent loop, tools, and (future) algorithm code.

Engineering scaffolding lives in sibling top-level packages:
``model_client`` (HTTP clients), ``storage`` (page assets / evidence stores),
``paddle_ocr`` (PDF ingestion), ``config`` (env + constants).
"""

__version__ = "0.1.0"

from agentic.agent.base import BaseAgent
from agentic.core.config import Config
from agentic.core.context import AgentContext
from agentic.tools.base import BaseTool
from agentic.tools.registry import ToolRegistry

__all__ = [
    "BaseAgent",
    "Config",
    "AgentContext",
    "BaseTool",
    "ToolRegistry",
    "__version__",
]
