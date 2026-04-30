"""Tool ABC."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return the OpenAI function-call schema."""

    @abstractmethod
    def execute(self, context: "AgentContext", **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Run the tool and return ``(result_text, log_dict)``."""
