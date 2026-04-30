"""Name-keyed tool registry."""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        return self._tools.get(name)

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        return [tool.get_schema() for tool in self._tools.values()]

    def execute(
        self, name: str, context: "AgentContext", **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found", {"error": "tool_not_found"}
        try:
            return tool.execute(context, **kwargs)
        except Exception as e:
            return f"Error executing tool: {str(e)}", {"error": str(e)}

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())
