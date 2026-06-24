"""``entity_inspect`` — entity disambiguation / neighborhood expansion."""

from typing import Any, Dict, TYPE_CHECKING

from agentic.tools.acquisition._common import err
from agentic.tools.acquisition.graph_explore.base import _GraphToolBase

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class EntityInspectTool(_GraphToolBase):
    """Disambiguate / expand an entity's neighborhood (focus audit)."""

    @property
    def name(self) -> str:
        return "entity_inspect"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "entity_inspect",
                "description": (
                    "Inspect entity/entities: canonical name, alias members, "
                    "top pages it appears on, co-occurring entities. Use to "
                    "disambiguate (which 'John Smith') or expand a found "
                    "entity's neighborhood."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Entity name(s) or cluster_id(s) to inspect.",
                        },
                    },
                    "required": ["focus"],
                },
            },
        }

    def execute(self, context: "AgentContext", **kwargs):
        focus_raw = kwargs.get("focus") or []
        if isinstance(focus_raw, str):
            focus_raw = [focus_raw]
        focus = [f.strip() for f in focus_raw if isinstance(f, str) and f.strip()]
        if not focus:
            return err(
                "invalid_argument",
                "entity_inspect requires `focus`.",
                remediation="Pass `focus` = one or more entity names or cluster_ids to inspect.",
                valid_example={"focus": ["Christopher Nolan"]},
            ), {"error": "invalid_argument"}
        channel = self._channel
        if channel.graph is None or len(channel.entity_store) == 0:
            return err(
                "graph_unavailable",
                "Entity store is empty; index the corpus first.",
                remediation="The entity layer is not available in this corpus.",
            ), {"error": "graph_unavailable"}
        _, _, focus_audit = self._resolve_focus(focus)
        return self._run_focus_audit(context, focus, focus_audit)


# ----------------------------------------------------------- helpers
