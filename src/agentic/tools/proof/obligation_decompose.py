"""``obligation_decompose`` — split a parent obligation into children.

Four registered rules:

* ``and_split``       — children share parent's scope/unit_type, their
                        predicates AND together to the parent's predicate.
* ``scope_partition`` — children share the parent's predicate, their
                        scopes form a disjoint partition of the parent's
                        inventory units.
* ``case_split``      — exactly two children differing in predicate
                        polarity (deferred validation in v1).
* ``map_over_domain`` — virtualised: parent's domain is enumerated and
                        children are materialised lazily on demand.
                        ``child_specs`` is ignored for this rule.
"""
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from pydantic import ValidationError

from agentic.proof import Plant
from agentic.proof.schema import (
    ObligationDecomposeArgsModel,
    compose_tool_description,
    flatten_refs,
    to_envelope,
)
from agentic.tools.base import BaseTool
from agentic.tools.proof._common import render

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


_DESCRIPTION = (
    "Decompose a parent obligation into children using a "
    "registered rule. The parent transitions to DECOMPOSED "
    "and auto-closes when all children close."
    "\n\n"
    "Rules:\n"
    "- and_split: children share parent's scope; their "
    "predicates AND together to the parent's predicate. "
    "At least one conjunct must be content-bearing.\n"
    "- scope_partition: children share parent's predicate; "
    "their scopes form a disjoint partition of the parent's "
    "inventory units.\n"
    "- case_split: exactly two children with opposite "
    "polarity.\n"
    "- map_over_domain: parent's domain is enumerated; "
    "children are materialised lazily on demand. Pass an "
    "empty child_specs list."
    "\n\n"
    "If this decompose discharges a missing_subobligation "
    "challenge, pass discharges_challenge."
)


class ObligationDecomposeTool(BaseTool):
    def __init__(self, plant: Plant):
        self._plant = plant

    @property
    def name(self) -> str:
        return "obligation_decompose"

    def get_schema(self) -> Dict[str, Any]:
        parameters = flatten_refs(
            ObligationDecomposeArgsModel.model_json_schema(mode="serialization")
        )
        return {
            "type": "function",
            "function": {
                "name": "obligation_decompose",
                "description": compose_tool_description(
                    ObligationDecomposeArgsModel, _DESCRIPTION
                ),
                "parameters": parameters,
            },
        }

    def execute(
        self,
        context: "AgentContext",
        parent_id: Optional[str] = None,
        rule_id: Optional[str] = None,
        child_specs: Optional[List[Dict[str, Any]]] = None,
        discharges_challenge: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        from agentic.tools.acquisition._common import err
        try:
            args = ObligationDecomposeArgsModel.model_validate(
                {
                    "parent_id": parent_id,
                    "rule_id": rule_id,
                    "child_specs": child_specs or [],
                    "discharges_challenge": discharges_challenge,
                }
            )
        except ValidationError as exc:
            from agentic.tools.proof._common import reject_validation
            return reject_validation(self._plant, exc, ObligationDecomposeArgsModel)
        result = self._plant.handle_obligation_decompose(**args.to_domain())
        log: Dict[str, Any] = {
            "ok": result.ok,
            "parent_id": args.parent_id,
            "rule_id": args.rule_id,
        }
        if result.ok:
            log["child_ids"] = result.payload.get("child_ids", [])
        else:
            log["error"] = (result.error or {}).get("code")
        context.add_retrieval_log(tool_name=self.name, tokens=0, metadata=log)
        return render(result, "ObligationDecompose"), log
