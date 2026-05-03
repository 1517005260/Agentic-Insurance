"""``obligation_challenge`` — open a mechanically dischargeable challenge.

Challenges are the only way to repair an obligation. The agent files a
challenge against an OPEN or DECOMPOSED obligation citing already-
ingested observations; the plant moves the obligation to CHALLENGED
and refuses to close it (or its descendants) until the postcondition
is met.

Six repair_kinds. ``predicate_mismatch`` is FORBIDDEN on the root —
use ``wrong_question_kind`` only in the pre-proof window (before any
required obligation has left OPEN), capped to once per session.
"""
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from pydantic import ValidationError

from agentic.proof import Plant
from agentic.proof.schema import (
    ObligationChallengeArgsModel,
    compose_tool_description,
    flatten_refs,
    to_envelope,
)
from agentic.tools.base import BaseTool
from agentic.tools.proof._common import render

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


_DESCRIPTION = (
    "Open a mechanically dischargeable challenge against an "
    "obligation. The plant moves the obligation to CHALLENGED "
    "and refuses to close it (or its descendants) until the "
    "challenge's postcondition is met."
    "\n\n"
    "Repair kinds:\n"
    "- scope_too_narrow / scope_too_broad: scope must be "
    "widened or narrowed; resolved by a replacement "
    "obligation_create with discharges_challenge.\n"
    "- predicate_mismatch: predicate is wrong; resolved by "
    "replacement obligation. FORBIDDEN on root (use "
    "wrong_question_kind for that).\n"
    "- missing_subobligation: parent should be decomposed; "
    "resolved by obligation_decompose with "
    "discharges_challenge.\n"
    "- wrong_question_kind: only on root, only in pre-proof "
    "window (before any required obligation leaves OPEN), "
    "capped at once per session."
    "\n\n"
    "evidence_ids must reference observations already on file."
)


class ObligationChallengeTool(BaseTool):
    def __init__(self, plant: Plant):
        self._plant = plant

    @property
    def name(self) -> str:
        return "obligation_challenge"

    def get_schema(self) -> Dict[str, Any]:
        parameters = flatten_refs(
            ObligationChallengeArgsModel.model_json_schema(mode="serialization")
        )
        return {
            "type": "function",
            "function": {
                "name": "obligation_challenge",
                "description": compose_tool_description(
                    ObligationChallengeArgsModel, _DESCRIPTION
                ),
                "parameters": parameters,
            },
        }

    def execute(
        self,
        context: "AgentContext",
        obligation_id: Optional[str] = None,
        repair_kind: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None,
        reason: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        from agentic.tools.acquisition._common import err
        try:
            args = ObligationChallengeArgsModel.model_validate(
                {
                    "obligation_id": obligation_id,
                    "repair_kind": repair_kind,
                    "evidence_ids": evidence_ids or [],
                    "reason": reason or "",
                }
            )
        except ValidationError as exc:
            from agentic.tools.proof._common import reject_validation
            return reject_validation(self._plant, exc, ObligationChallengeArgsModel)
        result = self._plant.handle_obligation_challenge(**args.to_domain())
        log: Dict[str, Any] = {
            "ok": result.ok,
            "obligation_id": args.obligation_id,
            "repair_kind": args.repair_kind,
        }
        if result.ok:
            log["challenge_id"] = result.payload.get("challenge_id")
        else:
            log["error"] = (result.error or {}).get("code")
        context.add_retrieval_log(tool_name=self.name, tokens=0, metadata=log)
        return render(result, "ObligationChallenge"), log
