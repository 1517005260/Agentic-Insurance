"""``answer_finalize`` — terminal answer-emission gate.

The plant returns one of three decisions:

* ``CERTIFIED`` — every active required obligation is CLOSED and no
  obligation in the closure cone is CHALLENGED. The draft text is
  accepted (with cited claims appended) as the final answer.
* ``ABSTAIN`` — budget exhausted with open or challenged required
  obligations. The agent must surface a graceful no-answer with the
  diagnostics returned in the gate snapshot.
* ``REJECT`` — required obligations remain open or the draft cites
  unbound claims. The agent may retry with a corrected draft.
"""
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from pydantic import ValidationError

from agentic.proof import Plant
from agentic.proof.schema import (
    AnswerFinalizeArgsModel,
    compose_tool_description,
    flatten_refs,
    to_envelope,
)
from agentic.tools.base import BaseTool
from agentic.tools.proof._common import render

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


_DESCRIPTION = (
    "Submit the draft answer to the proof gate. Returns:\n"
    "  - CERTIFIED + final_answer  when all required obligations\n"
    "    are CLOSED, no obligation in the closure cone is\n"
    "    CHALLENGED, every numeric closed_value appears as a\n"
    "    token in draft_text, and cited_claim_ids is a subset of the claims\n"
    "    that actually closed required obligations.\n"
    "  - REJECT  when the draft is premature (open or\n"
    "    challenged obligations) OR draft_text contradicts a\n"
    "    closed numeric value OR cited_claim_ids contains an\n"
    "    id that did not close any required obligation.\n"
    "  - ABSTAIN  when budget is exhausted.\n\n"
    "WHERE cited_claim_ids COME FROM:\n"
    "  - Every successful `evidence_ingest` returns "
    "    `claim_id` in its payload — collect these.\n"
    "  - Every `pattern_search(exhaustive=True)` that auto-"
    "    extracts a ScanClaim returns the staged ids on the "
    "    observation's `auto_extract_claim_ids` field.\n"
    "  - The `gate.closed` list on every state-changing tool "
    "    result also lists which claim ids closed which "
    "    obligations (`{obligation_id, used_claim_ids}`).\n\n"
    "Pass the union of those ids that closed your required "
    "obligations — typically one or two for an `exists` "
    "obligation, the auto-extracted ScanClaim id for `count` "
    "/ `set` / `forall`, and one WitnessClaim id per unit for "
    "`argmax`."
)


class AnswerFinalizeTool(BaseTool):
    def __init__(self, plant: Plant, *, budget_check=None):
        self._plant = plant
        self._budget_check = budget_check

    @property
    def name(self) -> str:
        return "answer_finalize"

    def get_schema(self) -> Dict[str, Any]:
        parameters = flatten_refs(
            AnswerFinalizeArgsModel.model_json_schema(mode="serialization")
        )
        return {
            "type": "function",
            "function": {
                "name": "answer_finalize",
                "description": compose_tool_description(
                    AnswerFinalizeArgsModel, _DESCRIPTION
                ),
                "parameters": parameters,
            },
        }

    def execute(
        self,
        context: "AgentContext",
        draft_text: Optional[str] = None,
        cited_claim_ids: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        from agentic.tools.acquisition._common import err
        try:
            args = AnswerFinalizeArgsModel.model_validate(
                {
                    "draft_text": draft_text,
                    "cited_claim_ids": cited_claim_ids or [],
                }
            )
        except ValidationError as exc:
            from agentic.tools.proof._common import reject_validation
            return reject_validation(self._plant, exc, AnswerFinalizeArgsModel)
        budget_exhausted = bool(self._budget_check() if self._budget_check else False)
        result = self._plant.handle_answer_finalize(
            **args.to_domain(),
            budget_exhausted=budget_exhausted,
        )
        decision = result.payload.get("decision", "REJECT" if not result.ok else "CERTIFIED")
        log: Dict[str, Any] = {
            "ok": result.ok,
            "decision": decision,
            "budget_exhausted": budget_exhausted,
        }
        if not result.ok and result.error is not None:
            log["error"] = result.error.get("code")
        context.add_retrieval_log(tool_name=self.name, tokens=0, metadata=log)
        return render(result, "AnswerFinalize"), log
