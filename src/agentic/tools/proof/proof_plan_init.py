"""``proof_plan_init`` — produce the initial flat required obligations.

The planner sits at the trusted-tool boundary. It uses a small LLM to
translate the question into typed obligations; the kernel checks
structural shape and drops anything malformed. ``proof_plan_init`` is
expected to be called once at the top of a run.
"""

import json
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from agentic.tools.acquisition._common import err, ok
from agentic.tools.base import BaseTool
from agentic.tools.proof.planner import propose_initial_obligations

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.closure.session import ProofSession


class ProofPlanInitTool(BaseTool):
    def __init__(self, session: "ProofSession") -> None:
        self._session = session

    @property
    def name(self) -> str:
        return "proof_plan_init"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "proof_plan_init",
                "description": (
                    "Translate the user's question into the flat list of required "
                    "obligations the gate will enforce. Call this once at the start "
                    "of a run. The planner is a trusted tool, NOT the closure "
                    "kernel: it may use a small LLM internally, but every produced "
                    "obligation is validated for structural shape before it is "
                    "accepted. Malformed obligations are dropped with a diagnostic."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The user's natural-language question.",
                        },
                        "corpus_hint": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional file_id list to bias the planner toward.",
                        },
                    },
                    "required": ["question"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        question: Optional[str] = None,
        corpus_hint: Optional[list] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        if not question or not str(question).strip():
            return (
                err(
                    "invalid_argument",
                    "`question` must be a non-empty string.",
                    remediation="Pass the user's question verbatim as `question`.",
                ),
                {"error": "invalid_argument"},
            )
        result = propose_initial_obligations(
            question=str(question),
            corpus_hint=list(corpus_hint or []),
        )
        # Idempotent: a second call replaces obligations + clears
        # candidate_gaps so the agent can re-plan after discovery
        # (toc / read / semantic_search) without leaving stale promotions.
        # CandidateGap is for proof-state correction, not main planning;
        # re-plan keeps that role separation.
        is_replan = bool(self._session.obligations)
        self._session.obligations.clear()
        self._session.candidate_gaps.clear()
        self._session.promoted_count = 0
        self._session.obligations.extend(result.obligations)

        diagnostics = [_diagnostic_summary(d) for d in result.diagnostics]
        if not result.obligations:
            return (
                ok(
                    "ProofPlanInit",
                    replan=is_replan,
                    obligations=[],
                    discovery_diagnostics=diagnostics,
                    next_step=(
                        "Planner produced no contract-valid obligation. "
                        "Discover scope/wording with list_files + toc + semantic_search, "
                        "then call proof_plan_init again with corpus_hint=[file_ids]. "
                        "If the question is unanswerable, propose a wrong_kind gap."
                    ),
                ),
                {"error": None, "obligations_count": 0, "discovery_only": True},
            )
        return (
            ok(
                "ProofPlanInit",
                replan=is_replan,
                obligations=[_obligation_summary(o) for o in result.obligations],
                discovery_diagnostics=diagnostics,
            ),
            {"error": None, "obligations_count": len(result.obligations), "replan": is_replan},
        )


def _diagnostic_summary(d) -> Dict[str, Any]:
    return {"code": d.code, "detail": d.detail, "hint": d.hint}


def _obligation_summary(o) -> Dict[str, Any]:
    return {
        "id": o.id,
        "kind": o.kind,
        "scope": {
            "file_ids": list(o.scope.file_ids),
            "section_ids": list(o.scope.section_ids) if o.scope.section_ids else None,
            "canonical_scope_id": o.scope.canonical_scope_id,
        },
        "unit_type": o.unit_type,
        "predicate": {
            "name": o.predicate.name,
            "args": o.predicate.args_dict(),
            "canonical_id": o.predicate.canonical_id,
        },
        "field": o.field,
        "score_field": o.score_field,
        "required": o.required,
        "status": o.status,
    }
