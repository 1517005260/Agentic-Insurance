"""``proof_gap_propose`` — bounded CandidateGap; never blocks finalize.

The LLM uses this tool when it suspects the certification contract
itself is defective (wrong kind, missing scope, ambiguous lookup, or
a missing predicate variant). Plant promotion is the only path that
turns a gap into a required obligation, and it is intentionally
small.
"""

import json
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from agentic.closure.candidate_gap import (
    MAX_CANDIDATE_GAPS,
    MAX_PROMOTED_GAPS,
    CandidateGap,
    EvidenceHint,
    accept_candidate_gap,
)
from agentic.closure.obligation import Obligation, PredicateRef, ScopeRef
from agentic.tools.acquisition._common import err, ok
from agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.closure.session import ProofSession


_VALID_KINDS = {
    "missing_scope",
    "wrong_kind",
    "predicate_variant_missing",
    "ambiguous_lookup",
}
_ALLOWED_OBLIGATION_KINDS = {
    "exists",
    "lookup",
    "count",
    "set",
    "forall",
    "negation",
    "argmax",
}
_ALLOWED_UNIT_TYPES = {"page", "passage", "table_row"}


class ProofGapProposeTool(BaseTool):
    def __init__(self, session: "ProofSession") -> None:
        self._session = session

    @property
    def name(self) -> str:
        return "proof_gap_propose"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "proof_gap_propose",
                "description": (
                    "Propose a CandidateGap suggesting that the current "
                    "certification contract may be defective. Does NOT block "
                    "finalization. Plant promotion is bounded: only gaps that "
                    "match the user's question or are canonical-equivalent to "
                    "an existing obligation may add a required obligation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": sorted(_VALID_KINDS),
                        },
                        "rationale": {
                            "type": "string",
                            "description": "≤200 chars. Why the contract may be defective.",
                        },
                        "proposed_obligation": {
                            "type": "object",
                            "description": "Optional Obligation spec (same shape proof_plan_init produces).",
                        },
                        "evidence_hint_unit_id": {
                            "type": "string",
                            "description": "Optional unit_id that motivated the proposal.",
                        },
                        "evidence_hint_predicate": {
                            "type": "object",
                            "description": "Optional alternative PredicateRef {name, args}.",
                        },
                        "priority": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "default": 3,
                        },
                    },
                    "required": ["kind", "rationale"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        kind: Optional[str] = None,
        rationale: Optional[str] = None,
        proposed_obligation: Optional[dict] = None,
        evidence_hint_unit_id: Optional[str] = None,
        evidence_hint_predicate: Optional[dict] = None,
        priority: int = 3,
    ) -> Tuple[str, Dict[str, Any]]:
        if not kind or kind not in _VALID_KINDS:
            return (
                err(
                    "invalid_argument",
                    f"`kind` must be one of {sorted(_VALID_KINDS)}.",
                    valid_example={"kind": "missing_scope", "rationale": "..."},
                ),
                {"error": "invalid_argument"},
            )
        if not rationale or not str(rationale).strip():
            return (
                err(
                    "invalid_argument",
                    "`rationale` must be a non-empty short string.",
                ),
                {"error": "invalid_argument"},
            )

        try:
            predicate_variant = (
                PredicateRef.build(
                    name=str(evidence_hint_predicate.get("name", "")),
                    args=evidence_hint_predicate.get("args") or {},
                )
                if evidence_hint_predicate
                else None
            )
        except (ValueError, TypeError) as exc:
            return (
                err(
                    "invalid_argument",
                    f"evidence_hint_predicate is malformed: {exc}",
                ),
                {"error": "invalid_argument"},
            )

        try:
            hint = EvidenceHint(
                rationale=str(rationale)[:200],
                unit_id=evidence_hint_unit_id,
                predicate_variant=predicate_variant,
            )
        except ValueError as exc:
            return (
                err("invalid_argument", str(exc)),
                {"error": "invalid_argument"},
            )

        try:
            obligation = (
                _materialise_proposed_obligation(proposed_obligation, len(self._session.candidate_gaps))
                if proposed_obligation
                else None
            )
        except ValueError as exc:
            return (
                err(
                    "invalid_argument",
                    f"proposed_obligation is malformed: {exc}",
                ),
                {"error": "invalid_argument"},
            )

        gap = CandidateGap(
            id=f"gap_{len(self._session.candidate_gaps) + 1:03d}",
            kind=kind,  # type: ignore[arg-type]
            proposed_obligation=obligation,
            evidence_hint=hint,
            priority=int(priority or 3),
        )

        accepted = accept_candidate_gap(gap, self._session.candidate_gaps)
        if accepted is None:
            return (
                ok(
                    "ProofGapProposeResult",
                    status="DISMISSED",
                    reason="active_gap_cap_reached",
                    cap=MAX_CANDIDATE_GAPS,
                    existing_gap_ids=[g.id for g in self._session.candidate_gaps],
                ),
                {"error": None},
            )

        if accepted is gap:
            self._session.candidate_gaps.append(gap)

        promoted = self._session.plant.validate_obligation_update(
            accepted,
            self._session.obligations,
            self._session.budget,
            promoted_so_far=self._session.promoted_count,
        )

        promoted_obligation_id: Optional[str] = None
        if promoted is not None:
            self._session.obligations.append(promoted)
            self._session.promoted_count += 1
            accepted.status = "PROMOTED"
            accepted.promoted_obligation_id = promoted.id
            promoted_obligation_id = promoted.id

        return (
            ok(
                "ProofGapProposeResult",
                gap_id=accepted.id,
                status=accepted.status,
                repeats=accepted.repeats,
                promoted_obligation_id=promoted_obligation_id,
                promoted_total=self._session.promoted_count,
                promoted_cap=MAX_PROMOTED_GAPS,
            ),
            {"error": None},
        )


def _materialise_proposed_obligation(spec: dict, gap_index: int) -> Obligation:
    if not isinstance(spec, dict):
        raise ValueError("proposed_obligation must be an object.")
    kind = str(spec.get("kind", "")).strip()
    if kind not in _ALLOWED_OBLIGATION_KINDS:
        raise ValueError(
            f"kind must be one of {sorted(_ALLOWED_OBLIGATION_KINDS)}; got {kind!r}."
        )
    unit_type = str(spec.get("unit_type", "")).strip()
    if unit_type not in _ALLOWED_UNIT_TYPES:
        raise ValueError(
            f"unit_type must be one of {sorted(_ALLOWED_UNIT_TYPES)}; got {unit_type!r}."
        )
    scope_raw = spec.get("scope") or {}
    if not isinstance(scope_raw, dict):
        raise ValueError("scope must be an object.")
    predicate_raw = spec.get("predicate") or {}
    if not isinstance(predicate_raw, dict):
        raise ValueError("predicate must be an object.")
    scope = ScopeRef.build(
        file_ids=tuple(scope_raw.get("file_ids") or ()),
        section_ids=tuple(scope_raw.get("section_ids") or ()) or None,
    )
    predicate = PredicateRef.build(
        name=str(predicate_raw.get("name", "")),
        args=predicate_raw.get("args") or {},
    )
    field = spec.get("field")
    if field is not None:
        field = str(field).strip() or None
    score_field = spec.get("score_field")
    if score_field is not None:
        score_field = str(score_field).strip() or None
    obligation = Obligation(
        id=f"o_gap_{gap_index + 1:03d}",
        kind=kind,  # type: ignore[arg-type]
        scope=scope,
        unit_type=unit_type,  # type: ignore[arg-type]
        predicate=predicate,
        required=bool(spec.get("required", True)),
        field=field,
        score_field=score_field,
    )
    # Run the same contract validator the planner uses, so a gap that
    # would fail validate_obligation never enters the session.
    from agentic.closure.contract import validate_obligation
    err_code = validate_obligation(obligation)
    if err_code is not None:
        raise ValueError(f"obligation rejected by contract: {err_code}")
    return obligation
