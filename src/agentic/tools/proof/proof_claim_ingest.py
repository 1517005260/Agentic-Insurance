"""``proof_claim_ingest`` — obligation-centric.

The LLM provides only an ``obligation_id`` plus an ``observation_id``
(and the unit/span/value bits the contract requires). The claim
shape, predicate, and field are derived from the obligation via
``closure.contract.contract_for(obligation.kind)``. Plant verifies
the observation matches the contract's accepted source types.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from agentic.closure.contract import claim_for, contract_for
from agentic.closure.plant import ErrorEnvelope
from agentic.tools.acquisition._common import err, ok
from agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.closure.session import ProofSession


class ProofClaimIngestTool(BaseTool):
    def __init__(self, session: "ProofSession") -> None:
        self._session = session

    @property
    def name(self) -> str:
        return "proof_claim_ingest"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "proof_claim_ingest",
                "description": (
                    "Submit evidence against an obligation. Plant decides "
                    "the claim shape from the obligation kind:\n"
                    "* set/count/forall/negation: prefer `proof_scan` "
                    "(handles ScanClaim end-to-end). Direct ingest works "
                    "via observation_id of a pattern_search result.\n"
                    "* exists: WitnessClaim from a read or pattern_search "
                    "observation; pass observation_id, unit_id, polarity.\n"
                    "* lookup (extracted): ValueClaim from a read; pass "
                    "observation_id, unit_id, value, value_type, span.\n"
                    "* lookup (DERIVED, computed answer): pass operation "
                    "(sum/percent_of/...) + input_claim_ids — kernel "
                    "re-runs the arithmetic over those ValueClaims. No "
                    "observation_id / unit_id / span needed.\n"
                    "* argmax: ValueClaim per unit; pass observation_id, "
                    "unit_id, value, span."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "obligation_id": {"type": "string"},
                        "observation_id": {
                            "type": "string",
                            "description": "Observation backing the claim (extracted path).",
                        },
                        "unit_id": {"type": "string"},
                        "polarity": {
                            "type": "string",
                            "enum": ["positive", "negative"],
                            "description": "WitnessClaim only.",
                        },
                        "value": {"description": "ValueClaim only."},
                        "value_type": {
                            "type": "string",
                            "enum": ["numeric", "percentage", "date_iso", "text", "integer_count"],
                            "description": "ValueClaim only.",
                        },
                        "span": {"type": "string", "description": "Verbatim citation span (extracted path)."},
                        "span_start": {"type": "integer"},
                        "span_end": {"type": "integer"},
                        "operation": {
                            "type": "string",
                            "enum": list(["sum", "product", "percent_of", "difference",
                                          "quotient", "max", "min"]),
                            "description": "DerivedValueClaim only — picks a whitelisted operation.",
                        },
                        "input_claim_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "DerivedValueClaim only — closed ValueClaim ids the kernel re-runs the operation over.",
                        },
                    },
                    "required": ["obligation_id"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        obligation_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        polarity: Optional[str] = None,
        value: Any = None,
        value_type: Optional[str] = None,
        span: Optional[str] = None,
        span_start: Optional[int] = None,
        span_end: Optional[int] = None,
        operation: Optional[str] = None,
        input_claim_ids: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        if not obligation_id:
            return _err("invalid_argument", "obligation_id is required.")
        obligation = self._session.find_obligation(str(obligation_id))
        if obligation is None:
            return _err("unknown_obligation", f"no obligation with id={obligation_id!r}")

        spec = contract_for(obligation.kind)
        plant = self._session.plant

        # Derived ValueClaim path — kernel-side arithmetic verification.
        # PCN-style claim-bound numerics (arXiv:2509.06902) + PoT-style
        # separation of compute from reasoning (arXiv:2211.12588), with
        # the kernel as the executor.
        if operation is not None or input_claim_ids:
            if obligation.kind != "lookup":
                return _err(
                    "derived_not_applicable",
                    "DerivedValueClaim only supported for lookup; argmax requires per-unit extracted values.",
                )
            if not (operation and input_claim_ids and value_type is not None):
                return _err(
                    "invalid_argument",
                    "Derived path needs operation + input_claim_ids + value + value_type.",
                )
            target_field = obligation.field
            if not target_field:
                return _err("invalid_argument", f"obligation {obligation.id} missing field for derived ValueClaim.")
            claim = plant.ingest_derived_value_claim(
                field=str(target_field),
                operation=str(operation),
                input_claim_ids=tuple(str(x) for x in input_claim_ids),
                value=value,
                value_type=str(value_type),
                all_claims=self._session.claims,
            )
        else:
            if not observation_id:
                return _err("invalid_argument", "observation_id required for non-derived claim.")
            target_shape = _resolve_claim_shape(spec.claim_for_evidence, polarity, unit_id, span)

            if target_shape == "ScanClaim":
                claim = plant.ingest_scan_claim(
                    observation_id=str(observation_id),
                    expected_unit_type=obligation.unit_type,
                )
            elif target_shape == "WitnessClaim":
                if not unit_id or polarity not in {"positive", "negative"}:
                    return _err("invalid_argument", "WitnessClaim requires unit_id and polarity.")
                claim = plant.ingest_witness_claim(
                    observation_id=str(observation_id),
                    unit_id=str(unit_id),
                    polarity=polarity,  # type: ignore[arg-type]
                    predicate=obligation.predicate,
                    expected_unit_type=obligation.unit_type,
                    span=span,
                    span_start=span_start,
                    span_end=span_end,
                )
            elif target_shape == "ValueClaim":
                if not (unit_id and value_type and span is not None):
                    return _err("invalid_argument", "ValueClaim requires unit_id, value, value_type, span.")
                target_field = obligation.field if obligation.kind == "lookup" else obligation.score_field
                if not target_field:
                    return _err("invalid_argument", f"obligation {obligation.id} has no field/score_field bound.")
                claim = plant.ingest_value_claim(
                    observation_id=str(observation_id),
                    unit_id=str(unit_id),
                    field=str(target_field),
                    value=value,
                    value_type=str(value_type),
                    span=str(span),
                    expected_unit_type=obligation.unit_type,
                    span_start=span_start,
                    span_end=span_end,
                )
            else:
                return _err("invalid_argument", f"unable to dispatch claim for kind={obligation.kind}")

        if isinstance(claim, ErrorEnvelope):
            ctx = dict(claim.context)
            # When scan / unit-type alignment fails, surface the obligation's
            # exact scope and unit_type so the agent can call proof_scan
            # without trial and error.
            if claim.code in {"scan_coverage_mismatch", "unit_type_mismatch"}:
                ctx.setdefault("expected_scope", {
                    "file_ids": list(obligation.scope.file_ids),
                    "section_ids": list(obligation.scope.section_ids) if obligation.scope.section_ids else None,
                })
                ctx.setdefault("expected_unit_type", obligation.unit_type)
                ctx.setdefault(
                    "recommended_action",
                    f"call proof_scan(obligation_id='{obligation.id}') for guaranteed canonical alignment.",
                )
            return err(claim.code, claim.remediation, **ctx), {"error": claim.code}

        claim = dataclasses.replace(claim, id=plant.mint_id("claim"))
        self._session.claims.append(claim)
        plant.run_closure(self._session.obligations, self._session.claims)
        status = _obligation_status(self._session.obligations)
        return (
            ok(
                "ProofClaimIngestResult",
                claim_id=claim.id,
                claim_type=claim.claim_type,
                obligation_status=status,
                must_finalize_next=status["must_finalize_next"],
            ),
            {"error": None, "must_finalize_next": status["must_finalize_next"]},
        )


def _resolve_claim_shape(
    allowed: tuple[str, ...],
    polarity: Optional[str],
    unit_id: Optional[str],
    span: Optional[str],
) -> Optional[str]:
    """Pick the claim shape. Single-shape kinds: trivial. forall/negation
    can be either Witness (unit_id+polarity) or Scan (no unit_id).
    """
    if len(allowed) == 1:
        return allowed[0]
    if "WitnessClaim" in allowed and unit_id and polarity:
        return "WitnessClaim"
    if "ScanClaim" in allowed:
        return "ScanClaim"
    return None


def _err(code: str, message: str) -> Tuple[str, Dict[str, Any]]:
    return err(code, message), {"error": code}


def _obligation_status(obligations: List) -> Dict[str, Any]:
    closed = [o.id for o in obligations if o.status == "CLOSED"]
    open_required = [
        {
            "id": o.id,
            "kind": o.kind,
            "unit_type": o.unit_type,
            "field": getattr(o, "field", None),
            "score_field": getattr(o, "score_field", None),
            "failure_kind": o.failure_kind,
            "diagnostic_data": o.diagnostic_data,
        }
        for o in obligations
        if o.required and o.status != "CLOSED"
    ]
    must_finalize = (len(open_required) == 0) and any(o.required for o in obligations)
    return {
        "summary": f"{len(closed)} closed / {len(open_required)} open required",
        "closed_obligation_ids": closed,
        "open_required": open_required,
        "must_finalize_next": must_finalize,
    }
