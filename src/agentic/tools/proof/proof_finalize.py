"""``proof_finalize`` — the only path to a user-visible answer.

Single source of certification. Always returns CERTIFIED or ABSTAIN
in strict mode. The plant runs closure first; the gate then walks
the obligation list and certifies iff every required obligation is
CLOSED.

Draft-numeric audit:

* Every quantity in ``draft_text`` must be *backed* — present in some
  closed_value or in the verbatim cited span of an ingested claim.
  A computed-but-uncited number (e.g. 90,000 × 27% = 24,300 written
  into the draft with nothing attesting 24,300) is rejected with
  ``draft_unbacked_numeric_tokens``.
* Quantities are extracted as ``(kind, canonical_value)`` pairs via
  ``predicates.extract_quantities`` so a draft "27%" is NOT considered
  backed by a claim mentioning a bare "27" (different kinds).
* Backing is exact-value: 24,300 backs 24,300 only — no range, no
  derivation. The kernel does not verify arithmetic.

Limitations (intentional — do not stretch the gate to cover these):

* Bare 1- and 2-digit unit-less integers (page numbers, list ranks,
  "5-year pay") are not extracted as quantities. This is by design:
  attestation of every descriptive numeral would force the LLM to
  cite a span for "page 15", which is structural noise. Soundness
  bound: a draft can put trivial integers freely; load-bearing
  numbers (with separators, decimals, ≥100, or a unit) are audited.
* The audit checks numeric quantities only. Spelled-out numbers
  ("twenty-seven percent"), rounded forms ("about 24k"), and
  unit-conversion equivalents ("0.27" vs "27%") are NOT auto-
  recognised as equivalent. The LLM should write the same surface
  form the cited span uses.
* Argument arithmetic (90,000 × 27% = 24,300) is the LLM's
  responsibility; the gate certifies the inputs and rejects the
  result unless an explicit ValueClaim anchors it. This matches the
  small-kernel philosophy: the kernel does not host a calculator.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from agentic.closure.finalize import (
    Abstain,
    Certified,
    Continue,
    KernelInvariantError,
    try_finalize,
)
from agentic.closure.predicates import extract_quantities, is_load_bearing_quantity
from agentic.tools.acquisition._common import err, ok
from agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.closure.session import ProofSession


class ProofFinalizeTool(BaseTool):
    def __init__(self, session: "ProofSession") -> None:
        self._session = session

    @property
    def name(self) -> str:
        return "proof_finalize"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "proof_finalize",
                "description": (
                    "Ask the gate to certify. Strict mode: returns either "
                    "CERTIFIED with a composed answer or ABSTAIN with the "
                    "open obligations and their diagnostics. The optional "
                    "`draft_text` is appended verbatim to the certified "
                    "answer. Every load-bearing number in draft_text must "
                    "be backed by either a closed_value or the verbatim "
                    "span of an ingested claim — uncited computed numbers "
                    "(e.g. 90,000 * 27% = 24,300 written without a claim "
                    "that anchors 24,300) are rejected. Pass "
                    "cited_claim_ids to limit the citations footer to a "
                    "chosen subset."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "draft_text": {
                            "type": "string",
                            "description": "Optional prose paragraph to append to the certified answer.",
                        },
                        "cited_claim_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional whitelist of claim ids to surface in citations.",
                        },
                    },
                    "required": [],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        draft_text: Optional[str] = None,
        cited_claim_ids: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        if cited_claim_ids:
            unknown = [
                cid for cid in cited_claim_ids
                if self._session.find_claim(cid) is None
            ]
            if unknown:
                return (
                    err(
                        "unknown_claim_id",
                        "cited_claim_ids must reference existing claims.",
                        unknown=unknown,
                    ),
                    {"error": "unknown_claim_id"},
                )

        self._session.plant.run_closure(
            self._session.obligations,
            self._session.claims,
        )

        try:
            result = try_finalize(
                self._session.obligations,
                self._session.claims,
                self._session.inventory,
                self._session.budget,
                draft_text=draft_text,
            )
        except KernelInvariantError as exc:
            return (
                err("kernel_invariant_error", str(exc)),
                {"error": "kernel_invariant_error"},
            )

        if isinstance(result, Certified):
            if draft_text:
                missing_closed, unbacked = _draft_numeric_audit(
                    result,
                    self._session.claims,
                    draft_text,
                )
                if missing_closed:
                    return (
                        err(
                            "draft_missing_numeric_tokens",
                            "Every numeric closed_value must appear in draft_text.",
                            missing=missing_closed,
                        ),
                        {"error": "draft_missing_numeric_tokens"},
                    )
                if unbacked:
                    return (
                        err(
                            "draft_unbacked_numeric_tokens",
                            "Numbers in draft_text must be backed by a closed_value "
                            "or the verbatim span of an ingested claim.",
                            unbacked=unbacked,
                            remediation="Either remove the unbacked number, ingest a "
                            "ValueClaim/WitnessClaim whose cited span contains it, "
                            "or rephrase the draft to avoid stating an uncited number.",
                        ),
                        {"error": "draft_unbacked_numeric_tokens"},
                    )
            return (
                ok(
                    "ProofFinalizeResult",
                    decision="CERTIFIED",
                    answer=result.answer,
                    closed_obligations=[_summary(s) for s in result.closed_obligations],
                ),
                {"error": None, "decision": "CERTIFIED"},
            )

        if isinstance(result, Abstain):
            return (
                ok(
                    "ProofFinalizeResult",
                    decision="ABSTAIN",
                    reason=result.reason,
                    answer=_compose_abstain(result),
                    open_obligations=[_summary(s) for s in result.open_obligations],
                    closed_obligations=[_summary(s) for s in result.closed_obligations],
                ),
                {"error": None, "decision": "ABSTAIN"},
            )

        assert isinstance(result, Continue)
        return (
            ok(
                "ProofFinalizeResult",
                decision="CONTINUE",
                reason=result.reason,
                open_obligations=[_summary(s) for s in result.open_obligations],
            ),
            {"error": None, "decision": "CONTINUE"},
        )


def _summary(s) -> Dict[str, Any]:
    return {
        "id": s.id,
        "kind": s.kind,
        "scope": s.canonical_scope_id,
        "unit_type": s.unit_type,
        "predicate": s.canonical_predicate_id,
        "status": s.status,
        "closed_value": s.closed_value,
        "closed_by": list(s.closed_by),
        "failure_kind": s.failure_kind,
        "diagnostic_data": s.diagnostic_data,
    }


# Closed_value dicts (argmax) carry both identifiers and a load-bearing
# score; only these keys feed the draft audit.
_VALUE_KEYS_OF_INTEREST: frozenset[str] = frozenset({"score", "value"})


def _quantities_from_value(value: Any) -> List[Tuple[str, str]]:
    """Project a closed_value into (kind, canonical) quantities.

    Strict rules (closed_values are kernel-certified facts):

    * int/float → emit as numeric.
    * str → extract via the shared tokeniser (catches "27%", "USD 10,000").
    * dict → recurse only into keys that hold load-bearing values
      (``score`` for argmax). Identifier keys like ``unit_id`` / ``field``
      are skipped.
    * list (e.g. set closure value) → nothing; the answer is a unit-id list,
      not a numerical fact.
    """

    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [("numeric", str(int(value)) if float(value).is_integer() else repr(float(value)))]
    if isinstance(value, dict):
        out: List[Tuple[str, str]] = []
        for k, v in value.items():
            if k in _VALUE_KEYS_OF_INTEREST:
                out.extend(_quantities_from_value(v))
        return out
    if isinstance(value, list):
        return []
    return list(extract_quantities(str(value)))


def _draft_numeric_audit(
    result: Certified,
    claims: Any,
    draft_text: str,
) -> Tuple[List[str], List[str]]:
    """Return (closed_value quantities not in draft, draft load-bearing
    quantities not backed). The two checks use different filters:
    closed_values must always appear (even bare "2"); unbacked check
    skips descriptive small integers like "p15"/"item 3".
    """

    draft_quants = extract_quantities(draft_text)
    draft_set = set(draft_quants)

    missing_closed: List[str] = []
    closed_quants: List[Tuple[str, str]] = []
    for s in result.closed_obligations:
        for kind, canonical in _quantities_from_value(s.closed_value):
            closed_quants.append((kind, canonical))
            if (kind, canonical) not in draft_set:
                missing_closed.append(_render(kind, canonical))

    backing: set[Tuple[str, str]] = set(closed_quants)
    for c in claims:
        cite = getattr(c, "citation", None)
        if cite is None or not getattr(cite, "span", ""):
            continue
        backing |= set(extract_quantities(cite.span))

    unbacked: List[str] = []
    seen: set[Tuple[str, str]] = set()
    for kind, canonical in draft_quants:
        if not is_load_bearing_quantity(kind, canonical):
            continue
        if (kind, canonical) in backing:
            continue
        if (kind, canonical) in seen:
            continue
        seen.add((kind, canonical))
        unbacked.append(_render(kind, canonical))
    return missing_closed, unbacked


def _render(kind: str, canonical: str) -> str:
    return f"{canonical}%" if kind == "percent" else canonical


def _compose_abstain(result: Abstain) -> str:
    lines = [f"Abstain: {result.reason}"]
    for s in result.open_obligations:
        lines.append(
            f"- {s.id} ({s.kind}) {s.failure_kind or 'open'}"
        )
    return "\n".join(lines) + "\n"
