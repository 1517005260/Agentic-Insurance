"""Build the LLM-facing :class:`GateView` snapshot.

A GateView is a read-only summary of plant state with diagnostics
attached: which obligations are open and why, which are closed, which
are challenged, and a small ring of recent claims. It is appended to
every state-changing tool result so the LLM can reason about
progress without polling.

This module owns the snapshot assembly and the per-obligation
helpers (``_diagnose_open``, ``_cursor_for``,
``_challenged_in_closure_cone``). Plant exposes 1-line forwarders so
test stubs and helpers that touch ``plant._diagnose_open`` etc.
continue to resolve.
"""

from typing import Any, Dict, List, Optional

from agentic.proof.closure import rules as closure_rules
from agentic.proof.state import transitions
from agentic.proof.types import (
    GateView,
    Obligation,
    ObligationStatus,
    ToolDiagnostic,
)


def _scope_to_dict(scope) -> Dict[str, Any]:
    return scope.to_dict()


def diagnose_open(plant: Any, obligation: Obligation) -> str:
    """Status-aware diagnosis. CHALLENGED / DECOMPOSED obligations
    cannot close even if Γ would otherwise hold; surfacing
    ``ready_to_close`` for them would mislead the LLM into a futile
    answer_finalize attempt."""
    if obligation.status == ObligationStatus.CHALLENGED:
        return "challenged_blocked"
    if obligation.status == ObligationStatus.DECOMPOSED:
        return "decomposed_pending_children"
    bound_claims = [
        plant.evidence.get_claim(b.claim_id)
        for b in plant.evidence.bindings_for_obligation(obligation.id)
    ]
    bound_claims = [c for c in bound_claims if c is not None]
    result = closure_rules.evaluate_closure(obligation, bound_claims, plant.inventory)
    if result.success:
        return "ready_to_close"
    return result.diagnostic or "unknown"


def cursor_for(plant: Any, obligation: Obligation) -> Optional[Dict[str, Any]]:
    if obligation.id not in plant.domain_maps:
        return None
    dm = plant.domain_maps.get(obligation.id)
    if dm is None:
        return None
    k, n = dm.k_of_n()
    return {"k": k, "n": n, "next": dm.cursor(limit=5)}


def challenged_in_closure_cone(plant: Any) -> List[Obligation]:
    """cone(R) := ⋃_{r ∈ R} ⋃_{a ∈ ancestors_inclusive(r)} descendants_inclusive(a).

    Closure cone definition lives in ``transitions.closure_cone`` so
    the formula has one source of truth.
    """
    required_ids = [
        o.id for o in (
            plant.obligations.active_required_open()
            + plant.obligations.active_required_closed()
        )
    ]
    cone_ids = transitions.closure_cone(plant.obligations, required_ids)
    return [
        o for o in plant.obligations.by_status(ObligationStatus.CHALLENGED)
        if o.id in cone_ids
    ]


def build_gate_view(plant: Any) -> GateView:
    diagnostics: List[ToolDiagnostic] = []
    open_payload: List[Dict[str, Any]] = []
    for o in plant.obligations.active_required_open():
        failure = diagnose_open(plant, o)
        diag = closure_rules.diagnose(failure, o.id)
        cursor = cursor_for(plant, o)
        if cursor is not None:
            diag.cursor = cursor
        # Diagnostics are kept on the dataclass for the test suite's
        # GateView.diagnostics consumers, but the LLM-facing payload
        # embeds the same info inline on each open_obligation so the
        # two arrays don't repeat the same failure_kind / suggested_*
        # strings (~30% byte saving per gate snapshot).
        diagnostics.append(diag)
        entry: Dict[str, Any] = {
            "id": o.id,
            "kind": o.spec.kind.value,
            "scope": _scope_to_dict(o.spec.scope),
            "predicate": o.spec.predicate.to_dict() if o.spec.predicate else None,
            "status": o.status.value,
            "is_root": o.is_root,
            "failure_kind": failure,
            "suggested_tools": list(diag.suggested_tools),
        }
        if diag.suggested_repair_kind is not None:
            entry["suggested_repair_kind"] = diag.suggested_repair_kind
        if diag.cursor is not None:
            entry["cursor"] = diag.cursor
        open_payload.append(entry)

    closed_payload: List[Dict[str, Any]] = []
    for o in plant.obligations.active_required_closed():
        closed_payload.append({
            "id": o.id,
            "kind": o.spec.kind.value,
            "value": o.closed_value,
            "used_claim_ids": list(o.closed_by),
        })

    challenged_payload: List[Dict[str, Any]] = []
    for o in plant.obligations.by_status(ObligationStatus.CHALLENGED):
        for ch in plant.challenges.open_for(o.id):
            challenged_payload.append({
                "obligation_id": o.id,
                "challenge_id": ch.id,
                "repair_kind": ch.repair_kind,
                "reason": ch.reason,
            })

    recent_claims = [
        {
            "id": c.id,
            "claim_type": c.claim_type.value,
            "scope": _scope_to_dict(c.scope),
            "unit_type": c.unit_type,
            "predicate": c.predicate.to_dict() if c.predicate else None,
            "positive_units": c.positive_units[:8],
            "negative_units_count": len(c.negative_units),
            "derivation": c.derivation,
        }
        for c in plant.evidence.recent_claims(limit=6)
    ]

    abstain_recommended = bool(open_payload or challenged_payload)
    abstain_reason = None
    if not plant.obligations.has_active_required():
        abstain_reason = "no_root_obligation"
        abstain_recommended = True
    return GateView(
        open_obligations=open_payload,
        closed_obligations=closed_payload,
        challenged_obligations=challenged_payload,
        diagnostics=diagnostics,
        recent_claims=recent_claims,
        abstain_recommended=abstain_recommended,
        abstain_reason=abstain_reason,
    )
