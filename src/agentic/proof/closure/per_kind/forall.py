"""Γ_forall — every domain unit must be positive in some aligned scan."""
from typing import Sequence

from agentic.proof.closure.helpers import (
    _aligned_complete_scans,
    _section_scope_ok,
)
from agentic.proof.types import Claim, ClosureResult, Obligation
from storage.inventory_store import InventoryStore


def gamma_forall(
    obligation: Obligation,
    claims: Sequence[Claim],
    inventory: InventoryStore,
) -> ClosureResult:
    if obligation.spec.unit_type == "section":
        guard = _section_scope_ok(obligation.spec.scope, inventory)
        if guard is not None:
            return ClosureResult(success=False, diagnostic=guard)
    scans = _aligned_complete_scans(obligation, claims, inventory)
    if not scans:
        return ClosureResult(success=False, diagnostic="missing_scan")
    for c in scans:
        if not c.negative_units:
            return ClosureResult(success=True, value=True, used_claim_ids=[c.id])
    return ClosureResult(success=False, diagnostic="counterexample_found")
