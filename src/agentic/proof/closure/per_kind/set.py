"""Γ_set — return the sorted set {u : predicate(u)} from one aligned scan."""
from typing import Sequence

from agentic.proof.closure.helpers import (
    _aligned_complete_scans,
    _section_scope_ok,
)
from agentic.proof.types import Claim, ClosureResult, Obligation
from storage.inventory_store import InventoryStore


def gamma_set(
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
    chosen = scans[0]
    return ClosureResult(
        success=True,
        value=sorted(chosen.positive_units),
        used_claim_ids=[chosen.id],
    )
