"""Γ_negation — sealed scope, no positive unit anywhere in the domain."""
from typing import Sequence

from agentic.proof.closure.helpers import (
    _aligned_complete_scans,
    _is_sealed,
    _section_scope_ok,
)
from agentic.proof.types import Claim, ClosureResult, Obligation
from storage.inventory_store import InventoryStore


def gamma_negation(
    obligation: Obligation,
    claims: Sequence[Claim],
    inventory: InventoryStore,
) -> ClosureResult:
    if not _is_sealed(obligation):
        return ClosureResult(success=False, diagnostic="unsealed_scope")
    if obligation.spec.unit_type == "section":
        guard = _section_scope_ok(
            obligation.spec.scope, inventory, require_high=obligation.is_root
        )
        if guard is not None:
            return ClosureResult(success=False, diagnostic=guard)
    scans = _aligned_complete_scans(obligation, claims, inventory)
    if not scans:
        return ClosureResult(success=False, diagnostic="missing_scan")
    for c in scans:
        if not c.positive_units:
            return ClosureResult(success=True, value=True, used_claim_ids=[c.id])
    return ClosureResult(
        success=False,
        diagnostic="positive_witness_found",
    )
