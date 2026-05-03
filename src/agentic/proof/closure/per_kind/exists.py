"""Γ_exists — one witness inside the obligation's domain is enough."""
from typing import Sequence

from agentic.proof import predicate as pr
from agentic.proof.closure.helpers import _scope_compatible, _scope_units
from agentic.proof.types import Claim, ClaimType, ClosureResult, Obligation
from storage.inventory_store import InventoryStore


def gamma_exists(
    obligation: Obligation,
    claims: Sequence[Claim],
    inventory: InventoryStore,
) -> ClosureResult:
    """Find any WitnessClaim that proves ``predicate(unit)`` for some
    unit inside the obligation's scope.

    Witnesses can come from any inventory level whose unit_type matches
    the obligation's. If a claim's ``positive_units`` is non-empty and
    the predicate entails the obligation's, that's enough.
    """
    domain = set(_scope_units(obligation, inventory))
    for claim in claims:
        if claim.claim_type != ClaimType.WITNESS:
            continue
        if claim.unit_type != obligation.spec.unit_type:
            continue
        if claim.predicate is None:
            continue
        if not pr.predicate_entails(claim.predicate, obligation.spec.predicate):
            continue
        if not _scope_compatible(claim.scope, obligation.spec.scope):
            continue
        # The witness must point at a unit inside the obligation's
        # domain. A claim whose scope is a strict superset of the
        # obligation's scope can have positive_units outside the
        # obligation's universe; those don't witness anything for
        # *this* obligation.
        in_domain = [u for u in claim.positive_units if u in domain]
        if not in_domain:
            continue
        return ClosureResult(success=True, value=in_domain[0], used_claim_ids=[claim.id])
    return ClosureResult(success=False, diagnostic="missing_witness")
