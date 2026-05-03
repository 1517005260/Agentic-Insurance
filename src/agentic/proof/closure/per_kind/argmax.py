"""Γ_argmax — verified per-unit values, then argmax under tie policy."""
from typing import Sequence

from agentic.proof.closure.helpers import _scope_compatible, _scope_units
from agentic.proof.score import registry as sr
from agentic.proof.types import Claim, ClaimType, ClosureResult, Obligation
from storage.inventory_store import InventoryStore


def gamma_argmax(
    obligation: Obligation,
    claims: Sequence[Claim],
    inventory: InventoryStore,
) -> ClosureResult:
    """Plant-mechanical argmax over witness ``value_map`` entries.

    The LLM ingests one WitnessClaim per unit in the domain, each with
    ``value_map[unit_id]`` set to a value the plant verified at ingest
    time. ΓArgmax collects the verified values, requires every domain
    unit to be present, and returns ``argmax`` per the obligation's
    tie policy.
    """
    if obligation.spec.score is None:
        return ClosureResult(success=False, diagnostic="missing_score_ref")
    domain = _scope_units(obligation, inventory)
    if not domain:
        return ClosureResult(success=False, diagnostic="empty_domain")

    score_name = obligation.spec.score.name
    if score_name not in sr.schemas():
        return ClosureResult(success=False, diagnostic="unsupported_score_field")

    # Aggregate verified values across all matching witness claims.
    # Two witnesses for the same unit must agree — argmax claims to
    # prove a unit's score, and silently picking the larger value
    # would let an LLM smuggle a higher number in a separate claim.
    aggregated: dict[str, tuple[float, str]] = {}   # unit -> (value, claim_id)
    for claim in claims:
        if claim.claim_type != ClaimType.WITNESS:
            continue
        if claim.unit_type != obligation.spec.unit_type:
            continue
        if claim.score is None or claim.score.name != score_name:
            continue
        if claim.score.args != obligation.spec.score.args:
            continue
        if not _scope_compatible(claim.scope, obligation.spec.scope):
            continue
        for unit, value in (claim.value_map or {}).items():
            try:
                v_num = float(value)
            except (TypeError, ValueError):
                continue
            existing = aggregated.get(unit)
            if existing is None:
                aggregated[unit] = (v_num, claim.id)
            elif not sr.values_match(existing[0], v_num):
                # Use values_match (epsilon-aware) so two ingest-side
                # epsilon-equal witnesses don't trigger spurious
                # contradictions; only genuinely different values block.
                return ClosureResult(success=False, diagnostic="argmax_contradiction")

    missing = [u for u in domain if u not in aggregated]
    if missing:
        return ClosureResult(
            success=False,
            diagnostic="missing_comparison",
        )

    # Identify argmax under tie_policy.
    items = [(unit, aggregated[unit][0], aggregated[unit][1]) for unit in domain]
    max_value = max(v for _, v, _ in items)
    winners = [u for u, v, _ in items if v == max_value]
    used_ids = sorted({cid for _, _, cid in items})
    if obligation.spec.tie_policy == "first":
        return ClosureResult(success=True, value=winners[0], used_claim_ids=used_ids)
    if obligation.spec.tie_policy == "all":
        return ClosureResult(success=True, value=sorted(winners), used_claim_ids=used_ids)
    if obligation.spec.tie_policy == "error":
        if len(winners) > 1:
            return ClosureResult(success=False, diagnostic="argmax_tie_unresolved")
        return ClosureResult(success=True, value=winners[0], used_claim_ids=used_ids)
    return ClosureResult(success=False, diagnostic="invalid_tie_policy")
