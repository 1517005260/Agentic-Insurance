"""Fixed-point reconcile loop and the auto-bind / auto-close helpers
it composes. Public entry point: :func:`reconcile`.

The loop runs four steps until no state changes (or a hard cap fires):

1. Auto-bind newly-OPEN obligations against every existing claim.
2. Discharge challenges whose mechanical postcondition is met.
3. Auto-close OPEN obligations whose Γ_kind passes on bound claims.
4. Auto-close DECOMPOSED parents whose children are CLOSED, by
   re-running the parent's Γ_kind on the union of bound + child-bound
   claims (and a virtual partition claim for scope_partition /
   map_over_domain decompositions).

Helpers (``has_bindings``, ``auto_bind``, ``auto_bind_obligation``,
``claim_matches_obligation``, ``all_children_closed``,
``postcondition_met``) are public to plant.py — it forwards them as
private methods so external test stubs continue to resolve.
"""

import logging
from typing import Any, Dict, List

from agentic.proof.closure import rules as closure_rules
from agentic.proof import predicate as pr
from agentic.proof.types import (
    Binding,
    Challenge,
    Claim,
    Obligation,
    ObligationStatus,
)


logger = logging.getLogger(__name__)

RECONCILE_MAX_LOOPS = 256


def has_bindings(plant: Any, obligation_id: str) -> bool:
    return bool(plant.evidence.bindings_for_obligation(obligation_id))


def claim_matches_obligation(claim: Claim, obligation: Obligation) -> bool:
    if claim.unit_type != obligation.spec.unit_type:
        return False
    if not closure_rules._scope_compatible(claim.scope, obligation.spec.scope):
        return False
    if claim.predicate is None:
        return False
    if not pr.predicate_entails(claim.predicate, obligation.spec.predicate):
        return False
    # v1 only emits positive WitnessClaim/ScanClaim. Negative-polarity
    # obligations have no certifying claim shape and go through
    # ObligationKind.NEGATION instead.
    if obligation.spec.polarity != "positive":
        return False
    return True


def auto_bind(plant: Any, claim: Claim) -> List[Binding]:
    bindings: List[Binding] = []
    for o in plant.obligations.by_status(ObligationStatus.OPEN):
        if o.status != ObligationStatus.OPEN:
            continue
        if not claim_matches_obligation(claim, o):
            continue
        binding = plant.evidence.add_binding(Binding(
            obligation_id=o.id, claim_id=claim.id, auto=True,
        ))
        bindings.append(binding)
    return bindings


def auto_bind_obligation(plant: Any, obligation: Obligation) -> bool:
    """Bind every existing claim that matches the freshly-OPEN
    obligation. Returns True iff any new binding was added."""
    existing = {b.claim_id for b in plant.evidence.bindings_for_obligation(obligation.id)}
    added = False
    for claim in plant.evidence.claims():
        if claim.id in existing:
            continue
        if not claim_matches_obligation(claim, obligation):
            continue
        plant.evidence.add_binding(Binding(
            obligation_id=obligation.id, claim_id=claim.id, auto=True,
        ))
        added = True
    return added


def all_children_closed(plant: Any, parent: Obligation) -> bool:
    if parent.id in plant.domain_maps:
        return plant.domain_maps.all_closed(parent.id)
    if not parent.children_ids:
        return False
    for cid in parent.children_ids:
        child = plant.obligations.get(cid)
        if child is None or child.status != ObligationStatus.CLOSED:
            return False
    return True


def postcondition_met(plant: Any, challenge: Challenge) -> bool:
    target = plant.obligations.get(challenge.obligation_id)
    if target is None:
        return False
    if challenge.repair_kind == "missing_subobligation":
        return target.status == ObligationStatus.DECOMPOSED and bool(target.children_ids)
    # scope_too_narrow / scope_too_broad / predicate_mismatch /
    # wrong_question_kind discharge inside _discharge_challenge_with_replacement
    # at create time — no postcondition fires for them here.
    return False


def reconcile(plant: Any) -> List[Dict[str, Any]]:
    """Fixed-point auto_bind + auto_close + challenge_discharge.

    Returns a list of closure events that fired during this call,
    each ``{"obligation_id": ..., "value": ..., "used_claim_ids": [...]}``.
    Hits a hard cap (proportional to the largest plausible
    decomposition chain) and logs a warning so a non-converged
    reconcile can be diagnosed instead of silently leaving the gate
    inconsistent.
    """
    triggered: List[Dict[str, Any]] = []
    changed = True
    loops = 0
    while changed and loops < RECONCILE_MAX_LOOPS:
        loops += 1
        changed = False

        # 1. auto_bind newly-open obligations against all claims.
        #    (auto_bind on ingest already did the per-claim version;
        #    here we re-bind in case a brand-new obligation just
        #    appeared.)
        for o in list(plant.obligations.all_obligations()):
            if o.status != ObligationStatus.OPEN:
                continue
            if not plant._has_bindings(o.id):
                if plant._auto_bind_obligation(o):
                    changed = True

        # 2. discharge challenges whose mechanical postcondition is met.
        for ch in plant.challenges.all():
            if ch.status != "pending":
                continue
            if plant._postcondition_met(ch):
                plant.challenges.discharge(ch.id, meta={"resolved_via": "postcondition"})
                plant.obligations.record_challenge_discharged(ch.obligation_id, ch.id)
                changed = True

        # 3. auto_close OPEN obligations.
        for o in list(plant.obligations.by_status(ObligationStatus.OPEN)):
            if plant._ancestor_challenged(o.id):
                continue
            claim_ids = [b.claim_id for b in plant.evidence.bindings_for_obligation(o.id)]
            bound_claims = [plant.evidence.get_claim(cid) for cid in claim_ids]
            bound_claims = [c for c in bound_claims if c is not None]
            # Defensive: step 1 should have caught any pre-existing
            # claims, but evaluate against the bound set we just
            # collected.
            result = closure_rules.evaluate_closure(o, bound_claims, plant.inventory)
            if result.success:
                plant.obligations.record_close(o.id, result.used_claim_ids, result.value)
                triggered.append({
                    "obligation_id": o.id,
                    "value": result.value,
                    "used_claim_ids": result.used_claim_ids,
                })
                plant._propagate_close_to_domain_map(o, result)
                changed = True

        # 4. auto_close DECOMPOSED parents whose children are CLOSED.
        #    Every parent — DomainMap or otherwise — must re-run its
        #    own Γ_kind against the union of bound + child-bound
        #    claims. Closing on child status alone is unsound because
        #    case_split / scope_partition / and_split cannot guarantee
        #    "all children closed" implies the parent's predicate
        #    holds without re-checking.
        for o in list(plant.obligations.by_status(ObligationStatus.DECOMPOSED)):
            # Defensive guard: never close a DECOMPOSED parent while
            # an ancestor is CHALLENGED. Mirrors the rule in step 3 so
            # the invariant survives any future loosening of the
            # state-machine table.
            if plant._ancestor_challenged(o.id):
                continue
            if not plant._all_children_closed(o):
                continue
            bound_claims = [
                plant.evidence.get_claim(b.claim_id)
                for b in plant.evidence.bindings_for_obligation(o.id)
            ]
            bound_claims = [c for c in bound_claims if c is not None]
            for cid in o.children_ids:
                for b in plant.evidence.bindings_for_obligation(cid):
                    c = plant.evidence.get_claim(b.claim_id)
                    if c is not None and c not in bound_claims:
                        bound_claims.append(c)
            # scope_partition / map_over_domain children prove disjoint
            # sub-scopes; their claims have narrower scope than the
            # parent so Γ_kind's scope_compatible check would otherwise
            # reject them. Synthesise a virtual parent-scoped claim
            # from the children's claim partitions so Γ_kind can close
            # the parent on the combined evidence.
            synthesised = plant._synthesise_scope_partition_claim(o)
            if synthesised is not None:
                bound_claims.append(synthesised)
            parent_result = closure_rules.evaluate_closure(o, bound_claims, plant.inventory)
            if not parent_result.success:
                # Children closed but parent's Γ disagrees — leave
                # parent DECOMPOSED so the LLM can see the failure_kind
                # via gate.diagnose.
                continue
            plant.obligations.record_close(
                o.id,
                parent_result.used_claim_ids,
                parent_result.value,
            )
            triggered.append({
                "obligation_id": o.id,
                "value": parent_result.value,
                "used_claim_ids": parent_result.used_claim_ids,
                "via": "domain_map_gamma" if o.id in plant.domain_maps else "decomposition_gamma",
            })
            changed = True

        # 5. auto_retire originals whose challenges discharged via
        #    replacement is handled inline in
        #    _discharge_challenge_with_replacement; no extra step here.
    if changed:
        logger.warning(
            "reconcile() reached the %d-loop cap before reaching a fixed point; "
            "the gate snapshot may be inconsistent.",
            RECONCILE_MAX_LOOPS,
        )
    return triggered
