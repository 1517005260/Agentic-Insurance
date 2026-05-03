"""Domain-map handlers — plant-side glue around DomainMapStore.

Plant entry points for ``map_over_domain`` decomposition,
canonical-key registration of children, propagating closure events
back into the domain map, and aggregating closed children into a
virtual parent claim. Each function takes the plant as a parameter so
testing stays plant-mock-friendly (no module-level state).

The store itself (``DomainMapStore``) holds only the data model;
business rules that touch obligations, claims, and predicates live
here.
"""

from typing import Any, Dict, List, Optional

from agentic.proof import predicate as pr
from agentic.proof.state.domain_map import canonical_key
from agentic.proof.types import (
    Citation,
    Claim,
    ClaimType,
    ClosureResult,
    Obligation,
    ObligationKind,
    ObligationSpec,
    ObligationStatus,
)


from agentic.proof.errors import make_envelope as _err  # canonical envelope builder


def validate_domain_map_child(
    plant: Any,
    parent: Obligation,
    spec: ObligationSpec,
    *,
    replacing_obligation_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """A child of a DomainMap parent must:

    1. Cover exactly one unit in the parent's domain (singleton scope).
    2. Match the canonical map_over_domain template: same kind,
       same predicate, same polarity, and (for argmax) same score.
    3. Have a canonical key not already materialised — duplicate
       registration is rejected outright.

    ``replacing_obligation_id`` signals same-slot challenge_replacement;
    the slot is expected to currently hold that id, and the predicate-
    match constraint is relaxed (predicate_mismatch repair *changes*
    the predicate; the parent's Γ catches any unsound divergence).
    """
    units = plant.inventory.units(
        spec.unit_type,
        file_ids=list(spec.scope.file_ids),
        section_ids=list(spec.scope.section_ids) if spec.scope.section_ids else None,
    )
    if len(units) != 1:
        return _err(
            "domain_map_non_singleton",
            "DomainMap children must scope to exactly one inventory unit",
            remediation="Narrow the child's scope so it resolves to exactly ONE inventory unit (one file_id for unit_type='file', or one section_id for 'section'); DomainMap children are singleton by construction.",
        )
    unit_id = units[0]
    domain_map = plant.domain_maps.get(parent.id)
    if domain_map is None or unit_id not in domain_map.domain_units:
        return _err(
            "domain_map_unit_outside_domain",
            f"unit {unit_id!r} is not part of parent's DomainMap",
            remediation="Pick a unit_id from the parent's domain (the parent's scope at the time of map_over_domain decomposition); inspect the parent obligation's scope or the prior decompose result for the domain.",
            unit_id=unit_id,
        )
    # Same-slot replacement is allowed; the calling transaction holds
    # the snapshot frame so canonical_keys + materialised_children swap
    # atomically inside replace_materialised.
    existing_holder = domain_map.materialised_children.get(unit_id)
    if existing_holder is not None and existing_holder != replacing_obligation_id:
        return _err(
            "domain_map_unit_already_materialised",
            f"unit {unit_id!r} already has obligation {existing_holder!r}",
            remediation=f"Use the existing obligation_id {existing_holder!r} instead of creating a new child for the same unit; only one obligation may represent each domain unit.",
            unit_id=unit_id,
            existing_obligation_id=existing_holder,
        )
    # Predicate-match constraint relaxed for challenge_replacement —
    # predicate_mismatch repair, by definition, diverges from the
    # parent's predicate. The parent's Γ_kind refuses to close on
    # divergent child claims if the divergence is unsound.
    if (
        replacing_obligation_id is None
        and pr.serialize_spec(spec.predicate) != pr.serialize_spec(parent.spec.predicate)
    ):
        return _err(
            "domain_map_predicate_mismatch",
            "DomainMap child predicate must equal parent predicate",
            remediation="Copy the parent's predicate verbatim into the child spec — DomainMap children inherit the parent predicate per-unit.",
        )
    if spec.kind != ObligationKind.EXISTS and spec.kind != parent.spec.kind:
        # Per-unit children are usually existence-style; we accept the
        # parent's completeness kind as the alternative.
        return _err(
            "domain_map_kind_mismatch",
            "DomainMap child kind must be 'exists' or match parent kind",
            remediation=f"Set the child's `kind` to 'exists' (canonical per-unit lookup) or to {parent.spec.kind.value!r} (matching the parent).",
        )
    if spec.polarity != parent.spec.polarity:
        return _err(
            "domain_map_polarity_mismatch", "child polarity must match parent",
            remediation=f"Set the child's `polarity` to {parent.spec.polarity!r} (matching the parent).",
        )
    if parent.spec.kind == ObligationKind.ARGMAX:
        if spec.score is None or parent.spec.score is None:
            return _err(
                "domain_map_score_missing", "argmax DomainMap children require score",
                remediation="Add a `score` field to the child spec matching the argmax parent's score (e.g. {'name':'numeric_amount','args':{}}).",
                valid_example={"name": "numeric_amount", "args": {}},
            )
        if spec.score.name != parent.spec.score.name or spec.score.args != parent.spec.score.args:
            return _err(
                "domain_map_score_mismatch", "child score must match parent score",
                remediation=f"Copy the parent's score verbatim: name={parent.spec.score.name!r}, args={parent.spec.score.args!r}.",
            )
    key = canonical_key(
        parent_id=parent.id,
        unit_id=unit_id,
        kind=spec.kind,
        predicate=spec.predicate,
        score=spec.score,
        polarity=spec.polarity,
    )
    if plant.domain_maps.lookup_by_key(parent.id, key) is not None:
        return _err(
            "domain_map_duplicate_child",
            f"unit {unit_id!r} already has a materialised child obligation",
            remediation="Reuse the existing child obligation for this unit instead of creating another with the same canonical (kind, predicate, score, polarity) signature.",
            unit_id=unit_id,
        )
    return None


def register_with_domain_map_if_applicable(
    plant: Any,
    obligation: Obligation,
    *,
    replacing_id: Optional[str] = None,
) -> None:
    """Register a freshly-created obligation in its parent's
    DomainMap. Without ``replacing_id``: standard initial
    ``materialise``. With ``replacing_id`` (same-slot swap):
    ``replace_materialised`` — slot must currently hold ``replacing_id``,
    and canonical_keys move atomically along with the obligation
    pointer."""
    parent_id = obligation.spec.parent_id
    if parent_id is None or parent_id not in plant.domain_maps:
        return
    units = plant.inventory.units(
        obligation.spec.unit_type,
        file_ids=list(obligation.spec.scope.file_ids),
        section_ids=list(obligation.spec.scope.section_ids) if obligation.spec.scope.section_ids else None,
    )
    if len(units) != 1:
        return
    unit_id = units[0]
    key = canonical_key(
        parent_id=parent_id,
        unit_id=unit_id,
        kind=obligation.spec.kind,
        predicate=obligation.spec.predicate,
        score=obligation.spec.score,
        polarity=obligation.spec.polarity,
    )
    if replacing_id is not None:
        plant.domain_maps.replace_materialised(
            parent_id=parent_id, unit_id=unit_id,
            old_id=replacing_id, new_id=obligation.id,
            new_canonical_key=key,
        )
    else:
        plant.domain_maps.materialise(parent_id, unit_id, obligation.id, key)


def propagate_close_to_domain_map(
    plant: Any, obligation: Obligation, result: ClosureResult,
) -> None:
    """Mark the obligation's slot in its parent's DomainMap closed.

    Marks against the unit_id this obligation was REGISTERED for, not
    its current scope — a replacement that drifts past the singleton
    check would otherwise mark every unit closed and fake-complete the
    parent's domain. Predicate-entailment guard: if the child's
    predicate doesn't entail the parent's (predicate_mismatch repair
    permits divergence), DO NOT mark the slot closed; the parent's
    Γ_kind would otherwise see "k/N closed" for a weaker property.
    """
    parent_id = obligation.spec.parent_id
    if parent_id is None:
        return
    if parent_id not in plant.domain_maps:
        return
    parent = plant.obligations.get(parent_id)
    if parent is not None:
        if not pr.predicate_entails(obligation.spec.predicate, parent.spec.predicate):
            return
    dm = plant.domain_maps.get(parent_id)
    if dm is None:
        return
    for unit_id, obl_id in dm.materialised_children.items():
        if obl_id == obligation.id:
            plant.domain_maps.mark_closed(parent_id, unit_id)
            return


def synthesise_scope_partition_claim(plant: Any, parent: Obligation) -> Optional[Claim]:
    """Aggregate decomposed children's claims into one virtual parent-
    scoped claim so the parent's Γ_kind can re-run on the combined
    evidence. Used by both scope_partition (children with disjoint
    sub-scopes) and map_over_domain (children with singleton scopes);
    both rules guarantee children's scopes union to the parent's
    domain.

    ScanClaim children synthesise into a virtual ScanClaim
    (positive/negative unit unions). WitnessClaim children synthesise
    into a virtual WitnessClaim (positive_units union, value_map merged
    with conflict rejection)."""
    history = parent.history or []
    rule = next(
        (h.get("rule_id") for h in history if h.get("event") == "decompose"),
        None,
    )
    if rule not in ("scope_partition", "map_over_domain"):
        return None
    scan_pos: List[str] = []
    scan_neg: List[str] = []
    wit_pos: List[str] = []
    wit_value_map: Dict[str, Any] = {}
    wit_citations: List[Citation] = []
    seen_scan_pos: set = set()
    seen_scan_neg: set = set()
    seen_wit_pos: set = set()
    scan_ids: List[str] = []
    wit_ids: List[str] = []
    for cid in parent.children_ids:
        child = plant.obligations.get(cid)
        if child is None or child.status != ObligationStatus.CLOSED:
            return None
        # Predicate-entailment guard: a child whose predicate does NOT
        # entail the parent's must not contribute to the synthesised
        # parent claim. Without this guard a predicate_mismatch
        # replacement on a DomainMap child could propagate weaker
        # proven content into the parent's Γ.
        if not pr.predicate_entails(child.spec.predicate, parent.spec.predicate):
            return None
        # Aggregate only the claims that actually closed the child —
        # using every binding would conflate ancillary claims whose
        # partition could disagree with the closing one.
        for claim_id in child.closed_by:
            c = plant.evidence.get_claim(claim_id)
            if c is None:
                continue
            if c.claim_type == ClaimType.SCAN:
                if c.id not in scan_ids:
                    scan_ids.append(c.id)
                for u in c.positive_units:
                    if u not in seen_scan_pos:
                        scan_pos.append(u)
                        seen_scan_pos.add(u)
                for u in c.negative_units:
                    if u not in seen_scan_neg:
                        scan_neg.append(u)
                        seen_scan_neg.add(u)
            elif c.claim_type == ClaimType.WITNESS:
                if c.id not in wit_ids:
                    wit_ids.append(c.id)
                for u in c.positive_units:
                    if u not in seen_wit_pos:
                        wit_pos.append(u)
                        seen_wit_pos.add(u)
                for unit_id, val in (c.value_map or {}).items():
                    if unit_id in wit_value_map and wit_value_map[unit_id] != val:
                        # Inconsistent witnesses for the same unit
                        # across siblings — abort aggregation; parent
                        # stays DECOMPOSED so the LLM sees the
                        # contradiction via diagnose.
                        return None
                    wit_value_map[unit_id] = val
                for ct in c.citations:
                    wit_citations.append(ct)
    if scan_ids:
        return Claim(
            id=f"virt_{parent.id}",
            observation_id=None,
            claim_type=ClaimType.SCAN,
            scope=parent.spec.scope,
            unit_type=parent.spec.unit_type,
            predicate=parent.spec.predicate,
            score=parent.spec.score,
            positive_units=scan_pos,
            negative_units=scan_neg,
            value_map={},
            citations=[],
            derivation="plant_aggregated",
        )
    if wit_ids:
        return Claim(
            id=f"virt_{parent.id}",
            observation_id=None,
            claim_type=ClaimType.WITNESS,
            scope=parent.spec.scope,
            unit_type=parent.spec.unit_type,
            predicate=parent.spec.predicate,
            score=parent.spec.score,
            positive_units=wit_pos,
            negative_units=[],
            value_map=wit_value_map,
            citations=wit_citations,
            derivation="plant_aggregated",
        )
    return None


def handle_map_over_domain(
    plant: Any,
    *,
    parent: Obligation,
    discharges_challenge: Optional[str],
    prepay_discharge: Optional[str] = None,
) -> Any:
    """Materialise a ``map_over_domain`` decomposition.

    Returns a ``PlantResult`` (imported lazily so this module stays
    free of plant.py imports at module load).
    """
    from agentic.proof.plant import PlantResult

    domain = plant.inventory.units(
        parent.spec.unit_type,
        file_ids=list(parent.spec.scope.file_ids),
        section_ids=list(parent.spec.scope.section_ids) if parent.spec.scope.section_ids else None,
    )
    if not domain:
        return PlantResult(
            ok=False,
            error=_err(
                "empty_domain", "map_over_domain on empty inventory",
                remediation="Widen the parent's scope.file_ids (or scope.section_ids) so the domain has at least one inventory unit; or pick a different decomposition rule (and_split / scope_partition / case_split).",
            ),
            gate=plant.gate_view(),
        )
    # Discharge first if parent was CHALLENGED, then decompose.
    if prepay_discharge is not None:
        plant.challenges.discharge(prepay_discharge, meta={"resolved_via": "map_over_domain"})
        plant.obligations.record_challenge_discharged(parent.id, prepay_discharge)
    elif discharges_challenge is not None:
        ch = plant.challenges.get(discharges_challenge)
        if ch is not None and ch.obligation_id == parent.id and ch.status == "pending":
            plant.challenges.discharge(ch.id, meta={"resolved_via": "map_over_domain"})
            plant.obligations.record_challenge_discharged(parent.id, ch.id)
    plant.obligations.record_decompose(parent.id, child_ids=[], rule_id="map_over_domain")
    plant.domain_maps.install(parent.id, domain)
    plant.reconcile()
    return PlantResult(
        ok=True,
        payload={"parent_id": parent.id, "rule_id": "map_over_domain", "domain_size": len(domain)},
        gate=plant.gate_view(),
    )
