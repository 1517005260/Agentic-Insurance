"""Per-handler validation + execute helpers.

Each top-level ``handle_*`` on Plant follows the same shape:

1. ``validate_<op>_path(plant, ...)`` — translate args + check
   preconditions, return either an error or the parsed objects the
   executor needs.
2. (For decompose) ``execute_decompose(plant, ...)`` — run the
   validated state mutation and assemble the ``PlantResult``.
3. ``validate_replacement`` — locate a pending challenge, confirm
   it's resolvable, and dispatch to the per-repair-kind contract.

Plant.handle_* keeps the public flow but calls these helpers for
the heavy lifting. The atomic snapshot/restore frame and the
challenge-replacement discharge stay on Plant — those touch state
across all three stores and the test patches plant._discharge_*
directly.
"""

from typing import Any, Dict, List, Optional, Tuple

from agentic.proof import predicate as pr
from agentic.proof.rules import decomposition as decomposition_rules
from agentic.proof.rules import repair as repair_contracts
from agentic.proof.score import registry as sr
from agentic.proof.types import (
    ObligationKind,
    ObligationSpec,
    ObligationStatus,
)


from agentic.proof.errors import make_envelope as _err  # canonical envelope builder


def _available_obligation_ids(plant: Any) -> Dict[str, List[str]]:
    """Echo the LLM's pickable obligation ids by status so an
    ``unknown_parent`` / ``unknown_obligation`` envelope doesn't make
    the LLM cross-reference ``gate.open_obligations`` to find a valid
    id. Splitting by status hints which path each id supports
    (decompose needs DECOMPOSED parent; challenge needs OPEN target)."""
    by_status: Dict[str, List[str]] = {"OPEN": [], "DECOMPOSED": [], "CHALLENGED": [], "CLOSED": []}
    for o in plant.obligations.all_obligations():
        bucket = by_status.get(o.status.value)
        if bucket is not None:
            bucket.append(o.id)
    return {k: v for k, v in by_status.items() if v}


def validate_create_path(
    plant: Any,
    spec: ObligationSpec,
    *,
    discharges_challenge: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], bool, Optional[Dict[str, Any]]]:
    """Run every pre-materialisation guard for obligation_create.

    Returns ``(replacement_meta, is_root, error)``. ``replacement_meta``
    is set iff this create is a challenge-replacement;
    ``handle_obligation_create`` consumes it to drive the atomic
    snapshot/restore frame and the eventual discharge call.

    Encapsulates predicate applicability, argmax score validation,
    parent-status checks, root-slot guard, DomainMap-child checks, and
    predicate_mismatch-on-root guard. The atomic frame, store inserts,
    and discharge handler stay in plant.py — only the branchy
    validation that doesn't touch state moves here.
    """
    if not pr.applicable_to_obligation_unit_type(spec.predicate, spec.unit_type):
        return None, False, _err(
            "universal_predicate", "predicate is universal on the obligation's unit_type",
            remediation="Replace the universal pattern (e.g. '.*' / '.+' / empty regex) with a literal-anchored predicate that actually constrains the unit; or use a different primitive like contains_string.",
            valid_example={"name": "contains_string", "args": {"pattern": "Premium", "case_sensitive": False}},
        )
    if not pr.predicate_applicable(spec.predicate, spec.unit_type):
        return None, False, _err(
            "predicate_not_applicable",
            f"predicate not applicable to unit_type={spec.unit_type}",
            remediation=f"Either change `unit_type` to one supported by this predicate, or pick a predicate compatible with unit_type={spec.unit_type!r} (e.g. section_title_contains for sections, contains_string works on both).",
        )
    # argmax requires score AND that score must be orderable.
    # text_field produces strings, which Γ_argmax cannot compare;
    # rejecting at create gives a precise remediation instead of a
    # silent missing_comparison at close time.
    if spec.kind == ObligationKind.ARGMAX:
        if spec.score is None:
            return None, False, _err(
                "missing_score_ref", "argmax obligation requires a score spec",
                remediation="Add a `score` field to the spec (e.g. {'name':'numeric_amount','args':{}}) — argmax cannot rank units without an extractor.",
                valid_example={"name": "numeric_amount", "args": {}},
            )
        if not sr.is_orderable(spec.score):
            return None, False, _err(
                "unsupported_score_for_argmax",
                f"score {spec.score.name!r} is not orderable; argmax requires a numeric or date score",
                remediation="Replace `score.name` with one of numeric_amount / percentage / integer_count / date_iso (text_field is not orderable).",
                valid_example={"name": "numeric_amount", "args": {}},
            )

    is_root = (
        plant.obligations.root() is None
        and spec.parent_id is None
        and discharges_challenge is None
    )
    replacement_meta: Optional[Dict[str, Any]] = None

    if discharges_challenge is not None:
        replacement_meta, err = plant._validate_replacement(spec, discharges_challenge)
        if err is not None:
            return None, False, err
        # A repair-replacement that retires the current root must
        # transfer root identity to the new obligation; ΓNegation
        # tightening, has_active_required and the pre-proof window are
        # all anchored on is_root.
        target_id = replacement_meta.get("target_obligation_id") if replacement_meta else None
        target = plant.obligations.get(target_id) if target_id else None
        if target is not None and target.is_root:
            is_root = True
        # Same-slot DomainMap replacement still has to satisfy the
        # singleton + canonical-key invariants. ``replacing_obligation_id``
        # signals it so the duplicate / predicate-match guards apply
        # the relaxed semantics.
        if (
            target is not None
            and target.spec.parent_id is not None
            and target.spec.parent_id in plant.domain_maps
        ):
            parent = plant.obligations.get(target.spec.parent_id)
            if parent is not None:
                err = plant._validate_domain_map_child(
                    parent, spec, replacing_obligation_id=target.id,
                )
                if err is not None:
                    return None, False, err
    elif spec.parent_id is not None:
        parent = plant.obligations.get(spec.parent_id)
        if parent is None:
            return None, False, _err(
                "unknown_parent",
                f"parent_id={spec.parent_id!r} does not exist",
                remediation="Pick an obligation_id from `available_obligation_ids` (a DECOMPOSED parent for a sub-obligation, or omit parent_id for a root create).",
                requested_parent_id=spec.parent_id,
                available_obligation_ids=_available_obligation_ids(plant),
            )
        if parent.status != ObligationStatus.DECOMPOSED:
            return None, False, _err(
                "parent_not_decomposed",
                "parent must be DECOMPOSED before child create",
                remediation="Call obligation_decompose on the parent first (with a registered rule_id) so it transitions to DECOMPOSED, then re-issue the child create.",
            )
        # Children of a DomainMap parent must be singleton-scoped
        # canonical-key children; otherwise a broad child's closure
        # would propagate-close every unit in the domain map.
        if spec.parent_id in plant.domain_maps:
            err = plant._validate_domain_map_child(parent, spec)
            if err is not None:
                return None, False, err
            # The schema doesn't surface ``derived_by``, so a singleton
            # child of a DomainMap parent must be tagged here so
            # _register_with_domain_map_if_applicable doesn't skip it.
            spec.derived_by = "map_over_domain"
    else:
        if not is_root:
            return None, False, _err(
                "root_already_exists",
                "a root obligation already exists; pass parent_id or discharges_challenge",
                remediation="Either set `parent_id` to attach this obligation under an existing DECOMPOSED parent, or set `discharges_challenge` to replace a CHALLENGED obligation; the root slot is already filled.",
            )
        if spec.sealed_scope_override:
            return None, False, _err(
                "override_on_root_forbidden",
                "sealed_scope_override is not allowed on root",
                remediation="Drop `sealed_scope_override` from the root spec (set false or omit it); use scope.sealed=true on the scope itself if you need a sealed root.",
            )

    if (
        discharges_challenge is not None
        and replacement_meta is not None
        and replacement_meta.get("forbids_root_predicate_mismatch")
    ):
        return None, False, _err(
            "predicate_mismatch_on_root_forbidden",
            "root cannot be replaced via predicate_mismatch",
            remediation="To change the root's predicate use the `wrong_question_kind` repair (only in pre-proof window, capped once per session). predicate_mismatch is for non-root obligations only.",
        )

    return replacement_meta, is_root, None


def validate_decompose_path(
    plant: Any,
    *,
    parent_id: str,
    rule_id: str,
    discharges_challenge: Optional[str],
) -> Tuple[Optional[Any], Optional[str], Optional[Dict[str, Any]]]:
    """Validate the parent / rule / challenge-discharge preconditions
    for ``handle_obligation_decompose``.

    Returns ``(parent, prepay_discharge, error)``. ``prepay_discharge``
    is the challenge_id to discharge LATER, after child validation
    passes — discharging earlier would leave a CHALLENGED parent
    half-discharged on a malformed retry.
    """
    parent = plant.obligations.get(parent_id)
    if parent is None:
        return None, None, _err(
            "unknown_parent", f"parent_id={parent_id!r} does not exist",
            remediation="Pick a DECOMPOSED parent's obligation_id from `available_obligation_ids` and re-issue.",
            requested_parent_id=parent_id,
            available_obligation_ids=_available_obligation_ids(plant),
        )
    prepay_discharge: Optional[str] = None
    if parent.status == ObligationStatus.CHALLENGED:
        if discharges_challenge is None:
            return None, None, _err(
                "parent_challenged",
                "CHALLENGED parent can only be decomposed when discharges_challenge is set",
                remediation="Find the pending missing_subobligation challenge_id from gate.challenged_obligations for this parent and pass it as `discharges_challenge`.",
            )
        ch = plant.challenges.get(discharges_challenge)
        if ch is None or ch.obligation_id != parent_id or ch.status != "pending":
            return None, None, _err(
                "invalid_discharge",
                "discharges_challenge does not match a pending challenge on this parent",
                remediation="Pass a `discharges_challenge` whose challenge has obligation_id == this parent_id AND status=='pending'; cross-check gate.challenged_obligations.",
            )
        if ch.repair_kind != "missing_subobligation":
            return None, None, _err(
                "wrong_repair_kind_for_decompose",
                "only missing_subobligation challenges can be discharged via decompose",
                remediation="If the pending challenge is scope_too_narrow/broad/predicate_mismatch/wrong_question_kind, discharge it via obligation_create(discharges_challenge=...) instead.",
            )
        prepay_discharge = ch.id
    elif parent.status != ObligationStatus.OPEN:
        return None, None, _err(
            "parent_not_open",
            f"parent status={parent.status.value}, expected OPEN",
            remediation="Pick a parent_id whose status is OPEN (CLOSED/DECOMPOSED/CHALLENGED parents cannot be re-decomposed); inspect gate.open_obligations.",
        )
    if rule_id not in {"and_split", "scope_partition", "case_split", "map_over_domain"}:
        return None, None, _err(
            "unknown_rule", f"rule_id={rule_id!r} not registered",
            remediation="Set rule_id to one of and_split / scope_partition / case_split / map_over_domain.",
        )
    return parent, prepay_discharge, None


def validate_challenge_path(
    plant: Any,
    *,
    obligation_id: str,
    repair_kind: str,
    evidence_ids: List[str],
) -> Tuple[Optional[Any], bool, Optional[Dict[str, Any]]]:
    """Validate the obligation / repair_kind / evidence preconditions
    for ``handle_obligation_challenge``.

    Returns ``(obligation, wrong_kind_eligible, error)``.
    ``wrong_kind_eligible`` signals the caller should consume the
    once-per-session wrong_question_kind cap LAST (after every other
    validation has passed) so a malformed retry doesn't burn the cap.
    """
    obligation = plant.obligations.get(obligation_id)
    if obligation is None:
        return None, False, _err(
            "unknown_obligation", f"obligation_id={obligation_id!r} does not exist",
            remediation="Pick an OPEN obligation_id from `available_obligation_ids` (only OPEN obligations are challengeable).",
            requested_obligation_id=obligation_id,
            available_obligation_ids=_available_obligation_ids(plant),
        )
    # Only OPEN obligations are directly challengeable. A DECOMPOSED
    # parent has no valid OPEN→CHALLENGED transition and would leave a
    # stranded pending challenge if accepted.
    if obligation.status != ObligationStatus.OPEN:
        return None, False, _err(
            "obligation_not_challengeable",
            f"status={obligation.status.value}; cannot challenge",
            remediation="Only OPEN obligations are challengeable. For a DECOMPOSED parent, challenge one of its children instead; for CLOSED/RETIRED, no challenge is needed.",
        )
    # ``unsealed_scope`` is omitted from v1 — its postcondition would
    # need to flip ScopeRef.sealed, but that field is frozen at create.
    if repair_kind not in {
        "scope_too_narrow", "scope_too_broad", "predicate_mismatch",
        "missing_subobligation", "wrong_question_kind",
    }:
        return None, False, _err(
            "unknown_repair_kind",
            f"repair_kind={repair_kind!r} not registered",
            remediation="Set repair_kind to one of scope_too_narrow / scope_too_broad / predicate_mismatch / missing_subobligation / wrong_question_kind.",
        )
    if obligation.is_root and repair_kind == "predicate_mismatch":
        return None, False, _err(
            "predicate_mismatch_on_root_forbidden",
            "root predicate is frozen; use wrong_question_kind in pre-proof window",
            remediation="To change the ROOT obligation's predicate, switch repair_kind to 'wrong_question_kind' (only valid in pre-proof window, capped once per session).",
        )
    wrong_kind_eligible = False
    if repair_kind == "wrong_question_kind":
        if not obligation.is_root:
            return None, False, _err(
                "wrong_kind_only_on_root",
                "wrong_question_kind only applies to root",
                remediation="Pick a different repair_kind for non-root obligations (predicate_mismatch / scope_too_narrow / scope_too_broad / missing_subobligation), or target the root obligation_id.",
            )
        if not plant._is_pre_proof_window():
            return None, False, _err(
                "not_pre_proof_window",
                "wrong_question_kind only allowed before any required obligation has left OPEN",
                remediation="The pre-proof window has closed (some required obligation is already CLOSED/CHALLENGED/DECOMPOSED). Either accept the existing root kind, or finalize/abstain with the current proof tree.",
            )
        wrong_kind_eligible = True
    for eid in evidence_ids:
        if plant.evidence.get_observation(eid) is None:
            return None, False, _err(
                "unknown_observation", f"evidence id {eid!r} unknown",
                remediation="Use observation_ids returned by recent acquisition tool calls (e.g. read_page, pattern_search). The observation must exist before it can be cited as evidence.",
                evidence_id=eid,
            )
    return obligation, wrong_kind_eligible, None


def validate_replacement(
    plant: Any,
    spec: ObligationSpec,
    challenge_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Replacement-validation envelope: locate the challenge, confirm
    it's pending and points to a real target, then dispatch to the
    per-repair-kind contract registry. Each repair_kind allows EXACTLY
    one field to change; everything else must match the target."""
    challenge = plant.challenges.get(challenge_id)
    if challenge is None:
        return None, _err(
            "unknown_challenge", f"challenge_id={challenge_id!r} unknown",
            remediation="Pass a valid challenge_id from gate.challenged_obligations (returned by obligation_challenge or visible in gate snapshots).",
            challenge_id=challenge_id,
        )
    if challenge.status != "pending":
        return None, _err(
            "challenge_already_resolved",
            f"challenge_id={challenge_id!r} status={challenge.status}",
            remediation="This challenge is already discharged; do not pass discharges_challenge for an already-resolved challenge. Inspect gate.challenged_obligations for currently pending challenges.",
            challenge_id=challenge_id,
            challenge_status=challenge.status,
        )
    target = plant.obligations.get(challenge.obligation_id)
    if target is None:
        return None, _err(
            "challenge_target_missing", "challenge target obligation not found",
            remediation="The obligation this challenge targets has been removed; this is a plant invariant violation — re-issue obligation_challenge against a current OPEN obligation.",
        )
    return repair_contracts.validate_replacement(
        spec, target, challenge.repair_kind, plant.inventory, _err,
    )


def execute_decompose(
    plant: Any,
    *,
    parent_id: str,
    rule_id: str,
    parent: Any,
    built_children: List[ObligationSpec],
    prepay_discharge: Optional[str],
    discharges_challenge: Optional[str],
) -> Any:
    """Execute the validated eager-rule decompose: discharge any
    pending challenge, mark parent DECOMPOSED, insert children, and
    return the assembled ``PlantResult``.

    ``map_over_domain`` has its own dedicated path
    (``state.domain_map_handler.handle_map_over_domain``) and never
    reaches this function.
    """
    from agentic.proof.plant import PlantResult

    coverage_err = decomposition_rules.validate_rule(
        rule_id, parent, built_children, plant.inventory, _err,
    )
    if coverage_err is not None:
        return PlantResult(ok=False, error=coverage_err, gate=plant.gate_view())

    for spec in built_children:
        spec.derived_by = rule_id  # type: ignore[assignment]

    # Validation passed. Discharge a pending missing_subobligation
    # challenge first so the OPEN→DECOMPOSED transition is legal.
    # Discharging exactly once and only at this point keeps the path
    # atomic — earlier failures leave state untouched.
    if prepay_discharge is not None:
        plant.challenges.discharge(prepay_discharge, meta={"resolved_via": "decomposition"})
        plant.obligations.record_challenge_discharged(parent_id, prepay_discharge)
    # Even if the LLM passed discharges_challenge for an OPEN parent
    # (pre-OPEN repair flow), close it the same way.
    elif discharges_challenge is not None:
        ch = plant.challenges.get(discharges_challenge)
        if ch is not None and ch.obligation_id == parent_id and ch.status == "pending":
            plant.challenges.discharge(ch.id, meta={"resolved_via": "decomposition"})
            # parent was OPEN; record_challenge_discharged tolerates
            # OPEN obligations — it just removes the challenge from
            # the open list.
            plant.obligations.record_challenge_discharged(parent_id, ch.id)

    plant.obligations.record_decompose(parent_id, child_ids=[], rule_id=rule_id)
    child_ids: List[str] = []
    for spec in built_children:
        child = plant.obligations.insert(spec, is_root=False)
        child_ids.append(child.id)
    parent_after = plant.obligations.get(parent_id)
    if parent_after is not None:
        for cid in child_ids:
            if cid not in parent_after.children_ids:
                parent_after.children_ids.append(cid)

    plant.reconcile()
    return PlantResult(
        ok=True,
        payload={"parent_id": parent_id, "child_ids": child_ids, "rule_id": rule_id},
        gate=plant.gate_view(),
    )


def validate_decomposition_rule(
    plant: Any,
    rule_id: str,
    parent: Any,
    children: List[ObligationSpec],
) -> Optional[Dict[str, Any]]:
    """Forward to the declarative decomposition_rules registry."""
    return decomposition_rules.validate_rule(
        rule_id, parent, children, plant.inventory, _err,
    )
