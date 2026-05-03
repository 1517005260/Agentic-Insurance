"""Spec resolvers — translate raw LLM payloads into typed dataclasses.

Every plant entry point that accepts user-supplied JSON funnels its
shape validation through the helpers here. Each resolver returns a
``(spec | None, error | None)`` pair so callers can fold structured
errors back into a ``PlantResult`` without raising.

The module has no Plant dependency; it composes registries
(``predicate``, ``score``) and the inventory facade. Plant exposes
each function via a thin private-method forwarder so existing
attribute-based call paths in tests and helpers continue to resolve.
"""

from typing import Any, Dict, List, Optional, Tuple

from agentic.proof import predicate as pr
from agentic.proof.score import registry as sr
from agentic.proof.types import (
    Claim,
    Observation,
    ObligationKind,
    ObligationSpec,
    PredicateSpec,
    ScopeRef,
)
from storage.inventory_store import InventoryStore


from agentic.proof.errors import make_envelope as _err  # canonical envelope builder


def resolve_scope(
    scope_payload: Dict[str, Any],
) -> Tuple[Optional[ScopeRef], Optional[Dict[str, Any]]]:
    if not isinstance(scope_payload, dict):
        return None, _err(
            "invalid_scope",
            "scope must be a dict with file_ids and optional section_ids",
            remediation="Pass `scope` as a JSON object with at least `file_ids`; see valid_example for the exact shape.",
            valid_example={"file_ids": ["<file_id>"], "section_ids": None, "sealed": False},
        )
    file_ids = scope_payload.get("file_ids")
    if not file_ids or not isinstance(file_ids, list):
        return None, _err(
            "invalid_scope",
            "scope.file_ids must be a non-empty list",
            remediation="Call list_files to discover valid file_ids, then put at least one of them inside `scope.file_ids` as a list.",
            valid_example={"file_ids": ["<file_id>"], "section_ids": None, "sealed": False},
        )
    files_clean = [str(f).strip() for f in file_ids if str(f).strip()]
    if not files_clean:
        return None, _err(
            "invalid_scope",
            "scope.file_ids has no usable entries",
            remediation="Replace empty/blank strings in `scope.file_ids` with real file_ids returned by list_files.",
            valid_example={"file_ids": ["<file_id>"], "section_ids": None, "sealed": False},
        )
    section_ids = scope_payload.get("section_ids")
    section_set: Optional[frozenset[str]] = None
    if section_ids is not None:
        if not isinstance(section_ids, list):
            return None, _err(
                "invalid_scope",
                "scope.section_ids must be a list (or omitted)",
                remediation="Pass `section_ids` as a JSON list of '<file_id>:sec_NNN' strings (from toc), or omit the field entirely.",
                valid_example={"file_ids": ["<file_id>"], "section_ids": ["<file_id>:sec_001"], "sealed": False},
            )
        cleaned = [str(s).strip() for s in section_ids if str(s).strip()]
        if cleaned:
            section_set = frozenset(cleaned)
    sealed = bool(scope_payload.get("sealed", False))
    return ScopeRef(
        file_ids=frozenset(files_clean),
        section_ids=section_set,
        sealed=sealed,
    ), None


def resolve_predicate(
    payload: Dict[str, Any],
) -> Tuple[Optional[PredicateSpec], Optional[Dict[str, Any]]]:
    if not isinstance(payload, dict):
        return None, _err(
            "invalid_predicate",
            "predicate must be a dict",
            remediation="Pass `predicate` as a JSON object with at least `name` and `args`; see valid_example.",
            valid_example={"name": "contains_string", "args": {"pattern": "Premium", "case_sensitive": False}},
        )
    name = payload.get("name")
    if not name:
        return None, _err(
            "invalid_predicate",
            "predicate.name is required",
            remediation="Set `predicate.name` to a registered primitive (contains_string, regex_match, field_equals, numeric_compare, date_compare, type_is, table_cell_contains, section_title_contains, range_in, list_contains, and).",
            valid_example={"name": "contains_string", "args": {"pattern": "Premium", "case_sensitive": False}},
        )
    if name == "and":
        # Accept both top-level ``conjuncts`` and ``args.conjuncts`` —
        # the LLM and the plant disagreed on the shape and tolerating
        # both kept the schema description honest.
        conjuncts_raw = payload.get("conjuncts")
        if not isinstance(conjuncts_raw, list) or not conjuncts_raw:
            args_payload = payload.get("args") or {}
            if isinstance(args_payload, dict):
                conjuncts_raw = args_payload.get("conjuncts")
        if not isinstance(conjuncts_raw, list) or not conjuncts_raw:
            return None, _err(
                "invalid_predicate",
                "and predicate requires non-empty 'conjuncts'",
                remediation="Re-emit the and-predicate with a non-empty `conjuncts` list of two or more sub-predicates (top-level OR nested under `args`).",
                valid_example={"name": "and", "conjuncts": [
                    {"name": "contains_string", "args": {"pattern": "Premium"}},
                    {"name": "contains_string", "args": {"pattern": "USD"}}]},
            )
        conjuncts: List[PredicateSpec] = []
        for child in conjuncts_raw:
            spec, err = resolve_predicate(child)
            if err is not None:
                return None, err
            conjuncts.append(spec)  # type: ignore[arg-type]
        try:
            return pr.build_and_spec(conjuncts), None
        except pr.PredicateError as exc:
            return None, _err(
                "invalid_predicate", str(exc),
                remediation="Inspect the message for the specific args/shape problem; consult the obligation_create schema for required args of each primitive (e.g. regex_match needs an anchored pattern, not bare '.*').",
                valid_example={"name": "and", "conjuncts": [
                    {"name": "contains_string", "args": {"pattern": "Premium"}},
                    {"name": "regex_match", "args": {"pattern": r"USD\s*\d"}}]},
            )
    args = payload.get("args") or {}
    try:
        return pr.build_spec(name, dict(args)), None
    except pr.PredicateError as exc:
        return None, _err(
            "invalid_predicate", str(exc),
            remediation="Fix the predicate args to match the primitive's signature (e.g. contains_string needs `pattern`; numeric_compare needs `field_path`, `op`, `value`); see obligation_create's schema description for the per-primitive args.",
            valid_example={"name": "contains_string", "args": {"pattern": "Premium", "case_sensitive": False}},
        )


def resolve_score(
    payload: Optional[Dict[str, Any]],
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """``None`` payload means no score; returns (ScoreSpec | None, error)."""
    if payload is None:
        return None, None
    if not isinstance(payload, dict):
        return None, _err(
            "invalid_score", "score must be a dict or omitted",
            remediation="Pass `score` as a JSON object {name, args} or omit the field entirely for non-argmax obligations.",
            valid_example={"name": "numeric_amount", "args": {}},
        )
    name = payload.get("name")
    if not name:
        return None, _err(
            "invalid_score", "score.name is required",
            remediation="Set `score.name` to one of numeric_amount / percentage / integer_count / date_iso (text_field is not orderable for argmax).",
            valid_example={"name": "numeric_amount", "args": {}},
        )
    args = payload.get("args") or {}
    try:
        return sr.build_spec(name, dict(args)), None
    except sr.ScoreError as exc:
        return None, _err(
            "invalid_score", str(exc),
            remediation="Fix the score args to match the registered extractor; numeric_amount/percentage/integer_count/date_iso each take args={}.",
            valid_example={"name": "numeric_amount", "args": {}},
        )


def check_scope_resolves(
    scope: ScopeRef, inventory: InventoryStore,
) -> Optional[Dict[str, Any]]:
    """Plant guard: every file_id and section_id in scope exists in
    inventory. Cross-checks against ``inventory.page_store`` directly
    because ``InventoryStore.units`` is a pass-through that doesn't
    validate against ingested content. Missing files would yield an
    empty universe and silently neuter every closure rule — exactly
    the silent-tampering case the gate exists to prevent."""
    known_files = {
        gid.split("/", 1)[0] for gid in inventory.page_store.ids() if "/" in gid
    }
    unknown = sorted(set(scope.file_ids) - known_files)
    if unknown:
        return _err(
            "unknown_file_id", f"unknown file_ids: {unknown}",
            remediation="Call list_files to enumerate valid file_ids in this corpus, then re-issue with one of those ids in scope.file_ids.",
            unknown_file_ids=unknown,
        )
    if scope.section_ids is not None:
        for sid in scope.section_ids:
            sec = inventory.get(sid)
            if sec is None:
                return _err(
                    "unknown_section_id", f"unknown section_id={sid!r}",
                    remediation="Call toc(file_id=...) to refresh the section list for this file, then use one of the returned section_ids (form '<file_id>:sec_NNN').",
                    unknown_section_id=sid,
                )
            # The owning file must also be in scope.file_ids; otherwise
            # an obligation nominally scoped to one file could be closed
            # on evidence drawn from a section that lives in a different
            # file.
            if sec.file_id not in scope.file_ids:
                return _err(
                    "section_outside_file_ids",
                    f"section {sid!r} belongs to file {sec.file_id!r} not in scope.file_ids",
                    remediation=f"Either add {sec.file_id!r} to scope.file_ids, or replace this section_id with one whose file_id is already in scope.file_ids.",
                    section_id=sid,
                    section_file_id=sec.file_id,
                )
    return None


def build_obligation_spec(
    payload: Dict[str, Any],
    *,
    parent_id_override: Optional[str] = None,
    discharges_challenge: Optional[str] = None,
) -> Tuple[Optional[ObligationSpec], Optional[Dict[str, Any]]]:
    if not isinstance(payload, dict):
        return None, _err(
            "invalid_spec", "spec must be a dict",
            remediation="Pass `spec` as a JSON object containing kind, scope, unit_type, and predicate (see obligation_create's schema description for the exact shape).",
            valid_example={
                "kind": "exists",
                "scope": {"file_ids": ["<file_id>"], "section_ids": None, "sealed": False},
                "unit_type": "file",
                "predicate": {"name": "contains_string", "args": {"pattern": "Premium"}},
            },
        )
    kind_raw = payload.get("kind")
    try:
        kind = ObligationKind(kind_raw)
    except ValueError:
        return None, _err(
            "invalid_kind", f"kind={kind_raw!r} not recognised",
            remediation="Set `kind` to one of exists / count / set / forall / negation / argmax based on the question shape.",
            valid_example={"kind": "exists"},
        )
    scope, err = resolve_scope(payload.get("scope") or {})
    if err is not None:
        return None, err
    unit_type = payload.get("unit_type")
    if unit_type not in ("file", "section"):
        return None, _err(
            "invalid_unit_type", "unit_type must be 'file' or 'section'",
            remediation="Set unit_type to either 'file' or 'section' (no other values are accepted; 'page' is not allowed).",
            valid_example={"unit_type": "file"},
        )
    predicate, err = resolve_predicate(payload.get("predicate") or {})
    if err is not None:
        return None, err
    score, err = resolve_score(payload.get("score"))
    if err is not None:
        return None, err
    polarity = payload.get("polarity") or "positive"
    if polarity not in ("positive", "negative"):
        return None, _err(
            "invalid_polarity", "polarity must be 'positive' or 'negative'",
            remediation="Set `polarity` to 'positive' (or omit the field — default is positive). 'negative' is not supported; use kind='negation' for absence questions.",
            valid_example={"polarity": "positive"},
        )
    # v1 has no certifying claim shape for negative polarity — see
    # ObligationKind.NEGATION for "predicate fails for all units".
    if polarity == "negative":
        return None, _err(
            "negative_polarity_unsupported",
            "polarity='negative' is not supported in v1; use kind='negation' instead",
            remediation="Drop `polarity` (or set it to 'positive') and switch `kind` to 'negation' if you want to prove the predicate fails for every unit in scope.",
            valid_example={"kind": "negation", "polarity": "positive"},
        )
    required = payload.get("required", True)
    parent_id = parent_id_override or payload.get("parent_id")
    sealed_override = bool(payload.get("sealed_scope_override", False))
    tie_policy = payload.get("tie_policy") or "first"
    if tie_policy not in ("first", "all", "error"):
        return None, _err(
            "invalid_tie_policy", "tie_policy must be one of first/all/error",
            remediation="Set `tie_policy` to one of 'first' (default), 'all', or 'error' — applies to argmax obligations only.",
            valid_example={"tie_policy": "first"},
        )
    derived_by_raw = payload.get("derived_by") or "user_constraint"
    if derived_by_raw not in (
        "root", "and_split", "scope_partition", "case_split",
        "map_over_domain", "user_constraint", "challenge_replacement",
    ):
        return None, _err(
            "invalid_derived_by", "derived_by not recognised",
            remediation="Drop `derived_by` from the spec (the plant sets it automatically based on parent_id / discharges_challenge / decomposition rule).",
        )
    spec = ObligationSpec(
        kind=kind,
        scope=scope,                           # type: ignore[arg-type]
        unit_type=unit_type,                   # type: ignore[arg-type]
        predicate=predicate,                   # type: ignore[arg-type]
        score=score,                           # type: ignore[arg-type]
        polarity=polarity,                     # type: ignore[arg-type]
        required=bool(required),
        parent_id=parent_id,
        derived_by=derived_by_raw,             # type: ignore[arg-type]
        sealed_scope_override=sealed_override,
        tie_policy=tie_policy,                 # type: ignore[arg-type]
    )
    return spec, None


def build_claim_candidate(
    plant: Any,
    observation: Observation,
    payload: Dict[str, Any],
) -> Tuple[Optional[Claim], Optional[Dict[str, Any]]]:
    """Forward to claim_validator with the plant-side helpers it needs.

    The validator must stay plant-import-free (so it can be exercised
    without a full plant in unit tests), so we hand it the resolvers
    and ``_err`` factory by injection."""
    from agentic.proof.evidence.claim_validator import validate_claim_candidate
    return validate_claim_candidate(
        payload, observation,
        plant=plant,
        _err=_err,
        resolve_scope=resolve_scope,
        resolve_predicate=resolve_predicate,
        resolve_score=resolve_score,
    )


