"""Declarative repair-contract registry.

Each :class:`RepairContract` enumerates exactly which fields a
replacement obligation may change for one ``repair_kind`` plus any
extra structural rules (root-only, sealed scope, etc.). The plant
calls :func:`validate_replacement` to dispatch; the contract returns
``(meta, None)`` on accept or ``(None, ErrorEnvelope)`` on reject.

Repair kinds:

* ``scope_too_narrow`` — replacement must STRICTLY widen the unit set;
  predicate / kind / unit_type / polarity / score must match target.
* ``scope_too_broad``  — replacement must STRICTLY narrow the unit set;
  same field-preservation rules; sealed scopes refuse narrowing.
* ``predicate_mismatch`` — replacement must propose a DIFFERENT
  predicate; everything else must match. Forbidden on root.
* ``wrong_question_kind`` — root-only; replacement proposes a
  different kind. Cap-bearing (one per session).
* ``missing_subobligation`` — discharged by ``obligation_decompose``
  not ``obligation_create``; this contract returns the redirect error.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from agentic.proof.types import (
    Challenge,
    Obligation,
    ObligationSpec,
    RepairKind,
)
from agentic.proof import predicate as pr


_ErrFn = Callable[..., Dict[str, Any]]


@dataclass(frozen=True)
class RepairContract:
    repair_kind: RepairKind
    """Discriminator name."""
    discharge_via: str               # "create" or "decompose"
    """Which tool entry point may discharge a challenge of this kind."""
    cap_per_session: Optional[int]   # only wrong_question_kind has cap=1


# ----------------------------------------------------------- envelope check


def envelope_check(target: Obligation, spec: ObligationSpec, _err: _ErrFn) -> Optional[Dict[str, Any]]:
    """Fields a replacement MUST inherit from the target — single
    source of truth shared by every repair_kind path."""
    if spec.required != target.spec.required:
        return _err(
            "required_changed",
            "replacement may not change `required`; the gate forbids dropping a required obligation",
            remediation="Re-emit the spec with `required` matching the original obligation (omit the field to inherit, or set it to the same boolean as the target).",
        )
    if spec.parent_id != target.spec.parent_id:
        return _err(
            "parent_id_changed",
            "replacement may not change `parent_id`; this would re-parent the proof tree",
            remediation="Set `parent_id` in the replacement spec to the same value as the target obligation (or omit it for a root replacement).",
        )
    if spec.sealed_scope_override != target.spec.sealed_scope_override:
        return _err(
            "sealed_scope_override_changed",
            "replacement may not change `sealed_scope_override`",
            remediation="Re-emit the replacement spec with `sealed_scope_override` set to the same boolean as the target obligation.",
        )
    if spec.tie_policy != target.spec.tie_policy:
        return _err(
            "tie_policy_changed", "replacement may not change `tie_policy`",
            remediation="Re-emit the replacement spec with `tie_policy` matching the target obligation (e.g. 'first').",
        )
    if spec.scope.sealed != target.spec.scope.sealed:
        return _err(
            "sealed_changed",
            "replacement may not change `scope.sealed`; ΓNegation depends on it",
            remediation="Set `scope.sealed` in the replacement spec to the same boolean as the target's scope.sealed.",
        )
    return None


# ----------------------------------------------------------- validators


def _score_equal(spec: ObligationSpec, target: Obligation) -> bool:
    if (spec.score is None) != (target.spec.score is None):
        return False
    if spec.score is None:
        return True
    return (
        spec.score.name == target.spec.score.name
        and spec.score.args == target.spec.score.args
    )


def _validate_scope_too_narrow(
    spec: ObligationSpec,
    target: Obligation,
    inventory: Any,
    target_units: set,
    spec_units: set,
    target_pred_hash: str,
    spec_pred_hash: str,
    _err: _ErrFn,
) -> Optional[Dict[str, Any]]:
    if not target_units < spec_units:
        return _err(
            "scope_not_widened",
            "replacement unit set must be a strict superset of the original",
            remediation="Widen scope.file_ids (or scope.section_ids) so the resolved unit set strictly contains the original — add files/sections, do not remove.",
        )
    if (
        spec.kind != target.spec.kind
        or spec.unit_type != target.spec.unit_type
        or spec_pred_hash != target_pred_hash
        or spec.polarity != target.spec.polarity
        or not _score_equal(spec, target)
    ):
        return _err(
            "non_scope_field_changed",
            "scope_too_narrow replacement may only change scope",
            remediation="Copy kind, unit_type, predicate, polarity, and score verbatim from the target obligation; only the scope may change in a scope_too_narrow repair.",
        )
    return None


def _validate_scope_too_broad(
    spec: ObligationSpec,
    target: Obligation,
    inventory: Any,
    target_units: set,
    spec_units: set,
    target_pred_hash: str,
    spec_pred_hash: str,
    _err: _ErrFn,
) -> Optional[Dict[str, Any]]:
    if target.spec.scope.sealed:
        return _err(
            "scope_too_broad_on_sealed", "cannot narrow a sealed scope",
            remediation="A sealed scope cannot be narrowed (the seal preserves universal closure). Pick a different repair_kind, or accept the current scope.",
        )
    if not spec_units < target_units:
        return _err(
            "scope_not_narrowed",
            "replacement unit set must be a strict subset of the original",
            remediation="Narrow scope.file_ids (or scope.section_ids) so the resolved unit set is strictly inside the original — remove files/sections, do not add.",
        )
    if (
        spec.kind != target.spec.kind
        or spec.unit_type != target.spec.unit_type
        or spec_pred_hash != target_pred_hash
        or spec.polarity != target.spec.polarity
        or not _score_equal(spec, target)
    ):
        return _err(
            "non_scope_field_changed",
            "scope_too_broad replacement may only change scope",
            remediation="Copy kind, unit_type, predicate, polarity, and score verbatim from the target obligation; only the scope may change in a scope_too_broad repair.",
        )
    return None


def _validate_predicate_mismatch(
    spec: ObligationSpec,
    target: Obligation,
    inventory: Any,
    target_units: set,
    spec_units: set,
    target_pred_hash: str,
    spec_pred_hash: str,
    _err: _ErrFn,
) -> Optional[Dict[str, Any]]:
    if spec_pred_hash == target_pred_hash:
        return _err(
            "predicate_unchanged",
            "predicate_mismatch replacement must propose a different predicate",
            remediation="Change the predicate (name or args) in the replacement spec — predicate_mismatch is a no-op if the predicate stays identical.",
        )
    if (
        spec.kind != target.spec.kind
        or spec.unit_type != target.spec.unit_type
        or spec.scope != target.spec.scope
        or spec.polarity != target.spec.polarity
        or not _score_equal(spec, target)
    ):
        return _err(
            "non_predicate_field_changed",
            "predicate_mismatch replacement may only change predicate",
            remediation="Copy kind, unit_type, scope, polarity, and score verbatim from the target obligation; only the predicate may change in a predicate_mismatch repair.",
        )
    return None


def _validate_wrong_question_kind(
    spec: ObligationSpec,
    target: Obligation,
    inventory: Any,
    target_units: set,
    spec_units: set,
    target_pred_hash: str,
    spec_pred_hash: str,
    _err: _ErrFn,
) -> Optional[Dict[str, Any]]:
    if not target.is_root:
        return _err(
            "wrong_kind_only_on_root", "wrong_question_kind only applies to root",
            remediation="Use predicate_mismatch / scope_too_narrow / scope_too_broad / missing_subobligation to repair non-root obligations; wrong_question_kind is reserved for the root.",
        )
    if spec.kind == target.spec.kind:
        return _err(
            "kind_unchanged",
            "wrong_question_kind replacement must propose a different kind",
            remediation="Change `kind` in the replacement spec to a different value (e.g. exists -> count) — wrong_question_kind is a no-op if kind stays identical.",
        )
    return None


def _validate_missing_subobligation(
    spec: ObligationSpec,
    target: Obligation,
    inventory: Any,
    target_units: set,
    spec_units: set,
    target_pred_hash: str,
    spec_pred_hash: str,
    _err: _ErrFn,
) -> Optional[Dict[str, Any]]:
    return _err(
        "wrong_path", "missing_subobligation discharges via obligation_decompose",
        remediation="Discharge a missing_subobligation challenge by calling obligation_decompose(parent_id=..., discharges_challenge=challenge_id), NOT obligation_create.",
    )


# ----------------------------------------------------------- registry


_REPAIRS: Dict[RepairKind, RepairContract] = {
    "scope_too_narrow":     RepairContract("scope_too_narrow",     "create",     None),
    "scope_too_broad":      RepairContract("scope_too_broad",      "create",     None),
    "predicate_mismatch":   RepairContract("predicate_mismatch",   "create",     None),
    "wrong_question_kind":  RepairContract("wrong_question_kind",  "create",     1),
    "missing_subobligation": RepairContract("missing_subobligation", "decompose", None),
}


_VALIDATORS: Dict[RepairKind, Callable[..., Optional[Dict[str, Any]]]] = {
    "scope_too_narrow":      _validate_scope_too_narrow,
    "scope_too_broad":       _validate_scope_too_broad,
    "predicate_mismatch":    _validate_predicate_mismatch,
    "wrong_question_kind":   _validate_wrong_question_kind,
    "missing_subobligation": _validate_missing_subobligation,
}


# ----------------------------------------------------------- public API


def all_repair_kinds() -> FrozenSet[str]:
    return frozenset(_REPAIRS.keys())


def get(repair_kind: str) -> Optional[RepairContract]:
    return _REPAIRS.get(repair_kind)  # type: ignore[arg-type]


def validate_replacement(
    spec: ObligationSpec,
    target: Obligation,
    repair_kind: RepairKind,
    inventory: Any,
    _err: _ErrFn,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run repair-kind-specific validation. Returns
    ``(meta, error)``: meta on accept, error on reject — each
    branch returns at most one of the two non-None.
    """
    env_err = envelope_check(target, spec, _err)
    if env_err is not None:
        return None, env_err

    target_pred_hash = pr.serialize_spec(target.spec.predicate)
    spec_pred_hash = pr.serialize_spec(spec.predicate)
    target_units = set(inventory.units(
        target.spec.unit_type,
        file_ids=list(target.spec.scope.file_ids),
        section_ids=list(target.spec.scope.section_ids) if target.spec.scope.section_ids else None,
    ))
    spec_units = set(inventory.units(
        spec.unit_type,
        file_ids=list(spec.scope.file_ids),
        section_ids=list(spec.scope.section_ids) if spec.scope.section_ids else None,
    ))

    validator = _VALIDATORS.get(repair_kind)
    if validator is None:
        return None, _err(
            "unknown_repair_kind", f"repair_kind={repair_kind!r} not registered",
            remediation="Set repair_kind to one of scope_too_narrow / scope_too_broad / predicate_mismatch / missing_subobligation / wrong_question_kind.",
        )
    err = validator(
        spec, target, inventory, target_units, spec_units,
        target_pred_hash, spec_pred_hash, _err,
    )
    if err is not None:
        return None, err
    meta: Dict[str, Any] = {
        "target_obligation_id": target.id,
        "repair_kind": repair_kind,
    }
    if repair_kind == "predicate_mismatch" and target.is_root:
        meta["forbids_root_predicate_mismatch"] = True
    return meta, None
