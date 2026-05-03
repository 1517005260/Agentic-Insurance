"""Declarative decomposition-rule registry.

Each :class:`RuleContract` enumerates exactly what an
``obligation_decompose`` call must satisfy for one ``rule_id``. The
plant calls :func:`validate_rule` to dispatch; the contract returns
``None`` on accept or an :class:`ErrorEnvelope` dict on reject. This
replaces the inline if/elif tree in plant.py so adding a new rule
touches one row in this file rather than four call sites.

Rules registered:

* ``and_split``        — children share parent's scope; their
  predicates AND together to the parent's predicate; AND must contain
  at least one content-bearing conjunct.
* ``scope_partition``  — children share parent's predicate + kind;
  their scopes form a disjoint cover of the parent's inventory units.
* ``case_split``       — exactly two children with parent's full spec
  (kind / unit_type / scope / predicate).
* ``map_over_domain``  — parent's domain is enumerated; children are
  materialised lazily. The contract here is **structural only** (no
  ``child_specs``); the runtime materialisation is handled inline by
  plant._handle_map_over_domain.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Literal, Optional

from agentic.proof.types import (
    DecompositionRule,
    Obligation,
    ObligationSpec,
)
from agentic.proof import predicate as pr


# ----------------------------------------------------------- types


_ErrFn = Callable[..., Dict[str, Any]]
ValidatorFn = Callable[[Obligation, List[ObligationSpec], Any, _ErrFn], Optional[Dict[str, Any]]]


@dataclass(frozen=True)
class RuleContract:
    rule_id: DecompositionRule
    validate: ValidatorFn        # (parent, children, inventory, _err) -> Optional[ErrorEnvelope]
    requires_child_specs: bool   # map_over_domain is the only False


# ----------------------------------------------------------- per-rule validators


def _validate_and_split(
    parent: Obligation,
    children: List[ObligationSpec],
    inventory: Any,
    _err: _ErrFn,
) -> Optional[Dict[str, Any]]:
    child_preds = [c.predicate for c in children]
    try:
        synthetic = pr.build_and_spec(child_preds)
    except pr.PredicateError as exc:
        return _err(
            "invalid_and_split", str(exc),
            remediation="Inspect the message for the specific shape problem with one of the child predicates; ensure each child predicate is a valid primitive with the right args.",
        )
    if pr.serialize_spec(synthetic) != pr.serialize_spec(parent.spec.predicate):
        return _err(
            "and_split_mismatch",
            "synthesised AND of children does not equal parent predicate",
            remediation="Ensure the AND of children's predicates equals the parent's predicate. If the parent predicate is and(P,Q), the children must be exactly two specs with predicates P and Q (in any order).",
        )
    if not pr.has_content_conjunct(parent.spec.predicate):
        return _err(
            "and_split_no_content", "parent predicate must contain a content conjunct",
            remediation="and_split requires the parent's predicate to contain at least one content-bearing conjunct (contains_string / regex_match / table_cell_contains / etc.). If the parent has only structural predicates, use scope_partition or case_split instead.",
        )
    for c in children:
        if c.unit_type != parent.spec.unit_type:
            return _err(
                "unit_type_mismatch", "child unit_type must match parent",
                remediation=f"Set every child's unit_type to {parent.spec.unit_type!r} (matching the parent obligation).",
            )
        if c.scope != parent.spec.scope:
            return _err(
                "scope_mismatch_in_and_split",
                "and_split children must share parent's scope",
                remediation="Copy the parent obligation's scope verbatim into every child spec — and_split children share scope; only their predicates differ.",
            )
    return None


def _validate_scope_partition(
    parent: Obligation,
    children: List[ObligationSpec],
    inventory: Any,
    _err: _ErrFn,
) -> Optional[Dict[str, Any]]:
    parent_units = set(inventory.units(
        parent.spec.unit_type,
        file_ids=list(parent.spec.scope.file_ids),
        section_ids=list(parent.spec.scope.section_ids) if parent.spec.scope.section_ids else None,
    ))
    child_universe: set[str] = set()
    for c in children:
        if c.kind != parent.spec.kind:
            return _err(
                "scope_partition_kind_mismatch",
                "scope_partition children must share parent's kind; "
                "aggregating a stronger parent from weaker children is unsound",
                remediation=f"Set every child's kind to {parent.spec.kind.value!r} (matching the parent); scope_partition only re-scopes, it does not change kind.",
            )
        if c.unit_type != parent.spec.unit_type:
            return _err(
                "unit_type_mismatch", "child unit_type must match parent",
                remediation=f"Set every child's unit_type to {parent.spec.unit_type!r} (matching the parent).",
            )
        if pr.serialize_spec(c.predicate) != pr.serialize_spec(parent.spec.predicate):
            return _err(
                "scope_partition_predicate_mismatch",
                "child predicates must equal parent predicate",
                remediation="Copy the parent's predicate verbatim into every child spec — scope_partition only narrows scope; predicate stays identical.",
            )
        cu = set(inventory.units(
            c.unit_type,
            file_ids=list(c.scope.file_ids),
            section_ids=list(c.scope.section_ids) if c.scope.section_ids else None,
        ))
        overlap = cu & child_universe
        if overlap:
            return _err(
                "scope_partition_overlap",
                "scope_partition children must have disjoint unit sets",
                remediation="Partition the parent's units across children so no two children share any unit (e.g. split file_ids non-overlappingly, or pick disjoint section_ids). Drop the listed `overlapping_units` from one child.",
                overlapping_units=sorted(overlap)[:10],
            )
        child_universe |= cu
    if child_universe != parent_units:
        missing = sorted(parent_units - child_universe)
        extra = sorted(child_universe - parent_units)
        return _err(
            "scope_partition_coverage",
            f"union of child scopes != parent scope (missing {len(missing)} unit(s), extra {len(extra)})",
            remediation="Add or expand children so the union of their unit sets equals the parent's universe — see `missing_units` for the gap and `extra_units` for any out-of-domain ids.",
            missing_units=missing[:10],
            extra_units=extra[:10],
        )
    return None


def _validate_case_split(
    parent: Obligation,
    children: List[ObligationSpec],
    inventory: Any,
    _err: _ErrFn,
) -> Optional[Dict[str, Any]]:
    if len(children) != 2:
        return _err(
            "case_split_children", "case_split requires exactly two children",
            remediation=f"Re-issue obligation_decompose with exactly two child_specs (got {len(children)}). Use scope_partition for >2 disjoint scopes.",
        )
    for c in children:
        if c.kind != parent.spec.kind:
            return _err(
                "case_split_kind_mismatch",
                "case_split children must share parent's kind",
                remediation=f"Set every child's kind to {parent.spec.kind.value!r} (matching the parent).",
            )
        if c.unit_type != parent.spec.unit_type:
            return _err(
                "unit_type_mismatch", "child unit_type must match parent",
                remediation=f"Set every child's unit_type to {parent.spec.unit_type!r} (matching the parent).",
            )
        if c.scope != parent.spec.scope:
            return _err(
                "case_split_scope_mismatch",
                "case_split children must share parent's scope",
                remediation="Copy the parent obligation's scope verbatim into every child spec — case_split children share scope; the difference is only in evidence routes.",
            )
        if pr.serialize_spec(c.predicate) != pr.serialize_spec(parent.spec.predicate):
            return _err(
                "case_split_predicate_mismatch",
                "case_split children must share parent's predicate",
                remediation="Copy the parent obligation's predicate verbatim into every child spec — case_split children differ only in how they're proven, not in what they prove.",
            )
    return None


def _noop(*_args, **_kwargs) -> Optional[Dict[str, Any]]:
    """Map_over_domain has no child_specs; the runtime materialisation
    is handled by plant._handle_map_over_domain. This validator is a
    no-op so the dispatcher can still resolve the rule id."""
    return None


# ----------------------------------------------------------- registry


_RULES: Dict[DecompositionRule, RuleContract] = {
    "and_split":       RuleContract("and_split",       _validate_and_split,       True),
    "scope_partition": RuleContract("scope_partition", _validate_scope_partition, True),
    "case_split":      RuleContract("case_split",      _validate_case_split,      True),
    "map_over_domain": RuleContract("map_over_domain", _noop,                     False),
}


# ----------------------------------------------------------- public API


def all_rule_ids() -> FrozenSet[str]:
    return frozenset(_RULES.keys())


def get(rule_id: str) -> Optional[RuleContract]:
    return _RULES.get(rule_id)  # type: ignore[arg-type]


def validate_rule(
    rule_id: str,
    parent: Obligation,
    children: List[ObligationSpec],
    inventory: Any,
    _err: _ErrFn,
) -> Optional[Dict[str, Any]]:
    """Dispatch to the registered :class:`RuleContract`. Returns
    ``None`` on accept or an envelope dict on reject. Unknown rule
    ids are caller-validated upstream — we treat them defensively as
    a noop here."""
    contract = _RULES.get(rule_id)  # type: ignore[arg-type]
    if contract is None:
        return None
    return contract.validate(parent, children, inventory, _err)
