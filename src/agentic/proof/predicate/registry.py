"""Registered predicate primitives + spec construction.

Ten primitives are registered out of the box. Each primitive lives in
its own module under ``predicate/primitives/`` and exports a
``SCHEMA`` constant carrying:

* ``applicable_unit_types`` — guards ``predicate_applicable`` checks.
* ``is_universal_on(args)``  — conservative blacklist that rejects
  trivially-true patterns at registration time.
* ``is_structural``         — splits structural-only conjuncts (type_is,
  section_title_contains) from content-bearing ones, enforced in
  ``and_split`` decomposition.
* ``canonicalize(args)``    — canonical tuple-of-pairs used to hash
  and compare.
* ``evaluate_field``        — payload key the dispatcher reads when
  invoking the primitive's eval callback.

Algebra (AND composition, serialisation, entailment, structural
checks) is in ``algebra.py``; the runtime dispatcher lives in
``evaluation.py``.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Tuple

from agentic.proof.types import PredicateSpec


@dataclass(frozen=True)
class PrimitiveSchema:
    """Static description of one predicate primitive.

    ``required_args`` are the keys that must appear in the user-supplied
    ``args`` dict. ``optional_defaults`` is filled in by the plant before
    canonicalisation so a caller can omit defaultable knobs (e.g.,
    ``case_sensitive=False`` for ``contains_string``).
    """

    name: str
    is_structural: bool
    applicable_unit_types: FrozenSet[str]
    required_args: Tuple[str, ...]
    optional_defaults: Tuple[Tuple[str, Any], ...]
    canonicalize: Callable[[Dict[str, Any]], Tuple[Tuple[str, Any], ...]]
    is_universal: Callable[[Dict[str, Any]], bool]
    evaluate_field: str        # which payload field to read on a unit; "" if N/A


class PredicateError(ValueError):
    """Raised when a spec is malformed, references unknown names, or
    is universal under the conservative blacklist."""


# Imported after ``PrimitiveSchema`` is defined so each primitive
# module can ``from agentic.proof.predicate.registry import PrimitiveSchema``
# while this module is still being initialised.
from agentic.proof.predicate.primitives import (  # noqa: E402
    contains_string,
    date_compare,
    field_equals,
    list_contains,
    numeric_compare,
    range_in,
    regex_match,
    section_title_contains,
    table_cell_contains,
    type_is,
)


_PRIMITIVES: Dict[str, PrimitiveSchema] = {
    p.name: p
    for p in (
        contains_string.SCHEMA,
        regex_match.SCHEMA,
        field_equals.SCHEMA,
        numeric_compare.SCHEMA,
        date_compare.SCHEMA,
        type_is.SCHEMA,
        table_cell_contains.SCHEMA,
        section_title_contains.SCHEMA,
        range_in.SCHEMA,
        list_contains.SCHEMA,
    )
}


def primitives() -> Dict[str, PrimitiveSchema]:
    """Return the registered primitives keyed by name (read-only view).

    Tools and tests can introspect the set; mutations are not supported
    in v1 — registry is static for soundness theorem clarity.
    """
    return dict(_PRIMITIVES)


def build_spec(name: str, args: Dict[str, Any]) -> PredicateSpec:
    """Validate + canonicalize a primitive spec.

    Raises :class:`PredicateError` if the primitive is unknown, args are
    incomplete, or the predicate is universal on its declared unit_type.
    For ``and_(...)``, see :func:`agentic.proof.predicate.build_and_spec`.
    """
    if name == "and":
        raise PredicateError("Use build_and_spec(conjuncts=[...]) for AND composition.")
    schema = _PRIMITIVES.get(name)
    if schema is None:
        raise PredicateError(f"Unknown predicate primitive: {name!r}")
    missing = [k for k in schema.required_args if k not in args]
    if missing:
        raise PredicateError(f"{name} missing required arg(s): {missing}")
    merged = dict(schema.optional_defaults)
    merged.update(args)
    canonical = schema.canonicalize(merged)
    canonical_args = dict(canonical)
    if schema.is_universal(canonical_args):
        raise PredicateError(
            f"{name} would always evaluate True under the supplied args; "
            f"reject as universal."
        )
    return PredicateSpec(name=name, args=canonical)
