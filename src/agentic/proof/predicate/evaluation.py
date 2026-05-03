"""Evaluation dispatcher and applicability checks.

The dispatcher decodes a ``PredicateSpec`` to the matching primitive
module's ``_eval`` function. AND short-circuits on the first ``False``.
"""
from typing import Any, Dict

from agentic.proof.predicate.helpers import _coerce_str
from agentic.proof.predicate.primitives import (
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
from agentic.proof.predicate.registry import _PRIMITIVES
from agentic.proof.types import PredicateSpec


# Maps primitive name -> evaluator callable. The callable's first arg
# is either ``unit_text`` (for ``evaluate_field == "text"``) or the
# whole ``unit_payload`` dict (for ``evaluate_field == "payload"``);
# the dispatcher picks based on the primitive's schema.
_TEXT_EVALS: Dict[str, Any] = {
    "contains_string": contains_string._contains_string_eval,
    "regex_match": regex_match._regex_eval,
}

_PAYLOAD_EVALS: Dict[str, Any] = {
    "field_equals": field_equals._field_equals_eval,
    "numeric_compare": numeric_compare._numeric_eval,
    "date_compare": date_compare._date_eval,
    "type_is": type_is._type_is_eval,
    "table_cell_contains": table_cell_contains._table_cell_eval,
    "section_title_contains": section_title_contains._section_title_eval,
    "range_in": range_in._range_eval,
    "list_contains": list_contains._list_contains_eval,
}


def evaluate(spec: PredicateSpec, unit_payload: Dict[str, Any]) -> bool:
    """Evaluate a spec against a unit payload. ``unit_payload`` is what
    the plant feeds the predicate; the plant assembles it from
    PageStore / InventoryStore.

    Conjunction short-circuits on the first False.
    """
    if spec.name == "and":
        for n, a in spec.args:
            if not evaluate(PredicateSpec(name=n, args=a), unit_payload):
                return False
        return True
    schema = _PRIMITIVES.get(spec.name)
    if schema is None:
        return False
    args = dict(spec.args)
    if schema.evaluate_field == "text":
        evaluator = _TEXT_EVALS.get(schema.name)
        if evaluator is None:
            return False
        return evaluator(unit_payload.get("text", ""), args)
    if schema.evaluate_field == "payload":
        evaluator = _PAYLOAD_EVALS.get(schema.name)
        if evaluator is None:
            return False
        return evaluator(unit_payload, args)
    return False


def predicate_applicable(spec: PredicateSpec, unit_type: str) -> bool:
    """True iff every primitive in ``spec`` is applicable to ``unit_type``.

    For composites, all conjuncts must be applicable.
    """
    if spec.name == "and":
        for child_name, child_args in spec.args:
            child = PredicateSpec(name=child_name, args=child_args)
            if not predicate_applicable(child, unit_type):
                return False
        return True
    schema = _PRIMITIVES.get(spec.name)
    if schema is None:
        return False
    return unit_type in schema.applicable_unit_types


def applicable_to_obligation_unit_type(spec: PredicateSpec, obligation_unit_type: str) -> bool:
    """Helper for type_is universality check: pass the obligation's
    unit_type into the universal probe to detect ``type_is(section)``
    when obligation.unit_type=="section".
    """
    if spec.name == "and":
        return all(
            applicable_to_obligation_unit_type(PredicateSpec(name=n, args=a), obligation_unit_type)
            for n, a in spec.args
        )
    if spec.name != "type_is":
        return True
    return _coerce_str(dict(spec.args).get("unit_type", "")) != obligation_unit_type
