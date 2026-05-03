"""``numeric_compare``: float comparison of a payload field against a
constant. Universal on unknown ops, NaN/Inf RHS, or ``!=`` against
infinity (which is true for every finite reading).
"""
from typing import Any, Dict

from agentic.proof.predicate.helpers import (
    NUMERIC_OPS,
    _bad_number,
    _coerce_str,
    _read_field,
)
from agentic.proof.predicate.registry import PrimitiveSchema


def _numeric_universal(args: Dict[str, Any]) -> bool:
    op = args.get("op")
    if op not in {">", ">=", "<", "<=", "==", "!="}:
        return True   # unknown op: treat as universal (rejects)
    if _bad_number(args.get("value")):
        return True
    if op == "!=" and _coerce_str(args.get("value")) in {"-inf", "inf", "+inf"}:
        return True
    return False


def _numeric_eval(payload: Dict[str, Any], args: Dict[str, Any]) -> bool:
    raw = _read_field(payload, args["field_path"])
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return False
    rhs = float(args["value"])
    op = args["op"]
    return NUMERIC_OPS[op](v, rhs)


SCHEMA = PrimitiveSchema(
    name="numeric_compare",
    is_structural=False,
    applicable_unit_types=frozenset({"file", "section"}),
    required_args=("field_path", "op", "value"),
    optional_defaults=(),
    canonicalize=lambda a: (
        ("field_path", _coerce_str(a["field_path"])),
        ("op", _coerce_str(a["op"])),
        ("value", float(a["value"])),
    ),
    is_universal=_numeric_universal,
    evaluate_field="payload",
)
