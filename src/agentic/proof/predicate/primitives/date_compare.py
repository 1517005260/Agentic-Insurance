"""``date_compare``: ISO-date comparison via POSIX timestamps. Both
field value and RHS must parse with ``datetime.fromisoformat`` (with a
``"Z"`` suffix tolerated). Universal on unknown ops or empty RHS.
"""
from typing import Any, Dict

from agentic.proof.predicate.helpers import (
    NUMERIC_OPS,
    _coerce_str,
    _parse_iso_date,
    _read_field,
)
from agentic.proof.predicate.registry import PrimitiveSchema


def _date_universal(args: Dict[str, Any]) -> bool:
    op = args.get("op")
    if op not in {">", ">=", "<", "<=", "==", "!="}:
        return True
    return not _coerce_str(args.get("value", "")).strip()


def _date_eval(payload: Dict[str, Any], args: Dict[str, Any]) -> bool:
    a = _parse_iso_date(_coerce_str(_read_field(payload, args["field_path"])))
    b = _parse_iso_date(_coerce_str(args["value"]))
    if a is None or b is None:
        return False
    return NUMERIC_OPS[args["op"]](a, b)


SCHEMA = PrimitiveSchema(
    name="date_compare",
    is_structural=False,
    applicable_unit_types=frozenset({"file", "section"}),
    required_args=("field_path", "op", "value"),
    optional_defaults=(),
    canonicalize=lambda a: (
        ("field_path", _coerce_str(a["field_path"])),
        ("op", _coerce_str(a["op"])),
        ("value", _coerce_str(a["value"])),
    ),
    is_universal=_date_universal,
    evaluate_field="payload",
)
