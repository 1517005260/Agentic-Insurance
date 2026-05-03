"""``table_cell_contains``: case-insensitive substring scan over the
``table_rows`` payload entry, restricted to one named column. Universal
if either ``column_name`` is blank or ``pattern`` is empty.
"""
from typing import Any, Dict

from agentic.proof.predicate.helpers import _coerce_str
from agentic.proof.predicate.registry import PrimitiveSchema


def _table_cell_universal(args: Dict[str, Any]) -> bool:
    return (
        not _coerce_str(args.get("column_name", "")).strip()
        or _coerce_str(args.get("pattern", "")) == ""
    )


def _table_cell_eval(payload: Dict[str, Any], args: Dict[str, Any]) -> bool:
    rows = payload.get("table_rows") or []
    col = args["column_name"]
    pat = _coerce_str(args["pattern"])
    pat_l = pat.lower()
    for row in rows:
        cell = _coerce_str(row.get(col, "")).lower()
        if pat_l in cell:
            return True
    return False


SCHEMA = PrimitiveSchema(
    name="table_cell_contains",
    is_structural=False,
    applicable_unit_types=frozenset({"file", "section"}),
    required_args=("column_name", "pattern"),
    optional_defaults=(),
    canonicalize=lambda a: (
        ("column_name", _coerce_str(a["column_name"])),
        ("pattern", _coerce_str(a["pattern"])),
    ),
    is_universal=_table_cell_universal,
    evaluate_field="payload",
)
