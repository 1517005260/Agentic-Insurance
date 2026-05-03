"""``list_contains``: case-insensitive substring scan over an iterable
payload field. Universal if either ``field_path`` is blank or
``item_pattern`` is empty.
"""
from typing import Any, Dict

from agentic.proof.predicate.helpers import _coerce_str, _read_field
from agentic.proof.predicate.registry import PrimitiveSchema


def _list_contains_universal(args: Dict[str, Any]) -> bool:
    return (
        not _coerce_str(args.get("field_path", "")).strip()
        or _coerce_str(args.get("item_pattern", "")) == ""
    )


def _list_contains_eval(payload: Dict[str, Any], args: Dict[str, Any]) -> bool:
    items = _read_field(payload, args["field_path"]) or []
    pat = _coerce_str(args["item_pattern"]).lower()
    for it in items:
        if pat in _coerce_str(it).lower():
            return True
    return False


SCHEMA = PrimitiveSchema(
    name="list_contains",
    is_structural=False,
    applicable_unit_types=frozenset({"file", "section"}),
    required_args=("field_path", "item_pattern"),
    optional_defaults=(),
    canonicalize=lambda a: (
        ("field_path", _coerce_str(a["field_path"])),
        ("item_pattern", _coerce_str(a["item_pattern"])),
    ),
    is_universal=_list_contains_universal,
    evaluate_field="payload",
)
