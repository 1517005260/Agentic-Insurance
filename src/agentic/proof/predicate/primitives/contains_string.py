"""``contains_string``: case-insensitive (default) substring match on
``unit.text``. Universal iff ``pattern`` is empty.
"""
from typing import Any, Dict

from agentic.proof.predicate.helpers import _coerce_str
from agentic.proof.predicate.registry import PrimitiveSchema


def _contains_string_universal(args: Dict[str, Any]) -> bool:
    return _coerce_str(args.get("pattern", "")) == ""


def _contains_string_eval(unit_text: str, args: Dict[str, Any]) -> bool:
    pat = _coerce_str(args["pattern"])
    if args.get("case_sensitive"):
        return pat in unit_text
    return pat.lower() in unit_text.lower()


SCHEMA = PrimitiveSchema(
    name="contains_string",
    is_structural=False,
    applicable_unit_types=frozenset({"file", "section"}),
    required_args=("pattern",),
    optional_defaults=(("case_sensitive", False),),
    canonicalize=lambda a: (
        ("case_sensitive", bool(a.get("case_sensitive", False))),
        ("pattern", _coerce_str(a["pattern"])),
    ),
    is_universal=_contains_string_universal,
    evaluate_field="text",
)
