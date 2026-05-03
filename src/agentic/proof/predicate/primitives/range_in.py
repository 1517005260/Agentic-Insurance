"""``range_in``: closed-interval ``[lo, hi]`` containment on a numeric
payload field. Universal if either bound is non-finite or ``lo > hi``.
"""
from typing import Any, Dict

from agentic.proof.predicate.helpers import _bad_number, _coerce_str, _read_field
from agentic.proof.predicate.registry import PrimitiveSchema


def _range_universal(args: Dict[str, Any]) -> bool:
    if _bad_number(args.get("lo")) or _bad_number(args.get("hi")):
        return True
    return float(args["lo"]) > float(args["hi"])


def _range_eval(payload: Dict[str, Any], args: Dict[str, Any]) -> bool:
    raw = _read_field(payload, args["field_path"])
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return False
    return float(args["lo"]) <= v <= float(args["hi"])


SCHEMA = PrimitiveSchema(
    name="range_in",
    is_structural=False,
    applicable_unit_types=frozenset({"file", "section"}),
    required_args=("field_path", "lo", "hi"),
    optional_defaults=(),
    canonicalize=lambda a: (
        ("field_path", _coerce_str(a["field_path"])),
        ("hi", float(a["hi"])),
        ("lo", float(a["lo"])),
    ),
    is_universal=_range_universal,
    evaluate_field="payload",
)
