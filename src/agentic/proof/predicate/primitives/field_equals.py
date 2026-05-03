"""``field_equals``: dotted-path payload read compared by ``==``.
Universal iff ``field_path`` is empty/whitespace.
"""
from typing import Any, Dict

from agentic.proof.predicate.helpers import _coerce_str, _read_field
from agentic.proof.predicate.registry import PrimitiveSchema


def _field_equals_universal(args: Dict[str, Any]) -> bool:
    return not _coerce_str(args.get("field_path", "")).strip()


def _field_equals_eval(payload: Dict[str, Any], args: Dict[str, Any]) -> bool:
    return _read_field(payload, args["field_path"]) == args.get("value")


SCHEMA = PrimitiveSchema(
    name="field_equals",
    is_structural=False,
    applicable_unit_types=frozenset({"file", "section"}),
    required_args=("field_path", "value"),
    optional_defaults=(),
    canonicalize=lambda a: (
        ("field_path", _coerce_str(a["field_path"])),
        ("value", a["value"]),
    ),
    is_universal=_field_equals_universal,
    evaluate_field="payload",
)
