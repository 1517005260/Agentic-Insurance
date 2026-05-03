"""``type_is``: compares the unit's ``unit_type`` to a constant. The
plant injects ``__obligation_unit_type`` into args so the universal
probe can flag ``type_is(section)`` on a ``section`` obligation as
trivially true.
"""
from typing import Any, Dict

from agentic.proof.predicate.helpers import _coerce_str
from agentic.proof.predicate.registry import PrimitiveSchema


def _type_is_universal(args: Dict[str, Any]) -> bool:
    """Universal iff the predicate compares the unit_type against itself."""
    target = _coerce_str(args.get("unit_type", ""))
    own = _coerce_str(args.get("__obligation_unit_type", ""))
    return bool(target) and target == own


def _type_is_eval(payload: Dict[str, Any], args: Dict[str, Any]) -> bool:
    return _coerce_str(payload.get("unit_type", "")) == _coerce_str(args["unit_type"])


SCHEMA = PrimitiveSchema(
    name="type_is",
    is_structural=True,
    applicable_unit_types=frozenset({"file", "section"}),
    required_args=("unit_type",),
    optional_defaults=(),
    canonicalize=lambda a: (("unit_type", _coerce_str(a["unit_type"])),),
    is_universal=_type_is_universal,
    evaluate_field="payload",
)
