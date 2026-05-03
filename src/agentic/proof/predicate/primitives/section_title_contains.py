"""``section_title_contains``: case-insensitive substring on
``payload.title``. Section-only structural primitive — used in the
structural prefix that ``and_split`` extracts.
"""
from typing import Any, Dict

from agentic.proof.predicate.helpers import _coerce_str
from agentic.proof.predicate.registry import PrimitiveSchema


def _section_title_universal(args: Dict[str, Any]) -> bool:
    return _coerce_str(args.get("pattern", "")) == ""


def _section_title_eval(payload: Dict[str, Any], args: Dict[str, Any]) -> bool:
    title = _coerce_str(payload.get("title", "")).lower()
    return _coerce_str(args["pattern"]).lower() in title


SCHEMA = PrimitiveSchema(
    name="section_title_contains",
    is_structural=True,
    applicable_unit_types=frozenset({"section"}),
    required_args=("pattern",),
    optional_defaults=(),
    canonicalize=lambda a: (("pattern", _coerce_str(a["pattern"])),),
    is_universal=_section_title_universal,
    evaluate_field="payload",
)
