"""``regex_match``: Python ``re.search`` against ``unit.text``. The
``flags`` arg is a flag-string (``"i"`` for IGNORECASE, ``"s"`` for
DOTALL); only those two are honoured. Universal if pattern matches the
empty string under the conservative blacklist.
"""
import re
from typing import Any, Dict

from agentic.proof.predicate.helpers import (
    UNIVERSAL_REGEX_BLACKLIST,
    _coerce_str,
    _matches_empty_string,
)
from agentic.proof.predicate.registry import PrimitiveSchema


def _regex_universal(args: Dict[str, Any]) -> bool:
    pat = _coerce_str(args.get("pattern", ""))
    if pat in UNIVERSAL_REGEX_BLACKLIST:
        return True
    return _matches_empty_string(pat)


def _regex_eval(unit_text: str, args: Dict[str, Any]) -> bool:
    flags = 0
    if "i" in _coerce_str(args.get("flags", "")):
        flags |= re.IGNORECASE
    if "s" in _coerce_str(args.get("flags", "")):
        flags |= re.DOTALL
    return re.search(_coerce_str(args["pattern"]), unit_text, flags) is not None


SCHEMA = PrimitiveSchema(
    name="regex_match",
    is_structural=False,
    applicable_unit_types=frozenset({"file", "section"}),
    required_args=("pattern",),
    optional_defaults=(("flags", ""),),
    canonicalize=lambda a: (
        ("flags", _coerce_str(a.get("flags", ""))),
        ("pattern", _coerce_str(a["pattern"])),
    ),
    is_universal=_regex_universal,
    evaluate_field="text",
)
