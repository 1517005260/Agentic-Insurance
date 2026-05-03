"""Shared helpers for predicate primitives.

Pure utility functions used by primitives' ``is_universal`` blacklists,
``canonicalize`` callbacks and ``evaluate`` paths. Kept free of any
imports from the primitives or registry modules so the dependency
graph stays a tree:

    primitives/* -> helpers
    registry, algebra, evaluation -> primitives, helpers
"""
import math
import re
from datetime import datetime
from typing import Any, Dict, FrozenSet, Optional, Tuple


# Regex-syntax sentinels we treat as "matches everything". This is a
# conservative deny-list — full universality is undecidable for Python
# regex with backreferences, so we reject what we can recognise and
# accept everything else as well-formed.
UNIVERSAL_REGEX_BLACKLIST: FrozenSet[str] = frozenset(
    {".*", "(?:.|\n)*", "(?s).*", ".+", "^", "$", "^.*$", "(.*?)"}
)


def _coerce_str(v: Any) -> str:
    return v if isinstance(v, str) else str(v)


def _matches_empty_string(pattern: str) -> bool:
    """Cheap probe: does the regex match the empty string? Catches
    ``""``, ``".*"``, ``"a*"`` and the like without trying to do real
    universality reasoning. False negatives are safe (we just register a
    pattern that may fail closure later); false positives would be a
    bug that wrongly rejects legal user predicates.
    """
    if not pattern:
        return True
    try:
        return re.match(pattern, "") is not None
    except re.error:
        return False


def _bad_number(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return True
    return math.isnan(f) or math.isinf(f)


def _canon_str_args(args: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((k, args[k]) for k in args))


def _read_field(payload: Dict[str, Any], path: str) -> Any:
    """Dotted-path read into a payload dict. Returns ``None`` on miss."""
    cur: Any = payload
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _parse_iso_date(s: str) -> Optional[float]:
    """Accept ``YYYY-MM-DD``, ``YYYY-MM-DDTHH:MM:SS`` or anything
    Python's ``datetime.fromisoformat`` will take. Returns the POSIX
    timestamp for ordering, or ``None`` on parse failure.
    """
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d.timestamp()
    except (ValueError, TypeError):
        return None


# Six comparison operators shared by numeric_compare and date_compare.
NUMERIC_OPS: Dict[str, Any] = {
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}
