"""Small deterministic predicate primitives used by the plant.

Two surfaces:

* ``evaluate_predicate(predicate, text)`` — runs a literal-pattern
  primitive against a unit's text. Used when minting WitnessClaim
  from a read_page observation.
* ``round_trip_value(value, value_type, span)`` — re-extracts a typed
  value from a cited span. Used when minting ValueClaim.

Text normalisation reuses ``ingestion.index.linear_rag.normalize``
(``cleanup`` for OCR/LaTeX/HTML residue, ``canonical_form`` for
NFKC + case folding + Traditional→Simplified) — same machinery the
entity index uses, so corpus-level handling is consistent.

The numeric / percentage / date tokenisers + ``coerce_number``
helper are exposed publicly so the finalize gate can reuse them
when comparing certified values against an LLM-supplied draft.
"""

import re
from datetime import date, datetime
from typing import Any, Callable, Optional

import regex as _regex

from ingestion.index.linear_rag.normalize import (
    canonical_form as _canonical_form,
    cleanup as _cleanup_text,
)


# ---------------------------------------------------------------- text normalisation


def _prepare_text(text: str, *, case_sensitive: bool) -> str:
    """Strip OCR artefacts and (optionally) fold case for matching."""

    cleaned = _cleanup_text(text or "")
    if case_sensitive:
        # canonical_form lowercases unconditionally; emulate the
        # case-sensitive path with a bare NFKC normalisation.
        import unicodedata
        return unicodedata.normalize("NFKC", cleaned)
    return _canonical_form(cleaned)


# ---------------------------------------------------------------- predicate primitives


def _eval_contains_string(args: dict, text: str) -> bool:
    pattern = str(args.get("pattern", ""))
    if not pattern:
        return False
    case_sensitive = bool(args.get("case_sensitive"))
    needle = _prepare_text(pattern, case_sensitive=case_sensitive)
    haystack = _prepare_text(text, case_sensitive=case_sensitive)
    return needle in haystack


def _eval_regex_match(args: dict, text: str) -> bool:
    pattern = str(args.get("pattern", ""))
    if not pattern:
        return False
    flags_str = str(args.get("flags", "i"))
    case_sensitive = "i" not in flags_str
    haystack = _prepare_text(text, case_sensitive=case_sensitive)
    flags = _regex_flags(flags_str)
    try:
        return _regex.search(pattern, haystack, flags) is not None
    except _regex.error:
        return False


def _regex_flags(flags: str) -> int:
    out = 0
    if "i" in flags:
        out |= _regex.IGNORECASE
    if "m" in flags:
        out |= _regex.MULTILINE
    if "s" in flags:
        out |= _regex.DOTALL
    return out


_PREDICATE_EVALUATORS: dict[str, Callable[[dict, str], bool]] = {
    "contains_string": _eval_contains_string,
    "regex_match": _eval_regex_match,
}


def has_evaluator(predicate_name: str) -> bool:
    return predicate_name in _PREDICATE_EVALUATORS


def evaluate_predicate(predicate, text: str) -> bool:
    name = predicate.name
    evaluator = _PREDICATE_EVALUATORS.get(name)
    if evaluator is None:
        raise KeyError(name)
    return evaluator(predicate.args_dict(), text)


def predicate_canonical_id_for_pattern_search(pattern: str) -> str:
    from agentic.closure.obligation import PredicateRef
    return PredicateRef.build("regex_match", {"pattern": pattern, "flags": "i"}).canonical_id


# ---------------------------------------------------------------- value tokenisers (public)
#
# Public so the finalize gate can reuse the same regex+coerce that
# ValueClaim round-trip uses; otherwise the two surfaces drift.


NUMERIC_TOKEN = re.compile(
    r"-?(?:\d{1,3}(?:[, ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
)
PERCENT_TOKEN = re.compile(r"-?\d+(?:\.\d+)?\s*%")
DATE_TOKEN = re.compile(r"\b(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?\b")


def coerce_number(token: str) -> Optional[float]:
    cleaned = token.replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalise_numeric_for_set(token: str) -> Optional[str]:
    """Canonical form for set-style equality across "8659" / "8,659"."""

    n = coerce_number(token)
    if n is None:
        return None
    return str(int(n)) if n.is_integer() else repr(n)


def prepare_for_value_extraction(span: str) -> str:
    """Strip artefacts before running NUMERIC/PERCENT/DATE tokenisers."""

    import unicodedata
    return unicodedata.normalize("NFKC", _cleanup_text(span or ""))


# ---------------------------------------------------------------- typed quantity audit
#
# A tighter shape for the finalize gate's draft check: a quantity is
# (kind, value) where kind ∈ {"percent", "numeric"}. This prevents
# "27%" from being treated as backed by an unrelated bare "27" in some
# claim's span — the kinds differ.


def extract_quantities(text: str) -> list[tuple[str, str]]:
    """Return ALL ``(kind, canonical)`` quantities found in ``text``.

    Caller decides whether to filter for load-bearing candidates via
    ``is_load_bearing_quantity`` — a closed_value's "2" must match a
    draft "2", but a descriptive "p15" does not need to be backed.
    """

    cleaned = prepare_for_value_extraction(text)
    out: list[tuple[str, str]] = []
    consumed: list[tuple[int, int]] = []
    for m in PERCENT_TOKEN.finditer(cleaned):
        digits = m.group(0).rstrip("%").strip()
        canonical = normalise_numeric_for_set(digits)
        if canonical is not None:
            out.append(("percent", canonical))
            consumed.append((m.start(), m.end()))
    for m in NUMERIC_TOKEN.finditer(cleaned):
        if any(start <= m.start() < end for start, end in consumed):
            continue
        canonical = normalise_numeric_for_set(m.group(0))
        if canonical is not None:
            out.append(("numeric", canonical))
    return out


def is_load_bearing_quantity(kind: str, canonical: str) -> bool:
    """Percent quantities are always load-bearing. Raw numerics count
    only when they carry a thousands/decimal separator or a magnitude
    >=100; small bare integers (page numbers, list ranks) are not.
    """

    if kind == "percent":
        return True
    if "," in canonical or "." in canonical or " " in canonical:
        return True
    n = coerce_number(canonical)
    return n is not None and abs(n) >= 100


# ---------------------------------------------------------------- value round-trip


def round_trip_value(value: Any, value_type: str, span: str) -> bool:
    span_clean = prepare_for_value_extraction(span)

    if value_type == "numeric":
        try:
            target = float(value)
        except (TypeError, ValueError):
            return False
        for m in NUMERIC_TOKEN.finditer(span_clean):
            n = coerce_number(m.group(0))
            if n is not None and n == target:
                return True
        return False

    if value_type == "integer_count":
        try:
            target = int(value)
        except (TypeError, ValueError):
            return False
        if target < 0:
            return False
        for m in NUMERIC_TOKEN.finditer(span_clean):
            n = coerce_number(m.group(0))
            if n is not None and n == float(target):
                return True
        return False

    if value_type == "percentage":
        if not isinstance(value, str):
            return False
        normalised = value.replace(" ", "").lower()
        for m in PERCENT_TOKEN.finditer(span_clean):
            if m.group(0).replace(" ", "").lower() == normalised:
                return True
        return False

    if value_type == "date_iso":
        if not isinstance(value, str):
            return False
        target = _parse_date(value)
        if target is None:
            return False
        for m in DATE_TOKEN.finditer(span_clean):
            try:
                year = int(m.group(1))
                month = int(m.group(2))
                day = int(m.group(3) or 1)
                if date(year, month, day) == target:
                    return True
            except ValueError:
                continue
        return False

    if value_type == "text":
        if not isinstance(value, str) or not value.strip():
            return False
        target = _canonical_form(value.strip())
        haystack = _canonical_form(span_clean)
        return target in haystack

    return False


def _parse_date(s: str) -> Optional[date]:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None
