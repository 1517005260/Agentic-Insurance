"""Registered score extractors for argmax / aggregate certification.

A WitnessClaim with a ``value_map`` is certifying only when each
``(unit_id, value)`` pair can be re-derived from the cited observation
span by a registered extractor. This makes the LLM's role in argmax
mechanically auditable: it picks units, declares values, and the plant
verifies the values came from the source text.

v1 ships five extractors:

* ``numeric_amount``    — parses currency / decimal numbers via
                          ``babel.numbers``; locale-aware (Chinese
                          full-width comma, EU "1.234,56", Indic, etc.).
* ``percentage``        — parses percentages; ``"82.5%"`` → 82.5
* ``date_iso``          — parses ISO / natural-language dates via
                          ``dateparser`` (zh / de / fr / ja / ...)
                          → POSIX seconds.
* ``integer_count``     — first integer
* ``text_field``        — fetches a structured field from observation payload

Extractors that fail to find a value raise :class:`ScoreExtractionError`,
which the plant turns into a ``score_extraction_failed`` reject for the
ingest call. We deliberately do not fall back to LLM normalisation —
that would re-introduce the value oracle the gate exists to prevent.
"""
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import babel.numbers
import dateparser

from agentic.proof.types import ScoreSpec


_FLOAT_EPSILON = 1e-6


class ScoreExtractionError(ValueError):
    """Raised when an extractor cannot produce a value from a span."""


@dataclass(frozen=True)
class ScoreSchema:
    """Static description of one score extractor."""

    name: str
    output_type: type           # float | int | str
    required_args: Tuple[str, ...]
    optional_defaults: Tuple[Tuple[str, Any], ...]
    canonicalize: Callable[[Dict[str, Any]], Tuple[Tuple[str, Any], ...]]
    extract: Callable[[str, Dict[str, Any], Dict[str, Any]], Any]
    """``extract(span_text, args, observation_payload)`` → value or raises."""


# ----------------------------------------------------------- helpers

# Span-level number regex is just the LOCATOR — we hand the matched
# chunk to ``babel.numbers`` for actual locale-aware parsing. The
# regex tolerates +/- sign, ASCII / Chinese / European / Arabic-Indic
# digit groupings, and a fractional / scientific tail.
_NUMBER_LOCATOR_RE = re.compile(
    r"-?(?:\d{1,3}(?:[,，.。．٬]\d{3})+|\d+)(?:[\.,。．٫]\d+)?(?:[eE][-+]?\d+)?"
)
_PERCENT_RE = re.compile(r"-?\d+(?:[\.,]\d+)?\s*%")


# Locales tried in order when the explicit-locale parse fails. Cover
# the common groupings: en_US ("1,234.56"), de_DE ("1.234,56"),
# zh_CN ("1,234.56" but tolerant of full-width ","), hi_IN (lakhs).
_NUMBER_FALLBACK_LOCALES: Tuple[str, ...] = (
    "en_US", "de_DE", "fr_FR", "zh_CN", "hi_IN",
)


def _parse_number_with_babel(token: str) -> float:
    """Try ``babel.numbers.parse_decimal`` against several locales until
    one succeeds. babel rejects ambiguous strings (``"1,234"`` could be
    1.234 or 1234 depending on locale), so we ALSO fall back to an
    ASCII-stripped float parse for the unambiguous case."""
    # Normalise full-width punctuation to ASCII for babel's lexer.
    normalised = (
        token.replace("，", ",")
             .replace("。", ".")
             .replace("．", ".")
             .replace("٬", ",")
             .replace("٫", ".")
    )
    last_exc: Optional[Exception] = None
    for locale in _NUMBER_FALLBACK_LOCALES:
        try:
            return float(babel.numbers.parse_decimal(normalised, locale=locale, strict=False))
        except (babel.numbers.NumberFormatError, ValueError) as exc:
            last_exc = exc
            continue
    # Final fallback: strip thousands separators and try plain float.
    try:
        return float(normalised.replace(",", ""))
    except ValueError:
        raise ScoreExtractionError(
            f"could not parse {token!r} as number ({last_exc})"
        )


# ----------------------------------------------------------- extractors


def _extract_numeric_amount(span: str, args: Dict[str, Any], payload: Dict[str, Any]) -> float:
    if not span:
        raise ScoreExtractionError("empty span")
    m = _NUMBER_LOCATOR_RE.search(span)
    if m is None:
        raise ScoreExtractionError(f"no number found in {span!r}")
    return _parse_number_with_babel(m.group(0))


def _extract_percentage(span: str, args: Dict[str, Any], payload: Dict[str, Any]) -> float:
    m = _PERCENT_RE.search(span or "")
    if m is None:
        raise ScoreExtractionError(f"no percentage in {span!r}")
    raw = m.group(0).replace("%", "").strip()
    return _parse_number_with_babel(raw)


def _extract_integer_count(span: str, args: Dict[str, Any], payload: Dict[str, Any]) -> int:
    m = re.search(r"-?\d+", span or "")
    if m is None:
        raise ScoreExtractionError(f"no integer in {span!r}")
    return int(m.group(0))


# Settings shared by the locate-then-parse fallback in _extract_date_iso.
_DATEPARSER_SETTINGS = {
    "RETURN_AS_TIMEZONE_AWARE": False,
    "STRICT_PARSING": False,
    # Stable interpretation of ambiguous d/m vs m/d strings.
    "DATE_ORDER": "YMD",
}


def _extract_date_iso(span: str, args: Dict[str, Any], payload: Dict[str, Any]) -> float:
    """Returns POSIX seconds — a numeric ordering for argmax.

    Uses ``dateparser.parse`` which understands ISO, English, Chinese,
    German, French, Japanese, etc. date forms out of the box. Tries the
    whole span first (cheapest); on failure falls back to a substring
    sweep — short candidates first so a date earlier in the prose is
    preferred over a year-only token at the end.
    """
    if not span:
        raise ScoreExtractionError("empty span")
    text = span.strip()
    parsed = dateparser.parse(text, settings=_DATEPARSER_SETTINGS)
    if parsed is not None:
        return parsed.timestamp()
    # Fallback: sweep contiguous chunks. This handles "effective on
    # 2026年5月3日 unless..." where prose surrounds a parseable date.
    for chunk in re.findall(
        r"[\d年月日\-/.,\s]{6,30}[\d]"   # generic numeric run, modest length
        r"|"
        r"\b\w{3,9}\.?\s+\d{1,2},?\s+\d{4}\b",
        text,
    ):
        parsed = dateparser.parse(chunk, settings=_DATEPARSER_SETTINGS)
        if parsed is not None:
            return parsed.timestamp()
    raise ScoreExtractionError(f"no parseable date in {span!r}")


def _extract_text_field(span: str, args: Dict[str, Any], payload: Dict[str, Any]) -> Any:
    """Read a structured field straight from the observation payload.

    For table-cell scores the LLM passes ``field_path``; the plant looks
    it up in the page's ``payload['fields']`` (or wherever the tool
    chose to put it). Span verification is light-touch: we still check
    the value's string form is present in the citation span — this is
    the cheap "did you read the page" check.
    """
    field_path = args.get("field_path")
    if not field_path:
        raise ScoreExtractionError("text_field requires field_path")
    cur: Any = payload
    for part in str(field_path).split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = None
            break
    if cur is None:
        raise ScoreExtractionError(f"text_field path {field_path!r} not found in payload")
    if span and str(cur) not in span:
        raise ScoreExtractionError(
            f"text_field value {cur!r} not present in citation span {span[:80]!r}"
        )
    return cur


# ----------------------------------------------------------- registry


_EXTRACTORS: Dict[str, ScoreSchema] = {
    "numeric_amount": ScoreSchema(
        name="numeric_amount",
        output_type=float,
        required_args=(),
        optional_defaults=(),
        canonicalize=lambda a: (),
        extract=_extract_numeric_amount,
    ),
    "percentage": ScoreSchema(
        name="percentage",
        output_type=float,
        required_args=(),
        optional_defaults=(),
        canonicalize=lambda a: (),
        extract=_extract_percentage,
    ),
    "integer_count": ScoreSchema(
        name="integer_count",
        output_type=int,
        required_args=(),
        optional_defaults=(),
        canonicalize=lambda a: (),
        extract=_extract_integer_count,
    ),
    "date_iso": ScoreSchema(
        name="date_iso",
        output_type=float,
        required_args=(),
        optional_defaults=(),
        canonicalize=lambda a: (),
        extract=_extract_date_iso,
    ),
    "text_field": ScoreSchema(
        name="text_field",
        output_type=str,
        required_args=("field_path",),
        optional_defaults=(),
        canonicalize=lambda a: (("field_path", str(a["field_path"])),),
        extract=_extract_text_field,
    ),
}


# ----------------------------------------------------------- public API


class ScoreError(ValueError):
    """Raised when a ScoreSpec is malformed or references unknown ext."""


def schemas() -> Dict[str, ScoreSchema]:
    return dict(_EXTRACTORS)


def is_orderable(spec: ScoreSpec) -> bool:
    """Whether a score's output type admits the < / > comparisons that
    Γ_argmax needs. Numeric and date-iso scores do; text_field does
    not. Used by the plant to reject argmax + non-orderable score at
    obligation creation rather than letting closure silently fail."""
    schema = _EXTRACTORS.get(spec.name)
    if schema is None:
        return False
    return schema.output_type in (int, float)


def build_spec(name: str, args: Optional[Dict[str, Any]] = None) -> ScoreSpec:
    schema = _EXTRACTORS.get(name)
    if schema is None:
        raise ScoreError(f"Unknown score extractor: {name!r}")
    args = args or {}
    missing = [k for k in schema.required_args if k not in args]
    if missing:
        raise ScoreError(f"{name} missing required arg(s): {missing}")
    canonical = schema.canonicalize(args)
    return ScoreSpec(name=name, args=canonical)


def extract_value(spec: ScoreSpec, span_text: str, observation_payload: Dict[str, Any]) -> Any:
    """Run an extractor against a citation span. Raises
    :class:`ScoreExtractionError` if the value cannot be produced.

    The plant calls this for every ``(unit, value)`` in a candidate
    WitnessClaim with ``value_map`` and rejects the whole claim if any
    extraction disagrees with the LLM-supplied value.
    """
    schema = _EXTRACTORS.get(spec.name)
    if schema is None:
        raise ScoreError(f"Unknown score extractor: {spec.name!r}")
    return schema.extract(span_text, dict(spec.args), observation_payload)


def values_match(extracted: Any, claimed: Any) -> bool:
    """Compare extractor output to LLM-claimed value.

    Floats use a relative tolerance (``_FLOAT_EPSILON``); strings and
    ints fall back to ``==``. Datetime-as-float values use the same
    float tolerance.
    """
    # Accept percentage strings (e.g. "80%") on either side by
    # normalising into a float before comparison; otherwise the LLM
    # claiming "80%" against a percentage extractor that returns 80.0
    # would silently mismatch on type.
    extracted = _coerce_percentage_string(extracted)
    claimed = _coerce_percentage_string(claimed)
    if isinstance(extracted, float) or isinstance(claimed, float):
        try:
            a = float(extracted)
            b = float(claimed)
        except (TypeError, ValueError):
            return False
        if math.isnan(a) or math.isnan(b):
            return False
        return abs(a - b) <= _FLOAT_EPSILON * max(1.0, abs(a), abs(b))
    return extracted == claimed


def _coerce_percentage_string(v: Any) -> Any:
    if isinstance(v, str):
        m = _PERCENT_RE.search(v)
        if m is not None:
            try:
                return _parse_number_with_babel(m.group(0).replace("%", "").strip())
            except ScoreExtractionError:
                return v
    return v
