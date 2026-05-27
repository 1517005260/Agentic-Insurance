"""Plant — derives claims from trusted observations and writes CLOSED status.

Validation is structural plus *origin*: a claim is admitted only when
the source observation is one of the contract's accepted source
observation types for the obligation kind. ScanClaim is copied
verbatim from PatternScanObservation; WitnessClaim binds either to a
scan classification or to a *ReadObservation whose unit_type matches
the obligation; ValueClaim round-trips against a cited span inside a
*ReadObservation.

``run_closure`` is the only writer of ``CLOSED`` status.
"""

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from agentic.closure.candidate_gap import (
    CandidateGap,
    equivalence_update as _equivalence_update,
    promote_candidate_gap as _promote_candidate_gap,
)
from agentic.closure.budget import Budget
from agentic.closure.claims import (
    Citation,
    Claim,
    DerivedProvenance,
    Polarity,
    ScanClaim,
    ScanProvenance,
    ValueClaim,
    WitnessClaim,
)
from agentic.closure.contract import COMPUTE_OPERATIONS
from agentic.closure.closures import Closed, Open, try_close
from agentic.closure.inventory import Inventory
from agentic.closure.obligation import (
    Obligation,
    PredicateRef,
    ScopeRef,
    UnitType,
)
from agentic.closure.predicates import (
    evaluate_predicate,
    has_evaluator,
    predicate_canonical_id_for_pattern_search,
    round_trip_value,
)


_READ_OBSERVATION_TYPES = {
    "PageReadObservation": "page",
    "PassageReadObservation": "passage",
    "TableRowReadObservation": "table_row",
}


@dataclass(frozen=True)
class ErrorEnvelope:
    code: str
    remediation: str
    context: dict


@runtime_checkable
class ObservationStore(Protocol):
    def get_text(self, observation_id: str) -> Optional[str]: ...
    def get_tool_name(self, observation_id: str) -> Optional[str]: ...


class KernelInvariantError(RuntimeError):
    pass


def _envelope(code: str, remediation: str, **context: Any) -> ErrorEnvelope:
    return ErrorEnvelope(code=code, remediation=remediation, context=context)


# ---------------------------------------------------------------- Plant


class Plant:
    def __init__(
        self,
        inventory: Inventory,
        observations: "ObservationStore | Mapping[str, str]",
    ) -> None:
        self._inventory = inventory
        self._observations: ObservationStore = (
            _MappingObservations(observations) if isinstance(observations, Mapping) else observations
        )
        self._minted_ids = 0

    def mint_id(self, prefix: str = "claim") -> str:
        self._minted_ids += 1
        return f"{prefix}_{self._minted_ids:06d}"

    # ---------------------------------------------------------- ingest

    def ingest_scan_claim(
        self,
        observation_id: str,
        *,
        expected_unit_type: Optional[UnitType] = None,
    ) -> ScanClaim | ErrorEnvelope:
        parsed = self._load_observation(observation_id, allowed_tools=frozenset({"pattern_search"}))
        if isinstance(parsed, ErrorEnvelope):
            return parsed
        if parsed.get("observation_type") != "PatternScanObservation":
            return _envelope(
                "scan_requires_pattern_search",
                "ScanClaim must be derived from a pattern_search observation.",
                observation_type=parsed.get("observation_type"),
            )
        if parsed.get("compact"):
            return _envelope(
                "scan_observation_compact",
                "PatternScanObservation was emitted with compact=True; it omits "
                "scanned_units / negative_units that the proof kernel needs to "
                "verify scope coverage. Re-run pattern_search with compact=False "
                "(the default) for any pattern_search call you intend to ingest "
                "as a ScanClaim.",
            )
        unit_type = parsed.get("unit_type")
        if unit_type not in {"page", "passage", "table_row"}:
            return _envelope("invalid_observation", "PatternScanObservation has no recognised unit_type.")
        if expected_unit_type is not None and unit_type != expected_unit_type:
            return _envelope(
                "unit_type_mismatch",
                f"observation unit_type={unit_type!r} differs from obligation unit_type={expected_unit_type!r}.",
            )
        scope_dict = parsed.get("scope") or {}
        scope = ScopeRef.build(
            file_ids=tuple(scope_dict.get("file_ids") or ()),
            section_ids=tuple(scope_dict.get("section_ids") or ()) or None,
        )
        pattern = parsed.get("pattern")
        if not pattern:
            return _envelope("invalid_observation", "PatternScanObservation has no pattern.")
        predicate = PredicateRef.build("regex_match", {"pattern": pattern, "flags": "i"})
        scanned = frozenset(parsed.get("scanned_units") or ())
        positive = frozenset(parsed.get("positive_units") or ())
        negative = frozenset(parsed.get("negative_units") or ())
        domain = self._inventory.units(scope, unit_type)
        if scanned != domain:
            return _envelope(
                "scan_coverage_mismatch",
                "scanned_units must equal Inventory.units(scope, unit_type). "
                "Re-run pattern_search with a scope that covers the obligation's full domain.",
                missing_from_scan=sorted(domain - scanned)[:32],
                extra_in_scan=sorted(scanned - domain)[:32],
            )
        try:
            return ScanClaim(
                id="",
                scope=scope,
                unit_type=unit_type,  # type: ignore[arg-type]
                predicate=predicate,
                scanned_units=scanned,
                positive_units=positive,
                negative_units=negative,
                exhaustive=True,
                provenance=ScanProvenance(observation_id=observation_id),
            )
        except ValueError as exc:
            return _envelope("invalid_scan", str(exc))

    def ingest_witness_claim(
        self,
        *,
        observation_id: str,
        unit_id: str,
        polarity: Polarity,
        predicate: PredicateRef,
        expected_unit_type: UnitType,
        span: Optional[str] = None,
        span_start: Optional[int] = None,
        span_end: Optional[int] = None,
    ) -> WitnessClaim | ErrorEnvelope:
        parsed = self._load_observation(
            observation_id,
            allowed_tools=frozenset({"pattern_search", "read"}),
        )
        if isinstance(parsed, ErrorEnvelope):
            return parsed
        observation_type = parsed.get("observation_type")
        if observation_type == "PatternScanObservation":
            return self._witness_from_scan(
                parsed=parsed,
                observation_id=observation_id,
                unit_id=unit_id,
                polarity=polarity,
                predicate=predicate,
                expected_unit_type=expected_unit_type,
            )
        if observation_type in _READ_OBSERVATION_TYPES:
            return self._witness_from_read(
                parsed=parsed,
                observation_id=observation_id,
                unit_id=unit_id,
                polarity=polarity,
                predicate=predicate,
                expected_unit_type=expected_unit_type,
                span=span,
                span_start=span_start,
                span_end=span_end,
            )
        return _envelope(
            "witness_requires_known_observation",
            "WitnessClaim must cite a pattern_search OR a read observation matching the obligation's unit_type.",
            observation_type=observation_type,
        )

    def ingest_value_claim(
        self,
        *,
        observation_id: str,
        unit_id: str,
        field: str,
        value: Any,
        value_type: str,
        span: str,
        expected_unit_type: UnitType,
        span_start: Optional[int] = None,
        span_end: Optional[int] = None,
    ) -> ValueClaim | ErrorEnvelope:
        parsed = self._load_observation(observation_id, allowed_tools=frozenset({"read"}))
        if isinstance(parsed, ErrorEnvelope):
            return parsed
        observation_type = parsed.get("observation_type")
        if observation_type not in _READ_OBSERVATION_TYPES:
            return _envelope(
                "value_requires_read",
                "ValueClaim must cite a read observation.",
                observation_type=observation_type,
            )
        obs_unit_type = _READ_OBSERVATION_TYPES[observation_type]
        if obs_unit_type != expected_unit_type:
            return _envelope(
                "unit_type_mismatch",
                f"observation unit_type={obs_unit_type!r} differs from obligation unit_type={expected_unit_type!r}; "
                "read at the obligation's granularity.",
            )
        unit = _find_unit(parsed, unit_id)
        if unit is None:
            return _envelope(
                "unit_not_in_observation",
                "Cited unit_id was not present in the read observation.",
                unit_id=unit_id,
            )
        unit_text = unit.get("text") or ""
        if not span or span not in unit_text:
            return _envelope(
                "citation_not_verbatim",
                "Cited span must appear verbatim in the unit's text.",
                unit_id=unit_id,
            )
        if not round_trip_value(value, value_type, span):
            return _envelope(
                "value_round_trip_failed",
                f"value={value!r} does not round-trip via value_type={value_type!r}.",
                value=value,
                value_type=value_type,
                span_preview=span[:80],
            )
        try:
            citation = _build_citation(
                unit=unit,
                unit_id=unit_id,
                span=span,
                observation_id=observation_id,
                span_start=span_start,
                span_end=span_end,
            )
        except ValueError as exc:
            return _envelope("invalid_citation", str(exc))
        return ValueClaim(
            id="",
            unit_id=unit_id,
            field=field,
            value=value,
            value_type=value_type,  # type: ignore[arg-type]
            citation=citation,
        )

    def ingest_derived_value_claim(
        self,
        *,
        field: str,
        operation: str,
        input_claim_ids: tuple[str, ...],
        value: Any,
        value_type: str,
        all_claims: Sequence[Claim],
    ) -> ValueClaim | ErrorEnvelope:
        """Build a derived ValueClaim by re-running ``operation`` over
        the cited input ValueClaims. The kernel evaluates the math
        itself (PCN-style claim-bound numeric verification, but the
        verifier is the kernel, not the renderer; the LLM never writes
        the arithmetic — it only nominates the operation + inputs).

        Failure modes:
        * unknown_operation — not in the contract whitelist
        * input_not_found / input_not_value_claim
        * input_type_incompatible — value_type doesn't match op signature
        * arithmetic_mismatch — kernel-computed value ≠ claimed value
        """

        op = COMPUTE_OPERATIONS.get(operation)
        if op is None:
            return _envelope(
                "unknown_operation",
                f"operation must be one of {sorted(COMPUTE_OPERATIONS)}.",
                operation=operation,
            )
        if not input_claim_ids:
            return _envelope("missing_inputs", "at least one input_claim_id required.")
        if op.arity == "binary" and len(input_claim_ids) != 2:
            return _envelope(
                "wrong_arity",
                f"operation {operation!r} is binary; got {len(input_claim_ids)} inputs.",
            )

        index = {c.id: c for c in all_claims}
        inputs: list[ValueClaim] = []
        for cid in input_claim_ids:
            claim = index.get(cid)
            if claim is None:
                return _envelope("input_not_found", "an input_claim_id does not resolve.", claim_id=cid)
            if not isinstance(claim, ValueClaim):
                return _envelope("input_not_value_claim", "derivation inputs must be ValueClaims.", claim_id=cid)
            inputs.append(claim)

        try:
            computed = _run_operation(operation, inputs)
        except _ArithmeticError as exc:
            return _envelope(str(exc.code), exc.detail, **exc.context)

        if not _values_equal(computed, value):
            return _envelope(
                "arithmetic_mismatch",
                f"kernel re-ran {operation!r} → {computed!r}, claimed value was {value!r}.",
                operation=operation,
                computed=computed,
                claimed=value,
            )

        try:
            return ValueClaim(
                id="",
                unit_id=None,
                field=field,
                value=value,
                value_type=value_type,  # type: ignore[arg-type]
                citation=None,
                derived=DerivedProvenance(
                    operation=operation,  # type: ignore[arg-type]
                    input_claim_ids=tuple(input_claim_ids),
                ),
            )
        except ValueError as exc:
            return _envelope("invalid_derived_value", str(exc))

    # ---------------------------------------------------------- obligation update

    def validate_obligation_update(
        self,
        gap: CandidateGap,
        obligations: list[Obligation],
        budget: Budget,
        *,
        promoted_so_far: int = 0,
    ) -> Optional[Obligation]:
        return _promote_candidate_gap(
            gap, obligations, self._inventory, budget,
            promoted_so_far=promoted_so_far,
        )

    def equivalence_update(self, old: Obligation, new: Obligation) -> bool:
        return _equivalence_update(old, new, self._inventory)

    # ---------------------------------------------------------- closure

    def run_closure(
        self,
        obligations: Sequence[Obligation],
        claims: Sequence[Claim],
    ) -> None:
        # Closure is *stateless*: we recompute every obligation's status
        # against the current claim set on each call. CLOSED is never
        # cached past contradiction — a second disagreeing claim flips a
        # lookup back to OPEN with ``ambiguous_lookup``.
        for o in obligations:
            result = try_close(o, claims, self._inventory)
            if isinstance(result, Closed):
                if result.value is None or not result.by:
                    raise KernelInvariantError(
                        f"closure rule for {o.kind!r} returned Closed without value/by"
                    )
                o.status = "CLOSED"
                o.closed_value = result.value
                o.closed_by = list(result.by)
                o.failure_kind = None
                o.diagnostic_data = result.diagnostic_data
            else:
                assert isinstance(result, Open)
                o.status = "OPEN"
                o.closed_value = None
                o.closed_by = []
                o.failure_kind = result.reason
                o.diagnostic_data = result.diagnostic_data

    # ---------------------------------------------------------- helpers

    def _load_observation(
        self,
        observation_id: str,
        *,
        allowed_tools: frozenset[str],
    ) -> dict | ErrorEnvelope:
        tool_name = self._observations.get_tool_name(observation_id)
        if tool_name is None:
            return _envelope("unknown_observation", "observation_id does not resolve.", observation_id=observation_id)
        if tool_name not in allowed_tools:
            return _envelope(
                "untrusted_observation_tool",
                "This claim type can only be derived from a trusted acquisition tool.",
                observation_id=observation_id,
                tool_name=tool_name,
                allowed_tools=sorted(allowed_tools),
            )
        text = self._observations.get_text(observation_id)
        if text is None:
            return _envelope("unknown_observation", "observation_id does not resolve.", observation_id=observation_id)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return _envelope("observation_not_structured", "Observation text is not JSON.", observation_id=observation_id)
        if not isinstance(parsed, dict):
            return _envelope("observation_not_structured", "Observation JSON is not an object.", observation_id=observation_id)
        return parsed

    def _witness_from_scan(
        self,
        *,
        parsed: dict,
        observation_id: str,
        unit_id: str,
        polarity: Polarity,
        predicate: PredicateRef,
        expected_unit_type: UnitType,
    ) -> WitnessClaim | ErrorEnvelope:
        if parsed.get("unit_type") != expected_unit_type:
            return _envelope(
                "unit_type_mismatch",
                f"observation unit_type={parsed.get('unit_type')!r} differs from obligation {expected_unit_type!r}.",
            )
        if parsed.get("compact"):
            return _envelope(
                "scan_observation_compact",
                "PatternScanObservation was emitted with compact=True; cannot "
                "derive a WitnessClaim from it (scanned_units / negative_units "
                "are needed to verify classification). Re-run pattern_search "
                "with compact=False, or cite a `read` observation for the "
                "WitnessClaim instead.",
            )
        scan_pattern = parsed.get("pattern", "")
        if predicate.canonical_id != predicate_canonical_id_for_pattern_search(scan_pattern):
            return _envelope(
                "predicate_mismatch_with_scan",
                "Witness predicate must match the pattern_search regex.",
                obligation_predicate=predicate.canonical_id,
                scan_pattern=scan_pattern,
            )
        scanned = set(parsed.get("scanned_units") or ())
        if unit_id not in scanned:
            return _envelope("unit_not_in_scan", "Cited unit was not part of the scan.", unit_id=unit_id)
        actual = (
            "positive" if unit_id in set(parsed.get("positive_units") or ())
            else "negative" if unit_id in set(parsed.get("negative_units") or ())
            else None
        )
        if actual is None:
            return _envelope("scan_classification_missing", "Unit appears in scanned_units but in neither set.", unit_id=unit_id)
        if actual != polarity:
            return _envelope(
                "polarity_disagrees_with_scan",
                f"Scan classified unit as {actual!r}; cannot mint a {polarity!r} witness.",
                unit_id=unit_id,
            )
        return WitnessClaim(
            id="",
            unit_id=unit_id,
            predicate=predicate,
            polarity=polarity,
            citation=Citation(
                file_id=_file_id_from_unit(unit_id) or "",
                page_id=_page_id_from_unit(unit_id) or "",
                unit_id=unit_id,
                span=scan_pattern,
                observation_id=observation_id,
            ),
        )

    def _witness_from_read(
        self,
        *,
        parsed: dict,
        observation_id: str,
        unit_id: str,
        polarity: Polarity,
        predicate: PredicateRef,
        expected_unit_type: UnitType,
        span: Optional[str],
        span_start: Optional[int],
        span_end: Optional[int],
    ) -> WitnessClaim | ErrorEnvelope:
        obs_unit_type = _READ_OBSERVATION_TYPES[parsed["observation_type"]]
        if obs_unit_type != expected_unit_type:
            return _envelope(
                "unit_type_mismatch",
                f"observation unit_type={obs_unit_type!r} differs from obligation {expected_unit_type!r}.",
            )
        unit = _find_unit(parsed, unit_id)
        if unit is None:
            return _envelope(
                "unit_not_in_observation",
                "Cited unit_id was not present in the read observation.",
                unit_id=unit_id,
            )
        unit_text = unit.get("text") or ""
        if not has_evaluator(predicate.name):
            return _envelope(
                "predicate_unsupported",
                f"Predicate {predicate.name!r} is not in the kernel's primitive registry.",
            )
        try:
            matches = evaluate_predicate(predicate, unit_text)
        except KeyError:
            return _envelope("predicate_unsupported", f"Unknown predicate {predicate.name!r}.")
        expected_match = (polarity == "positive")
        if matches != expected_match:
            return _envelope(
                "polarity_disagrees_with_text",
                f"Predicate primitive returned match={matches}; cannot mint a {polarity!r} witness on this unit.",
                unit_id=unit_id,
            )
        if span is None:
            return _envelope("span_required", "WitnessClaim from a read observation must carry a verbatim span.")
        if span not in unit_text:
            return _envelope("citation_not_verbatim", "Cited span must appear verbatim in the unit's text.", unit_id=unit_id)
        try:
            citation = _build_citation(
                unit=unit,
                unit_id=unit_id,
                span=span,
                observation_id=observation_id,
                span_start=span_start,
                span_end=span_end,
            )
        except ValueError as exc:
            return _envelope("invalid_citation", str(exc))
        return WitnessClaim(
            id="",
            unit_id=unit_id,
            predicate=predicate,
            polarity=polarity,
            citation=citation,
        )


# ---------------------------------------------------------------- helpers


def _find_unit(parsed: dict, unit_id: str) -> Optional[dict]:
    for u in parsed.get("units") or []:
        if isinstance(u, dict) and u.get("unit_id") == unit_id:
            return u
    return None


def _build_citation(
    *,
    unit: dict,
    unit_id: str,
    span: str,
    observation_id: str,
    span_start: Optional[int],
    span_end: Optional[int],
) -> Citation:
    file_id = unit.get("file_id") or _file_id_from_unit(unit_id) or ""
    page_id = unit.get("page_id") or _page_id_from_unit(unit_id) or ""
    return Citation(
        file_id=file_id,
        page_id=page_id,
        unit_id=unit_id,
        span=span,
        observation_id=observation_id,
        span_start=span_start,
        span_end=span_end,
    )


def _file_id_from_unit(unit_id: str) -> Optional[str]:
    if "/" in unit_id:
        return unit_id.split("/", 1)[0]
    if ":" in unit_id:
        return unit_id.split(":", 1)[0]
    return None


def _page_id_from_unit(unit_id: str) -> Optional[str]:
    if "/" in unit_id:
        rest = unit_id.split("/", 1)[1]
        return rest.split(":", 1)[0] if ":" in rest else rest
    return unit_id


class _ArithmeticError(Exception):
    def __init__(self, code: str, detail: str, **context: Any) -> None:
        self.code = code
        self.detail = detail
        self.context = context


def _coerce_numeric(claim: ValueClaim) -> float:
    """Project a ValueClaim's value to a float for arithmetic."""

    vt = claim.value_type
    val = claim.value
    if vt == "numeric":
        return float(val)
    if vt == "integer_count":
        return float(int(val))
    if vt == "percentage" and isinstance(val, str):
        cleaned = val.replace(" ", "").rstrip("%")
        return float(cleaned)
    if vt == "percentage" and isinstance(val, (int, float)):
        return float(val)
    raise _ArithmeticError(
        "input_type_incompatible",
        f"cannot project value_type={vt!r} value={val!r} to numeric.",
        claim_id=claim.id, value_type=vt,
    )


def _run_operation(operation: str, inputs: list[ValueClaim]) -> float:
    """Kernel-side arithmetic. White-listed ops only. PCN/PoT note:
    the LLM nominates inputs + operation; the kernel does the math.
    No code_run output trusted — this is in-kernel verification."""

    if operation == "sum":
        return sum(_coerce_numeric(c) for c in inputs)
    if operation == "product":
        result = 1.0
        for c in inputs:
            result *= _coerce_numeric(c)
        return result
    if operation == "max":
        return max(_coerce_numeric(c) for c in inputs)
    if operation == "min":
        return min(_coerce_numeric(c) for c in inputs)
    a, b = _coerce_numeric(inputs[0]), _coerce_numeric(inputs[1])
    if operation == "difference":
        return a - b
    if operation == "quotient":
        if b == 0:
            raise _ArithmeticError("divide_by_zero", "quotient input[1] is 0.")
        return a / b
    if operation == "percent_of":
        # input[1] interpreted as percentage; if its value_type='percentage'
        # _coerce_numeric already returns the % literal (e.g. 27.0); divide
        # by 100 here. If it's a bare numeric, treat the same.
        return a * b / 100.0
    raise _ArithmeticError("unknown_operation", f"operation {operation!r} not implemented.")


def _values_equal(computed: float, claimed: Any) -> bool:
    """PCN policy: exact-equality tolerance with float epsilon."""

    if isinstance(claimed, str):
        cleaned = claimed.replace(",", "").replace(" ", "").rstrip("%")
        try:
            claimed_f = float(cleaned)
        except ValueError:
            return False
    else:
        try:
            claimed_f = float(claimed)
        except (TypeError, ValueError):
            return False
    return abs(computed - claimed_f) < 1e-6


class _MappingObservations:
    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, str]) -> None:
        self._data = data

    def get_text(self, observation_id: str) -> Optional[str]:
        return self._data.get(observation_id)

    def get_tool_name(self, observation_id: str) -> Optional[str]:
        text = self._data.get(observation_id)
        if text is None:
            return None
        try:
            obs = json.loads(text)
        except json.JSONDecodeError:
            return "__mapping__"
        otype = obs.get("observation_type") if isinstance(obs, dict) else None
        if otype == "PatternScanObservation":
            return "pattern_search"
        if otype in _READ_OBSERVATION_TYPES:
            return "read"
        return "__mapping__"
