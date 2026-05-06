"""Validated evidence claims.

Three shapes only — Witness / Scan / Value. Domain closure is encoded
inside ``ScanClaim`` (``exhaustive`` flag plus ``scanned_units ==
inventory.units``) rather than in a separate seal claim.

Claims that exit the plant are fully validated; the closure rules
trust their inputs and never re-parse cited spans.
"""
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from agentic.closure.obligation import PredicateRef, ScopeRef, UnitType


Polarity = Literal["positive", "negative"]


@dataclass(frozen=True)
class Citation:
    file_id: str
    page_id: str
    unit_id: str
    span: str
    observation_id: str
    span_start: Optional[int] = None
    span_end: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.observation_id or not self.observation_id.strip():
            raise ValueError("Citation.observation_id is mandatory.")
        if not self.span or not self.span.strip():
            raise ValueError("Citation.span must be a non-empty verbatim string.")
        if (self.span_start is None) ^ (self.span_end is None):
            raise ValueError("span_start/span_end must be provided together or both omitted.")
        if (
            self.span_start is not None
            and self.span_end is not None
            and self.span_end <= self.span_start
        ):
            raise ValueError("span_end must be strictly greater than span_start.")


@dataclass(frozen=True)
class WitnessClaim:
    id: str
    unit_id: str
    predicate: PredicateRef
    polarity: Polarity
    citation: Citation
    claim_type: Literal["WitnessClaim"] = "WitnessClaim"


@dataclass(frozen=True)
class ScanProvenance:
    observation_id: str
    supporting_observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.observation_id or not self.observation_id.strip():
            raise ValueError("ScanProvenance.observation_id is required.")


@dataclass(frozen=True)
class ScanClaim:
    id: str
    scope: ScopeRef
    unit_type: UnitType
    predicate: PredicateRef
    scanned_units: frozenset[str]
    positive_units: frozenset[str]
    negative_units: frozenset[str]
    exhaustive: bool
    provenance: ScanProvenance
    claim_type: Literal["ScanClaim"] = "ScanClaim"

    def __post_init__(self) -> None:
        if not self.exhaustive:
            raise ValueError("ScanClaim is only minted with exhaustive=True; partial scans stay observations.")
        if self.positive_units & self.negative_units:
            raise ValueError("ScanClaim positive/negative sets must be disjoint.")
        cover = self.positive_units | self.negative_units
        if cover != self.scanned_units:
            raise ValueError("ScanClaim positive ∪ negative must equal scanned_units.")


@dataclass(frozen=True)
class DerivedProvenance:
    """Anchor for a computed ValueClaim. The kernel re-runs ``operation``
    over the input claims' values and rejects on mismatch — code_run
    output is *not* trusted as the source of truth (see plant._run_operation).

    Adapted from PCN claim-bound numerics (arXiv:2509.06902) and PoT
    disentangling of computation from reasoning (arXiv:2211.12588) — the
    kernel is the executor, not an LLM-written program.
    """

    operation: Literal[
        "sum", "product", "percent_of", "difference", "quotient", "max", "min",
    ]
    input_claim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.input_claim_ids:
            raise ValueError("DerivedProvenance requires at least one input claim id.")
        if any(not isinstance(i, str) or not i for i in self.input_claim_ids):
            raise ValueError("DerivedProvenance.input_claim_ids must be non-empty strings.")


@dataclass(frozen=True)
class ValueClaim:
    """Typed value at a unit (extracted) OR derived from prior closed
    ValueClaims via a whitelisted arithmetic operation.

    Exactly one of ``citation`` or ``derived`` is set. Extracted claims
    have ``unit_id`` + ``citation``; derived claims have ``unit_id=None``
    + ``derived`` (their domain residency is established via every
    input claim's unit_id, checked at closure time).
    """

    id: str
    unit_id: Optional[str]
    field: str
    value: Any
    value_type: Literal["numeric", "percentage", "date_iso", "text", "integer_count"]
    citation: Optional[Citation] = None
    derived: Optional[DerivedProvenance] = None
    claim_type: Literal["ValueClaim"] = "ValueClaim"

    def __post_init__(self) -> None:
        if not self.field or not str(self.field).strip():
            raise ValueError("ValueClaim.field must be non-empty.")
        has_citation = self.citation is not None
        has_derived = self.derived is not None
        if has_citation == has_derived:
            raise ValueError(
                "ValueClaim must carry exactly one of (citation, derived); "
                f"got citation={has_citation!r} derived={has_derived!r}."
            )
        if has_citation and not self.unit_id:
            raise ValueError("Extracted ValueClaim requires unit_id.")
        if has_derived and self.unit_id is not None:
            raise ValueError("Derived ValueClaim must have unit_id=None.")

    @property
    def is_derived(self) -> bool:
        return self.derived is not None


Claim = Union[WitnessClaim, ScanClaim, ValueClaim]
