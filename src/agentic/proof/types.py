"""Shared dataclasses and type aliases for the proof-obligation gate.

These types are the boundary contract between the LLM-facing tools, the
plant, the closure rules, and the ProofAgent. Keeping them in one place
avoids the schema drift that arises when each tool maintains its own
envelope.

Two design notes worth surfacing here rather than scattering them:

* All references to ``unit_id`` are page global ids (``<file>/<page>``)
  for ``unit_type=="file"`` aggregation, or section ids
  (``<file>:sec_NNN``) for ``unit_type=="section"``. The plant resolves
  scope and unit_type together — never look up a section id with a page
  helper or vice versa.

* The state machine in ``ObligationStatus`` is small and terminal-aware
  by design. ``CLOSED`` is terminal in v1; mistakes route through
  ``CHALLENGED`` (during exploration) or ``ABSTAIN`` (at finalize), not
  through re-opening a closed obligation.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Tuple


# ----------------------------------------------------------------- enums


class ObligationKind(str, Enum):
    EXISTS = "exists"
    COUNT = "count"
    SET = "set"
    FORALL = "forall"
    NEGATION = "negation"
    ARGMAX = "argmax"


class ObligationStatus(str, Enum):
    OPEN = "OPEN"
    CHALLENGED = "CHALLENGED"
    DECOMPOSED = "DECOMPOSED"
    CLOSED = "CLOSED"
    RETIRED = "RETIRED"


class ClaimType(str, Enum):
    WITNESS = "WitnessClaim"
    SCAN = "ScanClaim"
    COMPARISON = "ComparisonClaim"        # plant-internal only
    BOUND = "BoundClaim"                  # non-certifying hint
    OPENWORLD_WITNESS = "OpenWorldWitnessClaim"  # business mode only


class ObservationType(str, Enum):
    FILE_LIST = "FileList"
    TOC = "Toc"
    PAGE_HITS_EXHAUSTIVE = "PageHitsExhaustive"
    PAGE_HITS_PARTIAL = "PageHitsPartial"
    PAGE_CANDIDATES = "PageCandidates"
    PAGE_CONTENT = "PageContent"
    COMPUTE_RESULT = "ComputeResult"


UnitType = Literal["file", "section"]
Polarity = Literal["positive", "negative"]
DerivedBy = Literal[
    "root", "and_split", "scope_partition", "case_split",
    "map_over_domain", "user_constraint", "challenge_replacement",
]
RepairKind = Literal[
    "scope_too_narrow", "scope_too_broad", "predicate_mismatch",
    "missing_subobligation", "wrong_question_kind",
]
DecompositionRule = Literal[
    "and_split", "scope_partition", "case_split", "map_over_domain",
]


# ----------------------------------------------------------- references


@dataclass(frozen=True)
class ScopeRef:
    """Resolved scope for an obligation.

    ``file_ids`` is required and non-empty. ``section_ids`` is None for
    file-level scopes; otherwise it must be a subset of the sections
    inside ``file_ids``. ``sealed`` is set at creation and never
    toggles — sealed_scope_override below is a separate explicit knob
    that weakens closure but cannot be flipped after the obligation
    was created.
    """

    file_ids: FrozenSet[str]
    section_ids: Optional[FrozenSet[str]]
    sealed: bool

    def is_file_level(self) -> bool:
        return self.section_ids is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_ids": sorted(self.file_ids),
            "section_ids": sorted(self.section_ids) if self.section_ids is not None else None,
            "sealed": self.sealed,
        }


@dataclass(frozen=True)
class Citation:
    """A pointer to where a claim's evidence lives in source material.

    The plant validates that ``file_id``, ``page_id`` and (when
    supplied) ``span`` exist. Tools may emit citations with just
    ``page_id`` for whole-page witnesses or with ``span`` for inline
    proofs. Citations are append-only — never rewritten by later tools.
    """

    file_id: str
    page_id: str
    span: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"file_id": self.file_id, "page_id": self.page_id}
        if self.span is not None:
            out["span"] = self.span
        return out


# ----------------------------------------------------------- predicates / scores


@dataclass(frozen=True)
class PredicateSpec:
    """The serialized form of a predicate as referenced by obligations
    and claims. ``name`` selects a primitive (or ``"and"`` for the
    composite); ``args`` is a JSON-serializable canonical form.

    Two specs are entailment-equal iff their canonical hash matches.
    For ``and_(...)`` the conjuncts are serialized in canonical
    (``serialize`` of each, sorted) order so set-equality is detectable
    via string hash.
    """

    name: str
    args: Tuple[Tuple[str, Any], ...]    # sorted (key, value) pairs for canonicality

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "args": dict(self.args)}


@dataclass(frozen=True)
class ScoreSpec:
    """Reference to a registered score extractor."""

    name: str
    args: Tuple[Tuple[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "args": dict(self.args)}


# ---------------------------------------------------------------- core records


@dataclass
class ObligationSpec:
    """The user-facing input to obligation_create. Plant resolves and
    validates each field before storing the materialized
    :class:`Obligation`."""

    kind: ObligationKind
    scope: ScopeRef
    unit_type: UnitType
    predicate: PredicateSpec
    score: Optional[ScoreSpec] = None
    polarity: Polarity = "positive"
    required: bool = True
    parent_id: Optional[str] = None
    derived_by: DerivedBy = "root"
    sealed_scope_override: bool = False
    tie_policy: Literal["first", "all", "error"] = "first"   # argmax only


@dataclass
class Obligation:
    id: str
    spec: ObligationSpec
    is_root: bool
    status: ObligationStatus
    closed_by: List[str] = field(default_factory=list)
    closed_value: Any = None
    open_challenges: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Observation:
    id: str
    tool_name: str
    observation_type: ObservationType
    payload: Dict[str, Any]
    citations: List[Citation]
    timestamp: float


@dataclass
class Claim:
    id: str
    observation_id: Optional[str]
    claim_type: ClaimType
    scope: ScopeRef
    unit_type: UnitType
    predicate: Optional[PredicateSpec]
    score: Optional[ScoreSpec]
    positive_units: List[str]
    negative_units: List[str]
    value_map: Dict[str, Any]
    citations: List[Citation]
    derivation: Literal["auto_extract", "llm_proposed", "plant_aggregated"]


@dataclass
class Binding:
    obligation_id: str
    claim_id: str
    auto: bool


@dataclass
class Challenge:
    """An accepted challenge sits in CHALLENGED state until its
    mechanical postcondition is satisfied. Each repair_kind has its
    own discharge rule (see plant.reconcile)."""

    id: str
    obligation_id: str
    repair_kind: RepairKind
    evidence_ids: List[str]
    reason: str
    status: Literal["pending", "discharged", "rejected"] = "pending"
    expected: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClosureResult:
    """Output of a Γ_kind closure rule."""

    success: bool
    value: Any = None
    used_claim_ids: List[str] = field(default_factory=list)
    diagnostic: Optional[str] = None


@dataclass
class ToolDiagnostic:
    """Per-obligation hint inserted into gate.diagnose."""

    obligation_id: str
    failure_kind: str
    suggested_tools: List[str] = field(default_factory=list)
    suggested_repair_kind: Optional[str] = None  # which obligation_challenge repair_kind unblocks failure_kind
    cursor: Optional[Dict[str, Any]] = None    # for map_over_domain


@dataclass
class GateView:
    """The state view appended to tool_results after a state-changing
    call. Not exposed as an LLM-facing tool; the LLM consumes it only
    as part of the tool_result envelope."""

    open_obligations: List[Dict[str, Any]]
    closed_obligations: List[Dict[str, Any]]
    challenged_obligations: List[Dict[str, Any]]
    diagnostics: List[ToolDiagnostic]
    recent_claims: List[Dict[str, Any]]
    abstain_recommended: bool
    abstain_reason: Optional[str]
