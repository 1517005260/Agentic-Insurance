"""Typed closure kernel.

The kernel decides one thing: when an obligation is closed by validated
evidence. The LLM may search and propose claims; the plant validates
them; only the gate certifies.

Trust-boundary limitations:

* NL→obligation faithfulness is attested by the planner LLM and
  recorded in ``predicate.canonical_id``. The kernel verifies the
  certified statement is supported by claims; it does NOT verify
  the statement matches the user's question.
* Soundness is relative to ``Inventory.units(scope, unit_type)``.
  If the corpus' atom granularity is coarser than the gold answer's
  granularity, set/count/forall/negation certify a true statement
  about the atoms but not about the gold items.
* The gate does not host arithmetic. Computed values (e.g. 27% of
  90,000) require an explicit ValueClaim that anchors the result;
  otherwise the draft check rejects the uncited number.
"""

from agentic.closure.budget import Budget
from agentic.closure.candidate_gap import (
    MAX_CANDIDATE_GAPS,
    MAX_PROMOTED_GAPS,
    CandidateGap,
    EvidenceHint,
    GapKind,
    GapStatus,
    accept_candidate_gap,
    equivalence_update,
    promote_candidate_gap,
    update_candidate_gap_lifecycle,
)
from agentic.closure.claims import (
    Citation,
    Claim,
    Polarity,
    ScanClaim,
    ScanProvenance,
    ValueClaim,
    WitnessClaim,
)
from agentic.closure.closures import (
    CLOSURE_RULES,
    Closed,
    ClosureResult,
    Open,
    close_argmax_exact,
    close_count,
    close_exists,
    close_forall,
    close_lookup,
    close_negation,
    close_set,
    try_close,
)
from agentic.closure.complete_scan import complete_scan, scan_coverage_diff
from agentic.closure.finalize import (
    Abstain,
    Certified,
    Continue,
    FinalizeResult,
    KernelInvariantError,
    ObligationSummary,
    build_answer_from_closed_obligations,
    try_finalize,
)
from agentic.closure.inventory import (
    Inventory,
    InventoryAdapter,
    UnknownScopeError,
)
from agentic.closure.obligation import (
    Obligation,
    ObligationKind,
    ObligationStatus,
    PredicateRef,
    ScopeRef,
    UnitType,
)
from agentic.closure.plant import ErrorEnvelope, Plant
from agentic.closure.session import Observation, ProofSession

__all__ = [
    # obligation
    "Obligation",
    "ObligationKind",
    "ObligationStatus",
    "ScopeRef",
    "PredicateRef",
    "UnitType",
    # claims
    "Citation",
    "Claim",
    "Polarity",
    "ScanClaim",
    "ScanProvenance",
    "ValueClaim",
    "WitnessClaim",
    # inventory
    "Inventory",
    "InventoryAdapter",
    "UnknownScopeError",
    # complete scan
    "complete_scan",
    "scan_coverage_diff",
    # closures
    "CLOSURE_RULES",
    "Closed",
    "ClosureResult",
    "Open",
    "close_argmax_exact",
    "close_count",
    "close_exists",
    "close_forall",
    "close_lookup",
    "close_negation",
    "close_set",
    "try_close",
    # candidate gap
    "MAX_CANDIDATE_GAPS",
    "MAX_PROMOTED_GAPS",
    "CandidateGap",
    "EvidenceHint",
    "GapKind",
    "GapStatus",
    "accept_candidate_gap",
    "equivalence_update",
    "promote_candidate_gap",
    "update_candidate_gap_lifecycle",
    # plant
    "ErrorEnvelope",
    "Plant",
    # session
    "Observation",
    "ProofSession",
    # finalize
    "Abstain",
    "Budget",
    "Certified",
    "Continue",
    "FinalizeResult",
    "KernelInvariantError",
    "ObligationSummary",
    "build_answer_from_closed_obligations",
    "try_finalize",
]
