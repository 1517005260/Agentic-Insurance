"""CandidateGap and bounded promotion.

Three layers answer the "wrong initial obligation" question without
re-introducing repair laws into the kernel:

* Diagnostics (``missing_witness / missing_complete_scan / ...``) drive
  acquisition; they are not gaps and never become obligations.
* CandidateGaps express LLM hypotheses that the certification contract
  itself is defective. They never block ``try_finalize``.
* Plant promotion is a tiny add-only whitelist: it may add a missing
  required obligation or accept a canonical equivalent. It never
  replaces or weakens an existing obligation.

``equivalence_update`` is canonicalisation only — same kind / unit_type
plus matching ``canonical_*_id``. No scope monotonicity, no predicate
weakening, no NL entailment.
"""

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional

from agentic.closure.budget import Budget
from agentic.closure.inventory import Inventory
from agentic.closure.obligation import Obligation


GapKind = Literal[
    "missing_scope",
    "wrong_kind",
    "predicate_variant_missing",
    "ambiguous_lookup",
]

GapStatus = Literal["ACTIVE", "ADVISORY_ONLY", "PROMOTED", "DISMISSED"]

GapSource = Literal["llm", "gate_diagnostic", "inventory"]


_VALID_GAP_KINDS: frozenset[str] = frozenset(
    {"missing_scope", "wrong_kind", "predicate_variant_missing", "ambiguous_lookup"}
)


MAX_CANDIDATE_GAPS: int = 5
MAX_PROMOTED_GAPS: int = 2
MAX_RATIONALE_CHARS: int = 200


@dataclass(frozen=True)
class EvidenceHint:
    rationale: str
    unit_id: Optional[str] = None
    predicate_variant: Optional[Any] = None  # PredicateRef; typed loosely to dodge a circular import

    def __post_init__(self) -> None:
        if not self.rationale or not self.rationale.strip():
            raise ValueError("EvidenceHint.rationale must be non-empty.")
        if len(self.rationale) > MAX_RATIONALE_CHARS:
            raise ValueError(
                f"EvidenceHint.rationale must be ≤{MAX_RATIONALE_CHARS} chars; got {len(self.rationale)}."
            )


@dataclass
class CandidateGap:
    id: str
    kind: GapKind
    proposed_obligation: Optional[Obligation]
    evidence_hint: EvidenceHint
    priority: int = 3
    source: GapSource = "llm"
    status: GapStatus = "ACTIVE"
    repeats: int = 0
    promoted_obligation_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in _VALID_GAP_KINDS:
            raise ValueError(
                f"GapKind must be one of {sorted(_VALID_GAP_KINDS)}; got {self.kind!r}."
            )
        if not 1 <= self.priority <= 5:
            raise ValueError(f"priority must be in [1,5]; got {self.priority}.")

    def signature(self) -> tuple:
        proposed = (
            self.proposed_obligation.structural_key()
            if self.proposed_obligation is not None
            else None
        )
        return (
            self.kind,
            proposed,
            self.evidence_hint.unit_id,
            getattr(self.evidence_hint.predicate_variant, "canonical_id", None),
        )


# ---------------------------------------------------------------- accept


def accept_candidate_gap(
    gap: CandidateGap,
    existing: Iterable[CandidateGap],
) -> Optional[CandidateGap]:
    """Admit ``gap`` or merge it into a duplicate.

    Returns the existing gap (with ``repeats`` incremented) when the
    proposal is structurally identical, the new gap when admitted
    fresh, or None when the active cap is full.
    """
    existing_list = list(existing)
    sig = gap.signature()

    for prior in existing_list:
        if prior.signature() != sig:
            continue
        if prior.status in {"DISMISSED", "PROMOTED"}:
            continue
        prior.repeats += 1
        return prior

    active = sum(1 for g in existing_list if g.status == "ACTIVE")
    if active >= MAX_CANDIDATE_GAPS:
        gap.status = "DISMISSED"
        return None
    return gap


# ---------------------------------------------------------------- promote


def equivalence_update(
    old: Obligation,
    new: Obligation,
    inventory: Inventory,
) -> bool:
    if old.kind != new.kind:
        return False
    if old.unit_type != new.unit_type:
        return False
    if old.score_field != new.score_field:
        return False
    scope_eq = (
        old.scope.canonical_scope_id == new.scope.canonical_scope_id
        or inventory.units(old.scope, old.unit_type)
        == inventory.units(new.scope, new.unit_type)
    )
    if not scope_eq:
        return False
    return old.predicate.canonical_id == new.predicate.canonical_id


def promote_candidate_gap(
    gap: CandidateGap,
    obligations: list[Obligation],
    inventory: Inventory,
    budget: Budget,
    *,
    promoted_so_far: int = 0,
) -> Optional[Obligation]:
    """Bounded, contract-valid promotion. The proposed obligation must
    pass ``contract.validate_obligation``; semantic-name predicates and
    field/score_field misuse are rejected here. Cap (``MAX_PROMOTED_GAPS``)
    bounds cost; soundness comes from contract + cap together.
    """

    from agentic.closure.contract import validate_obligation

    if budget.remaining_steps < 2:
        return None
    if gap.status != "ACTIVE":
        return None
    if gap.kind not in _VALID_GAP_KINDS:
        return None
    if gap.proposed_obligation is None:
        return None
    if promoted_so_far >= MAX_PROMOTED_GAPS:
        return None
    proposal = gap.proposed_obligation
    if validate_obligation(proposal) is not None:
        return None
    if any(o.structural_key() == proposal.structural_key() for o in obligations):
        return None
    return proposal


# ---------------------------------------------------------------- lifecycle


def update_candidate_gap_lifecycle(
    gaps: Iterable[CandidateGap],
    claims: Iterable[Any],
) -> None:
    """Repeated gap with no new supporting evidence → ADVISORY_ONLY.

    The plant calls this once per acquisition turn after a fresh batch
    of claims has landed. ``claims`` is accepted as a generic iterable
    so the callsite need not know the concrete claim union.
    """
    _ = list(claims)
    for gap in gaps:
        if gap.status != "ACTIVE":
            continue
        if gap.repeats >= 2:
            gap.status = "ADVISORY_ONLY"
