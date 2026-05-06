"""``try_finalize`` — the read-only certification gate.

The gate never mutates obligations. By contract, ``Plant.run_closure``
is the sole writer of ``CLOSED`` status; ``try_finalize`` only walks
the obligation list, refuses if any required obligation is still open
or invalid, and otherwise composes the certified payload.

Decisions are ``Certified`` or ``Abstain`` only (strict mode). A
``Continue`` is returned while the budget allows another acquisition
turn — the agent loop sees this and asks the LLM for the next action.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

from agentic.closure.budget import Budget
from agentic.closure.claims import Claim
from agentic.closure.inventory import Inventory
from agentic.closure.obligation import Obligation


class KernelInvariantError(RuntimeError):
    """Raised when a CLOSED obligation lacks a value/closed_by, or when the gate is asked to mutate state."""


@dataclass(frozen=True)
class ObligationSummary:
    id: str
    kind: str
    canonical_scope_id: str
    unit_type: str
    canonical_predicate_id: str
    status: str
    closed_value: object = None
    closed_by: tuple = ()
    failure_kind: Optional[str] = None
    diagnostic_data: Optional[dict] = None

    @classmethod
    def of(cls, o: Obligation) -> "ObligationSummary":
        return cls(
            id=o.id,
            kind=o.kind,
            canonical_scope_id=o.scope.canonical_scope_id,
            unit_type=o.unit_type,
            canonical_predicate_id=o.predicate.canonical_id,
            status=o.status,
            closed_value=o.closed_value,
            closed_by=tuple(o.closed_by),
            failure_kind=o.failure_kind,
            diagnostic_data=o.diagnostic_data,
        )


@dataclass(frozen=True)
class Certified:
    answer: str
    closed_obligations: tuple[ObligationSummary, ...]
    decision: str = "CERTIFIED"


@dataclass(frozen=True)
class Continue:
    reason: str
    open_obligations: tuple[ObligationSummary, ...]
    decision: str = "CONTINUE"


@dataclass(frozen=True)
class Abstain:
    reason: str
    open_obligations: tuple[ObligationSummary, ...]
    closed_obligations: tuple[ObligationSummary, ...] = ()
    decision: str = "ABSTAIN"


FinalizeResult = Union[Certified, Continue, Abstain]


# Diagnostic codes that no amount of further acquisition can resolve.
# When one of these surfaces on a still-open obligation, the gate
# abstains immediately rather than burning more budget.
_DEAD_END_FAILURES: frozenset[str] = frozenset(
    {"ambiguous_lookup", "argmax_tie", "unknown_obligation_kind", "empty_domain"}
)


def _required_open(obligations: Iterable[Obligation]) -> list[Obligation]:
    return [o for o in obligations if o.required and o.status != "CLOSED"]


def _required_closed(obligations: Iterable[Obligation]) -> list[Obligation]:
    return [o for o in obligations if o.required and o.status == "CLOSED"]


def _verify_closed_invariant(obligations: Iterable[Obligation]) -> None:
    for o in obligations:
        if o.status != "CLOSED":
            continue
        if o.closed_value is None or not o.closed_by:
            raise KernelInvariantError(
                f"Obligation {o.id!r} is CLOSED but missing closed_value/closed_by; "
                "Plant.run_closure wrote a bad transition."
            )


def try_finalize(
    obligations: Sequence[Obligation],
    claims: Sequence[Claim],
    inventory: Inventory,
    budget: Budget,
    *,
    draft_text: Optional[str] = None,
) -> FinalizeResult:
    _verify_closed_invariant(obligations)

    required = [o for o in obligations if o.required]
    if not required:
        # No certification contract — refuse to certify a no-op answer.
        return Abstain(
            reason="no_required_obligations",
            open_obligations=(),
        )

    open_required = _required_open(obligations)

    if not open_required:
        closed = _required_closed(obligations)
        answer = build_answer_from_closed_obligations(closed, claims, draft_text=draft_text)
        return Certified(
            answer=answer,
            closed_obligations=tuple(ObligationSummary.of(o) for o in closed),
        )

    dead_end = next(
        (o for o in open_required if o.failure_kind in _DEAD_END_FAILURES),
        None,
    )
    if dead_end is not None:
        return Abstain(
            reason=dead_end.failure_kind or "unknown_failure",
            open_obligations=tuple(ObligationSummary.of(o) for o in open_required),
            closed_obligations=tuple(ObligationSummary.of(o) for o in _required_closed(obligations)),
        )

    if budget.has_room():
        first = open_required[0]
        return Continue(
            reason=first.failure_kind or "missing_evidence",
            open_obligations=tuple(ObligationSummary.of(o) for o in open_required),
        )

    return Abstain(
        reason=budget.exhausted_kind or "budget_exhausted_with_open_obligations",
        open_obligations=tuple(ObligationSummary.of(o) for o in open_required),
        closed_obligations=tuple(ObligationSummary.of(o) for o in _required_closed(obligations)),
    )


# ---------------------------------------------------------------- answer composition


def _format_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def build_answer_from_closed_obligations(
    obligations: Sequence[Obligation],
    claims: Sequence[Claim],
    *,
    draft_text: Optional[str] = None,
) -> str:
    if not obligations:
        return "Certified: (no obligations required)\n"
    lines = ["Certified:"]
    for o in obligations:
        lines.append(
            f"- {o.id} ({o.kind}): {_format_value(o.closed_value)}"
        )
    if draft_text and draft_text.strip():
        lines.append("")
        lines.append(draft_text.strip())

    citation_ids = {cid for o in obligations for cid in o.closed_by}
    relevant = [c for c in claims if getattr(c, "id", None) in citation_ids]
    if relevant:
        lines.append("")
        lines.append("Citations:")
        for c in relevant:
            lines.append(f"- {c.id}: {_summarize_claim_citation(c)}")
    return "\n".join(lines) + "\n"


def _summarize_claim_citation(claim) -> str:
    cite = getattr(claim, "citation", None)
    if cite is not None:
        return f"{cite.file_id}#{cite.page_id}:{cite.unit_id}"
    prov = getattr(claim, "provenance", None)
    if prov is not None:
        return f"scan obs={prov.observation_id} units={len(claim.scanned_units)}"
    return getattr(claim, "id", "<unknown>")
