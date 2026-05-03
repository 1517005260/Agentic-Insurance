"""State-machine table for :class:`agentic.proof.types.Obligation`.

Every status mutation in the proof gate must go through one
chokepoint — :meth:`agentic.proof.obligation_store.ObligationStore.apply_event`
— which consumes the declarative ``_TRANSITIONS`` table here.
``record_close`` / ``record_decompose`` / ``record_challenge_open`` /
``record_challenge_discharged`` / ``record_retire`` become one-liners
that call ``apply_event`` with the right :class:`Event`. M1 race
window (closed_by set AFTER status flip) closes because the delta is
applied under the same lock as the status change.

Multi-challenge accumulation is recorded as a metadata-only
self-loop on ``CHALLENGED`` via
:meth:`ObligationStore.record_partial_discharge` — it is NOT a
transition; this table never has a self-loop.

Two read-only graph queries also live here so the closure cone
formula has one home:

* :func:`can_close` — "may obligation o be closed right now?"
  Consulted by reconcile step 3 (OPEN → CLOSED) AND step 4
  (DECOMPOSED → CLOSED). Encodes B1's defensive guard against
  ancestor-CHALLENGED + DECOMPOSED-descendant.
* :func:`closure_cone` — set of obligation ids that share a closure
  cone with at least one required obligation. B3's fix:
  ``cone(R) := ⋃_{r ∈ R} ⋃_{a ∈ ancestors_inclusive(r)} descendants_inclusive(a)``.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING

from agentic.proof.types import (
    Obligation,
    ObligationStatus,
)

if TYPE_CHECKING:
    from agentic.proof.state.obligation_store import ObligationStore


class Event(str, Enum):
    """Typed events that can produce a state transition."""

    CLOSE               = "close"                # OPEN -> CLOSED       (Γ_kind succeeded)
    DECOMPOSE           = "decompose"            # OPEN -> DECOMPOSED   (children created)
    CHILDREN_CLOSED     = "children_closed"      # DECOMPOSED -> CLOSED (parent Γ re-eval)
    CHALLENGE_OPEN      = "challenge_open"       # OPEN -> CHALLENGED
    CHALLENGE_DISCHARGE = "challenge_discharge"  # CHALLENGED -> OPEN (last challenge cleared)
    RETIRE              = "retire"               # OPEN/CHALLENGED/DECOMPOSED -> RETIRED


@dataclass(frozen=True)
class TransitionRule:
    from_state: ObligationStatus
    event: Event
    to_state: ObligationStatus


# Declarative table — adding a new transition means adding a row here,
# never editing apply_event. Self-loops are forbidden (CHALLENGED
# multi-discharge metadata bypasses the table; see record_partial_discharge).
_TRANSITIONS: Dict[Tuple[ObligationStatus, Event], TransitionRule] = {
    (ObligationStatus.OPEN, Event.CLOSE):
        TransitionRule(ObligationStatus.OPEN, Event.CLOSE, ObligationStatus.CLOSED),
    (ObligationStatus.OPEN, Event.DECOMPOSE):
        TransitionRule(ObligationStatus.OPEN, Event.DECOMPOSE, ObligationStatus.DECOMPOSED),
    (ObligationStatus.OPEN, Event.CHALLENGE_OPEN):
        TransitionRule(ObligationStatus.OPEN, Event.CHALLENGE_OPEN, ObligationStatus.CHALLENGED),
    (ObligationStatus.OPEN, Event.RETIRE):
        TransitionRule(ObligationStatus.OPEN, Event.RETIRE, ObligationStatus.RETIRED),
    (ObligationStatus.CHALLENGED, Event.CHALLENGE_DISCHARGE):
        TransitionRule(ObligationStatus.CHALLENGED, Event.CHALLENGE_DISCHARGE, ObligationStatus.OPEN),
    # CHALLENGED -> RETIRED is INTENTIONALLY absent. Cross-store
    # discharge sequences CHALLENGE_DISCHARGE -> RETIRE so the
    # retired obligation is RETIRED-from-OPEN, never CHALLENGED.
    (ObligationStatus.DECOMPOSED, Event.CHILDREN_CLOSED):
        TransitionRule(ObligationStatus.DECOMPOSED, Event.CHILDREN_CLOSED, ObligationStatus.CLOSED),
    (ObligationStatus.DECOMPOSED, Event.RETIRE):
        TransitionRule(ObligationStatus.DECOMPOSED, Event.RETIRE, ObligationStatus.RETIRED),
}


def lookup(from_state: ObligationStatus, event: Event) -> Optional[TransitionRule]:
    return _TRANSITIONS.get((from_state, event))


# ----------------------------------------------------------- graph queries


def ancestor_challenged(store: "ObligationStore", obligation_id: str) -> bool:
    for ancestor in store.ancestors(obligation_id):
        if ancestor.status == ObligationStatus.CHALLENGED:
            return True
    return False


def can_close(obligation: Obligation, store: "ObligationStore") -> Optional[str]:
    """Returns ``None`` when ``obligation`` may be closed right now,
    or a short reason string otherwise. Consulted by both reconcile
    step 3 (OPEN → CLOSED) and step 4 (DECOMPOSED → CLOSED). Encodes
    B1's defensive guard."""
    if obligation.status not in (ObligationStatus.OPEN, ObligationStatus.DECOMPOSED):
        return f"status_{obligation.status.value.lower()}"
    if ancestor_challenged(store, obligation.id):
        return "ancestor_challenged"
    return None


def closure_cone(store: "ObligationStore", required_ids: Iterable[str]) -> Set[str]:
    """B3 fix:
    ``cone(R) := ⋃_{r ∈ R} ⋃_{a ∈ ancestors_inclusive(r)} descendants_inclusive(a)``.

    For each required obligation r, walk the chain from r up to the
    root and at each ancestor (inclusive of r itself) include the
    entire subtree under that ancestor. CHALLENGED nodes anywhere in
    that closure block finalize."""
    cone: Set[str] = set()
    for rid in required_ids:
        for ancestor in store.ancestors_inclusive(rid):
            for node in store.descendants_inclusive(ancestor.id):
                cone.add(node.id)
    return cone
