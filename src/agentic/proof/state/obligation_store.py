"""ObligationStore — state machine and lookup for proof obligations.

Allowed transitions:

    OPEN  ──challenge_accepted──>  CHALLENGED
    CHALLENGED ──discharge──>      OPEN
    OPEN  ──decompose──>           DECOMPOSED
    OPEN  ──close (Γ success)──>   CLOSED
    OPEN  ──retire (cert/repl)──>  RETIRED
    DECOMPOSED ──auto_close──>     CLOSED
    DECOMPOSED ──retire──>         RETIRED

CLOSED and RETIRED are terminal: a wrong root predicate exits via
ABSTAIN at finalize, not by re-opening. RETIRED obligations stay in the
store for audit and are excluded from ``active_required_open`` quorums.
"""
import copy
import threading
import uuid
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from agentic.proof.state.transitions import Event, lookup as _lookup_transition
from agentic.proof.types import (
    Obligation,
    ObligationKind,
    ObligationSpec,
    ObligationStatus,
)


class ObligationStateError(RuntimeError):
    """Illegal state transition attempted."""


# Derived from the declarative ``transitions._TRANSITIONS`` table —
# ``apply_event`` is the canonical entry point. This map is kept for
# legacy callers of ``transition()`` that still pass raw status pairs.
_VALID_TRANSITIONS: Dict[ObligationStatus, frozenset] = {
    ObligationStatus.OPEN: frozenset({
        ObligationStatus.CHALLENGED,
        ObligationStatus.DECOMPOSED,
        ObligationStatus.CLOSED,
        ObligationStatus.RETIRED,
    }),
    ObligationStatus.CHALLENGED: frozenset({
        ObligationStatus.OPEN,
    }),
    ObligationStatus.DECOMPOSED: frozenset({
        ObligationStatus.CLOSED,
        ObligationStatus.RETIRED,
    }),
    ObligationStatus.CLOSED: frozenset(),
    ObligationStatus.RETIRED: frozenset(),
}


class ObligationStore:
    """In-memory, append-only obligation store.

    Append-only: every transition leaves a record on the obligation's
    ``history`` list so the trace can reconstruct who closed what and
    why. Reads are not lock-protected because the agent loop is single-
    threaded; we only lock state-changing helpers in case a future tool
    fan-out shows up.
    """

    def __init__(self) -> None:
        self._items: Dict[str, Obligation] = {}
        self._lock = threading.Lock()
        self._root_id: Optional[str] = None
        self._wrong_kind_used: bool = False

    # ----------------------------------------------------------- lookups

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Obligation]:
        return iter(self._items.values())

    def get(self, obligation_id: str) -> Optional[Obligation]:
        return self._items.get(obligation_id)

    def all_obligations(self) -> List[Obligation]:
        return list(self._items.values())

    def by_status(self, status: ObligationStatus) -> List[Obligation]:
        return [o for o in self._items.values() if o.status == status]

    def root(self) -> Optional[Obligation]:
        if self._root_id is None:
            return None
        return self._items.get(self._root_id)

    def has_active_required(self) -> bool:
        """Used by Phase A→Phase B gate."""
        for o in self._items.values():
            if not o.spec.required:
                continue
            if o.status in (ObligationStatus.RETIRED,):
                continue
            return True
        return False

    def active_required_open(self) -> List[Obligation]:
        """Required obligations that have not reached CLOSED or RETIRED.

        Used by gate.diagnose to flag work the agent still owes.
        """
        out: List[Obligation] = []
        for o in self._items.values():
            if not o.spec.required:
                continue
            if o.status in (ObligationStatus.CLOSED, ObligationStatus.RETIRED):
                continue
            out.append(o)
        return out

    def active_required_closed(self) -> List[Obligation]:
        """Closed required obligations (excludes RETIRED)."""
        return [
            o for o in self._items.values()
            if o.spec.required and o.status == ObligationStatus.CLOSED
        ]

    def descendants(self, obligation_id: str) -> List[Obligation]:
        """All transitive descendants — used by closure-cone CHALLENGED check."""
        out: List[Obligation] = []
        seen: set[str] = set()
        frontier = [obligation_id]
        while frontier:
            cur = frontier.pop()
            obligation = self._items.get(cur)
            if obligation is None:
                continue
            for child_id in obligation.children_ids:
                if child_id in seen:
                    continue
                seen.add(child_id)
                child = self._items.get(child_id)
                if child is not None:
                    out.append(child)
                    frontier.append(child_id)
        return out

    def descendants_inclusive(self, obligation_id: str) -> List[Obligation]:
        """``descendants(id)`` plus the obligation itself. Used by the
        closure cone formula
        ``cone(R) := ⋃_{r ∈ R} ⋃_{a ∈ ancestors_inclusive(r)} descendants_inclusive(a)``
        (B3 fix). Returning a list (not a set) keeps ordering stable for
        diagnostics."""
        self_obl = self._items.get(obligation_id)
        out: List[Obligation] = [self_obl] if self_obl is not None else []
        out.extend(self.descendants(obligation_id))
        return out

    def ancestors(self, obligation_id: str) -> List[Obligation]:
        out: List[Obligation] = []
        cur_id: Optional[str] = obligation_id
        seen: set[str] = set()
        while cur_id:
            o = self._items.get(cur_id)
            if o is None or cur_id in seen:
                break
            seen.add(cur_id)
            cur_id = o.spec.parent_id
            if cur_id and cur_id in self._items:
                out.append(self._items[cur_id])
        return out

    def ancestors_inclusive(self, obligation_id: str) -> List[Obligation]:
        """Chain ``[id, parent, grandparent, ..., root]`` — includes the
        obligation itself. Used by B3's closure_cone formula."""
        self_obl = self._items.get(obligation_id)
        out: List[Obligation] = [self_obl] if self_obl is not None else []
        out.extend(self.ancestors(obligation_id))
        return out

    # ----------------------------------------------------------- writes

    def insert(
        self,
        spec: ObligationSpec,
        is_root: bool,
    ) -> Obligation:
        """Add a new OPEN obligation to the store and return it."""
        with self._lock:
            obligation_id = self._mint_id(spec.kind)
            obligation = Obligation(
                id=obligation_id,
                spec=spec,
                is_root=is_root,
                status=ObligationStatus.OPEN,
                history=[
                    {"event": "create", "status": ObligationStatus.OPEN.value, "is_root": is_root}
                ],
            )
            self._items[obligation_id] = obligation
            if is_root and self._root_id is None:
                self._root_id = obligation_id
            if spec.parent_id is not None:
                parent = self._items.get(spec.parent_id)
                if parent is not None and obligation_id not in parent.children_ids:
                    parent.children_ids.append(obligation_id)
            return obligation

    def transition(
        self,
        obligation_id: str,
        new_status: ObligationStatus,
        *,
        event: str,
        meta: Optional[Dict[str, object]] = None,
    ) -> Obligation:
        """Move an obligation to ``new_status`` if the transition is
        valid. Records the event on the obligation's history.

        Prefer :meth:`apply_event` for new code — it consults the
        declarative ``transitions._TRANSITIONS`` table rather than the
        legacy ``_VALID_TRANSITIONS`` set, and applies the side-effect
        delta atomically with the status flip. ``transition`` remains
        for callers that already serialise the side effect themselves.
        """
        with self._lock:
            obligation = self._items.get(obligation_id)
            if obligation is None:
                raise ObligationStateError(f"unknown obligation_id={obligation_id!r}")
            allowed = _VALID_TRANSITIONS.get(obligation.status, frozenset())
            if new_status not in allowed:
                raise ObligationStateError(
                    f"cannot transition {obligation_id} from {obligation.status} → {new_status}"
                )
            obligation.status = new_status
            entry: Dict[str, object] = {"event": event, "status": new_status.value}
            if meta:
                entry.update(meta)
            obligation.history.append(entry)
            return obligation

    def apply_event(
        self,
        obligation_id: str,
        event: Event,
        *,
        delta: Optional[Callable[[Obligation], None]] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> Obligation:
        """Atomic state mutation chokepoint.

        1. Look up the :class:`TransitionRule` for ``(current_status, event)``;
        2. Raise :class:`ObligationStateError` if no rule exists;
        3. Apply ``delta(obligation)`` (if provided) AND flip ``status``
           AND append the history record — all under the same lock so
           a partial mutation is impossible. M1 race window closes here.

        ``record_close`` / ``record_decompose`` / etc. become one-line
        wrappers that pass an event-specific ``delta`` lambda.
        """
        with self._lock:
            obligation = self._items.get(obligation_id)
            if obligation is None:
                raise ObligationStateError(f"unknown obligation_id={obligation_id!r}")
            rule = _lookup_transition(obligation.status, event)
            if rule is None:
                raise ObligationStateError(
                    f"no transition for {obligation_id}: {obligation.status} -[{event.value}]-> ?"
                )
            if delta is not None:
                delta(obligation)
            obligation.status = rule.to_state
            entry: Dict[str, object] = {"event": event.value, "status": rule.to_state.value}
            if meta:
                entry.update(meta)
            obligation.history.append(entry)
            return obligation

    def record_close(
        self,
        obligation_id: str,
        used_claim_ids: List[str],
        value: object,
    ) -> Obligation:
        """Close an obligation with the Γ result. ``closed_by`` /
        ``closed_value`` are written under the same lock as the status
        flip — M1 race window closes here.

        Picks the right transition event based on the current status:
        ``OPEN -> CLOSED`` is ``Event.CLOSE``; ``DECOMPOSED -> CLOSED``
        is ``Event.CHILDREN_CLOSED`` (parent's Γ re-evaluated against
        children's claims). Both deltas are identical — same
        ``closed_by`` / ``closed_value`` slot — so callers don't need
        to know which event fires.
        """
        used = list(used_claim_ids)

        def _delta(o: Obligation) -> None:
            o.closed_by = list(used)
            o.closed_value = value

        with self._lock:
            obligation = self._items.get(obligation_id)
            if obligation is None:
                raise ObligationStateError(f"unknown obligation_id={obligation_id!r}")
            current = obligation.status
        event = Event.CLOSE if current == ObligationStatus.OPEN else Event.CHILDREN_CLOSED
        return self.apply_event(
            obligation_id,
            event,
            delta=_delta,
            meta={"used_claim_ids": used, "value": value},
        )

    def record_challenge_open(
        self,
        obligation_id: str,
        challenge_id: str,
    ) -> Obligation:
        def _delta(o: Obligation) -> None:
            if challenge_id not in o.open_challenges:
                o.open_challenges.append(challenge_id)

        return self.apply_event(
            obligation_id,
            Event.CHALLENGE_OPEN,
            delta=_delta,
            meta={"challenge_id": challenge_id},
        )

    def record_challenge_discharged(
        self,
        obligation_id: str,
        challenge_id: str,
    ) -> Obligation:
        with self._lock:
            obligation = self._items.get(obligation_id)
            if obligation is None:
                raise ObligationStateError(f"unknown obligation_id={obligation_id!r}")
            if challenge_id in obligation.open_challenges:
                obligation.open_challenges.remove(challenge_id)
            remaining = list(obligation.open_challenges)
        if not remaining:
            return self.apply_event(
                obligation_id,
                Event.CHALLENGE_DISCHARGE,
                meta={"challenge_id": challenge_id},
            )
        # Self-loop on CHALLENGED: metadata-only, NOT a transition.
        with self._lock:
            obligation.history.append({
                "event": "challenge_discharged",
                "status": obligation.status.value,
                "challenge_id": challenge_id,
                "remaining": remaining,
            })
        return obligation

    def record_decompose(
        self,
        parent_id: str,
        child_ids: List[str],
        rule_id: str,
    ) -> Obligation:
        cids = list(child_ids)

        def _delta(o: Obligation) -> None:
            for cid in cids:
                if cid not in o.children_ids:
                    o.children_ids.append(cid)

        return self.apply_event(
            parent_id,
            Event.DECOMPOSE,
            delta=_delta,
            meta={"rule_id": rule_id, "child_ids": cids},
        )

    def record_retire(
        self,
        obligation_id: str,
        reason: str,
        *,
        replacement_id: Optional[str] = None,
        coverage_certificate: Optional[List[str]] = None,
    ) -> Obligation:
        meta: Dict[str, object] = {"reason": reason}
        if replacement_id is not None:
            meta["replacement_id"] = replacement_id
        if coverage_certificate is not None:
            meta["coverage_certificate"] = list(coverage_certificate)
        return self.apply_event(obligation_id, Event.RETIRE, meta=meta)

    # ----------------------------------------------------------- root specials

    def consume_wrong_kind_attempt(self) -> bool:
        """Single-use cap for the ``wrong_question_kind`` challenge.

        Returns True iff the caller may proceed; False if the cap has
        already been spent. Resetting requires a new session.
        """
        with self._lock:
            if self._wrong_kind_used:
                return False
            self._wrong_kind_used = True
            return True

    def replace_root(self, new_root_id: str) -> None:
        """Plant calls this after a wrong_question_kind discharge swaps
        the root. The old root must already be RETIRED."""
        with self._lock:
            if new_root_id not in self._items:
                raise ObligationStateError(f"unknown new_root_id={new_root_id!r}")
            new_root = self._items[new_root_id]
            new_root.is_root = True
            self._root_id = new_root_id

    # ----------------------------------------------------------- snapshot

    def snapshot(self) -> Dict[str, Any]:
        """Deep-copy every mutable field. Used by the plant to roll back
        a partial cross-store replacement on exception (B2 atomicity)."""
        with self._lock:
            return {
                "items": copy.deepcopy(self._items),
                "root_id": self._root_id,
                "wrong_kind_used": self._wrong_kind_used,
            }

    def restore(self, snap: Dict[str, Any]) -> None:
        with self._lock:
            self._items = copy.deepcopy(snap["items"])
            self._root_id = snap["root_id"]
            self._wrong_kind_used = snap["wrong_kind_used"]

    # ----------------------------------------------------------- internals

    def _mint_id(self, kind: ObligationKind) -> str:
        # Short prefix gives the agent a hint when reading traces; the
        # uuid suffix guarantees uniqueness across sessions if the trace
        # is replayed.
        return f"o_{kind.value[:3]}_{uuid.uuid4().hex[:8]}"
