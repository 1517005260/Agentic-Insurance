"""ChallengeStore — tracks open / discharged / rejected challenges.

A challenge is opened by ``obligation_challenge``; it sits in
``pending`` while plant.reconcile evaluates whether the mechanical
postcondition has been met. On postcondition met, status flips to
``discharged`` and the obligation goes back to OPEN (or RETIRED if a
replacement-link was supplied).
"""
import copy
import threading
import uuid
from typing import Any, Dict, Iterable, List, Optional

from agentic.proof.types import Challenge, RepairKind


class ChallengeStore:
    def __init__(self) -> None:
        self._items: Dict[str, Challenge] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._items)

    def get(self, challenge_id: str) -> Optional[Challenge]:
        return self._items.get(challenge_id)

    def all(self) -> List[Challenge]:
        return list(self._items.values())

    def open_for(self, obligation_id: str) -> List[Challenge]:
        return [
            c for c in self._items.values()
            if c.obligation_id == obligation_id and c.status == "pending"
        ]

    def insert(
        self,
        *,
        obligation_id: str,
        repair_kind: RepairKind,
        evidence_ids: List[str],
        reason: str,
        expected: Optional[Dict[str, object]] = None,
    ) -> Challenge:
        with self._lock:
            challenge_id = f"ch_{uuid.uuid4().hex[:10]}"
            challenge = Challenge(
                id=challenge_id,
                obligation_id=obligation_id,
                repair_kind=repair_kind,
                evidence_ids=list(evidence_ids),
                reason=reason,
                status="pending",
                expected=dict(expected or {}),
            )
            self._items[challenge_id] = challenge
            return challenge

    def discharge(self, challenge_id: str, *, meta: Optional[Dict] = None) -> Challenge:
        with self._lock:
            challenge = self._items.get(challenge_id)
            if challenge is None:
                raise KeyError(challenge_id)
            challenge.status = "discharged"
            if meta:
                challenge.expected.update(meta)
            return challenge

    def reject(self, challenge_id: str, *, reason: str) -> Challenge:
        with self._lock:
            challenge = self._items.get(challenge_id)
            if challenge is None:
                raise KeyError(challenge_id)
            challenge.status = "rejected"
            challenge.expected["rejection"] = reason
            return challenge

    # ----------------------------------------------------------- snapshot

    def snapshot(self) -> Dict[str, Any]:
        """Deep-copy of every mutable field. Used by the plant to roll
        back a partial cross-store replacement on exception (B2 atomicity)."""
        with self._lock:
            return {"items": copy.deepcopy(self._items)}

    def restore(self, snap: Dict[str, Any]) -> None:
        with self._lock:
            self._items = copy.deepcopy(snap["items"])
