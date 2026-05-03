"""Virtualised ``map_over_domain`` bookkeeping.

A ``map_over_domain`` decomposition declares that the parent obligation
holds iff each unit ``u`` in ``inventory.units(scope, unit_type)`` has
a witness for the parent's predicate (or a per-unit value for argmax).
We do NOT eagerly create N child obligations — that explodes the LLM's
context list. Instead, the plant maintains a DomainMap per parent that
records which units are "covered" by closing claims, returning a
``k/N + cursor`` summary in gate.diagnose.

Direct ``obligation_create`` for a child unit canonicalises onto the
DomainMap; if the LLM creates a separate obligation with the same
canonical key, the plant reuses the materialised id (or rejects with
``duplicate_virtual_child`` when conflicts are unresolvable).
"""
import copy
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agentic.proof.types import (
    ObligationKind,
    PredicateSpec,
    ScoreSpec,
)
from agentic.proof.predicate import serialize_spec


def canonical_key(
    parent_id: str,
    unit_id: str,
    kind: ObligationKind,
    predicate: Optional[PredicateSpec],
    score: Optional[ScoreSpec],
    polarity: str,
) -> str:
    """Stable id used to dedupe materialised vs virtual children.

    Includes parent_id (so different parents' DomainMaps don't collide),
    unit_id, kind, the canonical predicate hash (if any), and polarity.
    """
    pred_hash = serialize_spec(predicate) if predicate is not None else "_"
    score_hash = f"{score.name}({score.args})" if score is not None else "_"
    return f"{parent_id}|{unit_id}|{kind.value}|{pred_hash}|{score_hash}|{polarity}"


@dataclass
class DomainMap:
    parent_id: str
    domain_units: List[str]
    materialised_children: Dict[str, str] = field(default_factory=dict)  # unit_id -> obligation_id
    closed_units: set = field(default_factory=set)
    canonical_keys: Dict[str, str] = field(default_factory=dict)         # canonical_key -> unit_id

    def k_of_n(self) -> Tuple[int, int]:
        return len(self.closed_units), len(self.domain_units)

    def cursor(self, limit: int = 5) -> List[str]:
        """Next ``limit`` units that are not yet closed (and prefer
        unmaterialised first so the LLM is nudged to materialise +
        prove the next unit)."""
        out: List[str] = []
        # First: unmaterialised + not-yet-closed.
        for u in self.domain_units:
            if u in self.materialised_children or u in self.closed_units:
                continue
            out.append(u)
            if len(out) >= limit:
                return out
        # Then: materialised but still open.
        for u in self.domain_units:
            if u in self.closed_units:
                continue
            if u in self.materialised_children and u not in out:
                out.append(u)
                if len(out) >= limit:
                    break
        return out


class DomainMapStore:
    def __init__(self) -> None:
        self._maps: Dict[str, DomainMap] = {}
        self._lock = threading.Lock()

    def __contains__(self, parent_id: str) -> bool:
        return parent_id in self._maps

    def get(self, parent_id: str) -> Optional[DomainMap]:
        return self._maps.get(parent_id)

    def install(self, parent_id: str, domain_units: List[str]) -> DomainMap:
        with self._lock:
            if parent_id in self._maps:
                return self._maps[parent_id]
            dm = DomainMap(parent_id=parent_id, domain_units=list(domain_units))
            self._maps[parent_id] = dm
            return dm

    def materialise(
        self,
        parent_id: str,
        unit_id: str,
        obligation_id: str,
        canonical_key_str: str,
    ) -> None:
        """Register a virtual child as materialised. Refuses to overwrite
        an existing mapping — only one obligation may represent a unit
        in the parent's DomainMap, regardless of canonical key. The
        plant must reject duplicate materialisation requests upstream;
        if the call slips through here we raise instead of silently
        replacing the existing child.
        """
        with self._lock:
            dm = self._maps.get(parent_id)
            if dm is None:
                raise KeyError(parent_id)
            if unit_id in dm.materialised_children:
                if dm.materialised_children[unit_id] != obligation_id:
                    raise ValueError(
                        f"DomainMap unit {unit_id!r} already materialised as "
                        f"{dm.materialised_children[unit_id]!r}; refusing to "
                        f"overwrite with {obligation_id!r}"
                    )
                return
            dm.materialised_children[unit_id] = obligation_id
            dm.canonical_keys[canonical_key_str] = unit_id

    def replace_materialised(
        self,
        parent_id: str,
        unit_id: str,
        old_id: str,
        new_id: str,
        new_canonical_key: str,
    ) -> None:
        """Atomically swap the obligation holding ``unit_id``'s slot.
        B2 fix: a challenge_replacement targeting an already-materialised
        DomainMap child needs to (a) re-point ``materialised_children``
        from ``old_id`` to ``new_id`` and (b) drop the old canonical_key
        and install the new one — predicate_mismatch may rewrite the
        canonical key. Both maps must move under the same lock so a
        partial write never leaves stale canonical_keys pointing at the
        retired obligation.
        """
        with self._lock:
            dm = self._maps.get(parent_id)
            if dm is None:
                raise KeyError(parent_id)
            existing = dm.materialised_children.get(unit_id)
            if existing != old_id:
                raise ValueError(
                    f"DomainMap unit {unit_id!r} holder is "
                    f"{existing!r}, not {old_id!r}; refusing to replace"
                )
            stale_keys = [k for k, u in dm.canonical_keys.items() if u == unit_id]
            for k in stale_keys:
                del dm.canonical_keys[k]
            dm.materialised_children[unit_id] = new_id
            dm.canonical_keys[new_canonical_key] = unit_id

    def find_materialised(self, obligation_id: str) -> Optional[Tuple[str, str]]:
        """Reverse-lookup: return ``(parent_id, unit_id)`` for the slot
        currently held by ``obligation_id``, or ``None`` if nowhere.
        Used by the cross-store replacement transaction so the caller
        does not need to remember which DomainMap a CHALLENGED child
        sits in."""
        for parent_id, dm in self._maps.items():
            for unit_id, obl_id in dm.materialised_children.items():
                if obl_id == obligation_id:
                    return parent_id, unit_id
        return None

    # ----------------------------------------------------------- snapshot

    def snapshot(self) -> Dict[str, Any]:
        """Deep-copy of every mutable field. Used by the plant to roll
        back a partial cross-store replacement on exception (B2 atomicity)."""
        with self._lock:
            return {"maps": copy.deepcopy(self._maps)}

    def restore(self, snap: Dict[str, Any]) -> None:
        """Replace internal state from a snapshot. Caller guarantees no
        concurrent writer holds the lock — the ProofAgent loop is
        single-threaded; this is defensive."""
        with self._lock:
            self._maps = copy.deepcopy(snap["maps"])

    def lookup_by_key(self, parent_id: str, canonical_key_str: str) -> Optional[str]:
        dm = self._maps.get(parent_id)
        if dm is None:
            return None
        unit_id = dm.canonical_keys.get(canonical_key_str)
        if unit_id is None:
            return None
        return dm.materialised_children.get(unit_id)

    def mark_closed(self, parent_id: str, unit_id: str) -> bool:
        """True iff this call transitioned the unit to closed."""
        with self._lock:
            dm = self._maps.get(parent_id)
            if dm is None:
                return False
            if unit_id in dm.closed_units:
                return False
            if unit_id not in dm.domain_units:
                return False
            dm.closed_units.add(unit_id)
            return True

    def all_closed(self, parent_id: str) -> bool:
        dm = self._maps.get(parent_id)
        if dm is None:
            return False
        return len(dm.closed_units) == len(dm.domain_units)
