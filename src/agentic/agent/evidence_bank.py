"""Evidence accounting for the agent loop.

A run accumulates tool observations; the loop-guard and (optionally) the
proof session need to know how much NEW evidence each observation brought
in, not just how many tool calls fired. ``EvidenceBank`` tracks the set of
evidence ids seen so far and reports, per ingest, how many were novel — the
signal a search-novelty loop-guard trips on.

Dependency-free by design: the unit it counts is an opaque id (a page
``file_id``, a passage hash, an observation id) supplied by the caller.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set


@dataclass
class Observation:
    id: str
    tool_name: str
    text: str


class EvidenceBank:
    """Records observations and tracks the running set of seen evidence ids."""

    def __init__(self) -> None:
        self._observations: Dict[str, Observation] = {}
        self._seen_ids: Set[str] = set()

    def ingest(
        self, obs_id: str, tool_name: str, text: str, ids: Iterable[str]
    ) -> int:
        """Store an observation; return how many of ``ids`` are new.

        ``n_new`` counts ids not previously seen across the whole run, then
        folds them into the seen set. A second observation that only re-cites
        already-known evidence returns 0 — the loop-guard's novelty signal.
        """
        self._observations[obs_id] = Observation(
            id=obs_id, tool_name=tool_name, text=text
        )
        n_new = 0
        for i in ids:
            if i not in self._seen_ids:
                self._seen_ids.add(i)
                n_new += 1
        return n_new

    def get_text(self, obs_id: str) -> Optional[str]:
        obs = self._observations.get(obs_id)
        return obs.text if obs is not None else None

    def get_tool_name(self, obs_id: str) -> Optional[str]:
        obs = self._observations.get(obs_id)
        return obs.tool_name if obs is not None else None

    @property
    def seen_count(self) -> int:
        return len(self._seen_ids)

    def __len__(self) -> int:
        return len(self._observations)
