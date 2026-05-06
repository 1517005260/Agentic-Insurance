"""Budget — read-only snapshot of how much room the agent loop has left.

Both ``promote_candidate_gap`` and ``try_finalize`` consult the same
budget, so the type lives in its own tiny module to avoid an import
cycle.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Budget:
    remaining_steps: int
    max_loops: int = 0
    exhausted_kind: Optional[str] = None

    def has_room(self) -> bool:
        return self.remaining_steps > 0 and self.exhausted_kind is None
