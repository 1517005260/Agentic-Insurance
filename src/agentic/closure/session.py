"""Per-run container threaded through the proof tools.

One ``ProofSession`` exists for the lifetime of one ``ProofAgent.run``.
It owns the obligation list, the validated claims, the candidate
gaps, the observations the agent has produced, and the plant. Tools
read and mutate the session through narrow, audited helpers.
"""

from dataclasses import dataclass, field
from typing import Optional

from agentic.agent.evidence_bank import EvidenceBank, Observation
from agentic.closure.budget import Budget
from agentic.closure.candidate_gap import CandidateGap
from agentic.closure.claims import Claim
from agentic.closure.inventory import Inventory
from agentic.closure.obligation import Obligation
from agentic.closure.plant import Plant


@dataclass
class ProofSession:
    inventory: Inventory
    budget: Budget
    plant: Optional[Plant] = None
    obligations: list[Obligation] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    candidate_gaps: list[CandidateGap] = field(default_factory=list)
    # Observation storage delegates to the shared EvidenceBank so the proof
    # path and the agent loop account for evidence the same way.
    _bank: EvidenceBank = field(default_factory=EvidenceBank)
    promoted_count: int = 0

    @property
    def observations(self) -> dict[str, Observation]:
        return self._bank._observations

    def append_observation(self, observation: Observation) -> None:
        self._bank.ingest(
            observation.id, observation.tool_name, observation.text, ()
        )

    # ObservationStore Protocol — Plant calls this to verify citations.
    def get_text(self, observation_id: str) -> Optional[str]:
        return self._bank.get_text(observation_id)

    def get_tool_name(self, observation_id: str) -> Optional[str]:
        return self._bank.get_tool_name(observation_id)

    def find_obligation(self, obligation_id: str) -> Optional[Obligation]:
        for o in self.obligations:
            if o.id == obligation_id:
                return o
        return None

    def find_claim(self, claim_id: str) -> Optional[Claim]:
        for c in self.claims:
            if c.id == claim_id:
                return c
        return None

    @classmethod
    def build(cls, *, inventory: Inventory, budget: Budget) -> "ProofSession":
        session = cls(inventory=inventory, budget=budget)
        session.plant = Plant(inventory=inventory, observations=session)
        return session
