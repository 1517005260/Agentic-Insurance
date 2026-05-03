"""Per-tool args wrappers. One BaseModel per LLM-facing tool.

Each model exposes ``to_domain()`` which returns the kwargs dict the
matching ``Plant.handle_*`` method expects today. The plant continues
to perform its own structural validation against inventory + registries
— pydantic only catches the cheap syntactic mistakes before the call
ever reaches the plant."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from agentic.proof.schema.obligations import (
    ClaimCandidateModel,
    ObligationSpecModel,
)


_REPAIR_KIND = Literal[
    "scope_too_narrow", "scope_too_broad", "predicate_mismatch",
    "missing_subobligation", "wrong_question_kind",
]
_DECOMPOSITION_RULE = Literal[
    "and_split", "scope_partition", "case_split", "map_over_domain",
]


class ObligationCreateArgsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: ObligationSpecModel
    discharges_challenge: Optional[str] = None

    def to_domain(self) -> Dict[str, Any]:
        return {
            "spec_payload": self.spec.to_domain(),
            "discharges_challenge": self.discharges_challenge,
        }


class ObligationDecomposeArgsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent_id: str = Field(min_length=1)
    rule_id: _DECOMPOSITION_RULE
    child_specs: List[Dict[str, Any]] = Field(default_factory=list)
    discharges_challenge: Optional[str] = None

    def to_domain(self) -> Dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "rule_id": self.rule_id,
            "child_specs": list(self.child_specs),
            "discharges_challenge": self.discharges_challenge,
        }


class ObligationChallengeArgsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obligation_id: str = Field(min_length=1)
    repair_kind: _REPAIR_KIND
    evidence_ids: List[str] = Field(default_factory=list)
    reason: str

    def to_domain(self) -> Dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "repair_kind": self.repair_kind,
            "evidence_ids": list(self.evidence_ids),
            "reason": self.reason,
        }


class EvidenceIngestArgsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(min_length=1)
    claim_candidate: ClaimCandidateModel

    def to_domain(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "claim_candidate": self.claim_candidate.to_domain(),
        }


class AnswerFinalizeArgsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_text: str
    cited_claim_ids: List[str] = Field(default_factory=list)

    def to_domain(self) -> Dict[str, Any]:
        return {
            "draft_text": self.draft_text,
            "cited_claim_ids": list(self.cited_claim_ids),
        }


# Forward-ref resolution: AndPredicate / PredicateSpecField are
# defined via string forward-refs, so every model reachable from the
# tools wrappers must be rebuilt once the inner predicate union is
# fully assembled.
ObligationCreateArgsModel.model_rebuild()
ObligationDecomposeArgsModel.model_rebuild()
ObligationChallengeArgsModel.model_rebuild()
EvidenceIngestArgsModel.model_rebuild()
AnswerFinalizeArgsModel.model_rebuild()


__all__ = [
    "ObligationCreateArgsModel",
    "ObligationDecomposeArgsModel",
    "ObligationChallengeArgsModel",
    "EvidenceIngestArgsModel",
    "AnswerFinalizeArgsModel",
]
