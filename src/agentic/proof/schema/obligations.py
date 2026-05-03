"""Pydantic models for the per-tool composite shapes: ObligationSpec,
ClaimCandidate, Citation.

These are the structures wrapped by the tool args models in ``tools.py``.
Each carries a ``to_domain()`` helper that yields the canonical dict the
plant accepts (the plant performs its own deeper validation against
inventory / registries; pydantic only handles the syntactic layer).
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from agentic.proof.schema import predicates as _predicates  # noqa: F401
from agentic.proof.schema.predicates import (  # noqa: F401  (used for forward-ref resolution)
    AndPredicate,
    ContainsStringPredicate,
    DateComparePredicate,
    FieldEqualsPredicate,
    ListContainsPredicate,
    NumericComparePredicate,
    PredicateSpecField,
    RangeInPredicate,
    RegexMatchPredicate,
    SectionTitleContainsPredicate,
    TableCellContainsPredicate,
    TypeIsPredicate,
)
from agentic.proof.schema.scope import ScopeRefModel, ScoreSpecModel


_OBLIGATION_KIND = Literal[
    "exists", "count", "set", "forall", "negation", "argmax",
]
_UNIT_TYPE = Literal["file", "section"]
_POLARITY = Literal["positive", "negative"]
_TIE_POLICY = Literal["first", "all", "error"]
_DERIVED_BY = Literal[
    "root", "and_split", "scope_partition", "case_split",
    "map_over_domain", "user_constraint", "challenge_replacement",
]


class ObligationSpecModel(BaseModel):
    """The user-facing input that becomes :class:`ObligationSpec` in the
    plant. The schema mirrors the dict shape ``handle_obligation_create``
    accepts today; the plant continues to enforce semantics (universal
    predicates, applicable unit_type, score orderability for argmax,
    etc.).

    ``extra='forbid'`` so a typo'd key (e.g. ``unit`` instead of
    ``unit_type``) is rejected at the boundary with a typo-suggestion
    instead of silently dropping the value.
    """

    model_config = ConfigDict(extra="forbid")

    kind: _OBLIGATION_KIND
    scope: ScopeRefModel
    unit_type: _UNIT_TYPE
    predicate: PredicateSpecField
    score: Optional[ScoreSpecModel] = None
    polarity: _POLARITY = "positive"
    required: bool = True
    parent_id: Optional[str] = None
    derived_by: Optional[_DERIVED_BY] = None
    sealed_scope_override: bool = False
    tie_policy: _TIE_POLICY = "first"

    def to_domain(self) -> Dict[str, Any]:
        """Canonical dict the plant's ``_build_obligation_spec`` accepts.

        ``model_dump`` already produces a JSON-compatible structure; we
        only post-process to drop the optional ``derived_by`` when the
        caller did not supply one, since the plant has its own default.
        """
        out = self.model_dump(exclude_none=False)
        if self.derived_by is None:
            out.pop("derived_by", None)
        return out


class CitationModel(BaseModel):
    """One citation entry inside a claim candidate."""

    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    span: Optional[str] = None


class ClaimCandidateModel(BaseModel):
    """v1 LLM-facing claim shape. Both WitnessClaim and ScanClaim share
    this envelope; the plant routes on ``claim_type`` and applies the
    per-type validation rules (citation coverage, scan partition equality,
    value_map score verification, etc.)."""

    model_config = ConfigDict(extra="forbid")

    claim_type: Literal["WitnessClaim", "ScanClaim"]
    scope: ScopeRefModel
    unit_type: _UNIT_TYPE
    predicate: Optional[PredicateSpecField] = None
    score: Optional[ScoreSpecModel] = None
    positive_units: List[str]
    negative_units: List[str] = Field(default_factory=list)
    value_map: Dict[str, Any] = Field(default_factory=dict)
    citations: List[CitationModel] = Field(min_length=1)

    def to_domain(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=False)


ObligationSpecModel.model_rebuild()
ClaimCandidateModel.model_rebuild()


__all__ = [
    "ObligationSpecModel",
    "ClaimCandidateModel",
    "CitationModel",
]
