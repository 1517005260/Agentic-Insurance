"""DTOs for the insurance workbenches.

One file for all five workbenches because their request / response
shapes are tiny and forking per-workbench would just multiply
imports. The order below mirrors the runner files:

* :class:`RegulationSearchRequest` / :class:`RegulationSearchResponse`
  — non-streaming Tavily + LLM summarize.
* :class:`CompareRequest` — N×M product comparison (BaseAgent SSE).
* :class:`ExclusionAuditRequest` — single-product underwriting audit
  (ProofAgent SSE).
* :class:`RecommendRequest` — customer-profile product recommendation
  (BaseAgent SSE).
* :class:`ClaimCheckRequest` — multi-product claim-coverage analysis
  (BaseAgent SSE).
* :class:`PolicyCalcRequest` — actuarial calculation workbench
  (BaseAgent + code_run SSE).

All SSE-streaming endpoints use the same SSE protocol the chat surface
uses; the request schemas only define the request body. Final-payload
shapes are documented at the runner level — they ride the ``final``
SSE event and the optional ``result_future`` rather than living here.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------- regulation


JurisdictionLiteral = Literal["hk", "cn", "both"]


class RegulationSearchRequest(BaseModel):
    """Non-streaming request body for ``POST /insurance/regulation-search``."""

    query: str = Field(..., min_length=1, max_length=2000)
    jurisdiction: JurisdictionLiteral = "both"
    max_results: Optional[int] = Field(default=None, ge=3, le=20)
    days: Optional[int] = Field(
        default=None,
        ge=1,
        le=3650,
        description=(
            "Restrict results to the last N days. Tavily honors this only "
            "when ``topic='news'``; for general regulation queries it is "
            "advisory."
        ),
    )


class RegulationSource(BaseModel):
    sup: int
    title: str
    url: str
    snippet: str
    score: float
    published_date: Optional[str] = None


class RegulationSearchResponse(BaseModel):
    answer: str
    summary_chars: int
    sources: List[RegulationSource]
    n_results: int
    n_cited: int
    jurisdiction: JurisdictionLiteral
    used_include_domains: Optional[List[str]] = None
    search_query: str


# ---------------------------------------------------------------- compare


class CompareRequest(BaseModel):
    """``POST /insurance/compare/stream`` body."""

    file_ids: List[str] = Field(
        ...,
        min_length=2,
        max_length=8,
        description="Two-to-eight indexed product file_ids to compare.",
    )
    properties: List[str] = Field(
        ...,
        min_length=1,
        max_length=12,
        description=(
            "Comparison dimensions; each becomes a column. Free-form "
            "Chinese / English (e.g. '等待期', 'waiting period', '免责')."
        ),
    )

    @model_validator(mode="after")
    def _no_dup_ids(self) -> "CompareRequest":
        if len(set(self.file_ids)) != len(self.file_ids):
            raise ValueError("file_ids must be unique")
        if len(set(self.properties)) != len(self.properties):
            raise ValueError("properties must be unique")
        return self


# ---------------------------------------------------------------- exclusion audit


class CustomerProfile(BaseModel):
    """Shared customer-profile payload (exclusion audit + recommend)."""

    age: int = Field(..., ge=0, le=120)
    gender: Literal["M", "F", "X"]
    occupation: str = Field(..., min_length=1, max_length=80)
    occupation_risk: Optional[Literal["low", "med", "high"]] = None
    health_history: List[str] = Field(default_factory=list, max_length=20)
    family_history: List[str] = Field(default_factory=list, max_length=20)
    budget_annual: Optional[int] = Field(default=None, ge=0, le=10_000_000)
    goal: Optional[str] = Field(default=None, max_length=80)
    notes: Optional[str] = Field(default=None, max_length=500)


class ExclusionAuditRequest(BaseModel):
    """``POST /insurance/exclusion-audit/stream`` body."""

    file_id: str = Field(..., min_length=1, max_length=128)
    customer: CustomerProfile


# ---------------------------------------------------------------- recommend


class RecommendRequest(BaseModel):
    """``POST /insurance/recommend/stream`` body."""

    customer: CustomerProfile


# ---------------------------------------------------------------- claim check


class ClaimEvent(BaseModel):
    type: str = Field(..., min_length=1, max_length=80)
    date: str = Field(..., min_length=1, max_length=40, description="ISO date or human-readable.")
    location: Optional[str] = Field(default=None, max_length=120)
    description: str = Field(..., min_length=1, max_length=2000)
    amount: Optional[float] = Field(default=None, ge=0)


class ClaimCheckRequest(BaseModel):
    """``POST /insurance/claim-check/stream`` body."""

    file_ids: List[str] = Field(..., min_length=1, max_length=8)
    event: ClaimEvent


# ---------------------------------------------------------------- policy calc


class PolicyParams(BaseModel):
    age_at_issue: int = Field(..., ge=0, le=100)
    gender: Literal["M", "F", "X"]
    premium_mode: Literal["annual", "monthly", "single"] = "annual"
    premium_amount: float = Field(..., ge=0)
    term_years: int = Field(..., ge=1, le=80)
    sum_assured: float = Field(..., ge=0)
    currency: str = Field(default="HKD", max_length=8)
    # Optional projection knobs — runners pass them through to the
    # agent prompt as additional structured context.
    target_age: Optional[int] = Field(default=None, ge=0, le=120)
    target_year: Optional[int] = Field(default=None, ge=0, le=100)


class PolicyCalcRequest(BaseModel):
    """``POST /insurance/policy-calc/stream`` body.

    ``calc_targets`` is intentionally free-form: the agent has the
    actuarial vocabulary baked into ``prompt.policy_calc`` and can
    interpret natural-language asks like "Embedded Value 演示" /
    "premium-financing breakeven assuming 5% loan rate" without us
    having to enumerate every variant in a Literal. The frontend
    surfaces a chip set of common targets + an "其他" free text.
    """

    file_id: str = Field(..., min_length=1, max_length=128)
    policy_params: PolicyParams
    calc_targets: List[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description=(
            "Up to 6 free-text calc targets. Each ≤ 300 chars. "
            "Common: cash value by year, surrender value at year N, "
            "IRR to age 65, break-even year, embedded value (VIF + ANAV), "
            "NBV margin, premium-financing net IRR, IRR breakdown "
            "(guaranteed vs non-guaranteed)."
        ),
    )

    @model_validator(mode="after")
    def _validate_calc_targets(self) -> "PolicyCalcRequest":
        cleaned = []
        for t in self.calc_targets:
            t = (t or "").strip()
            if not t:
                raise ValueError("calc_targets items cannot be empty")
            if len(t) > 300:
                raise ValueError("each calc_targets item must be ≤ 300 chars")
            cleaned.append(t)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("calc_targets must be unique")
        # Pydantic v2 models are mutable by default; plain attribute
        # assignment is the conventional way to write back the
        # normalized list.
        self.calc_targets = cleaned
        return self


# ---------------------------------------------------------------- fraud-ppr


class FraudPPRRequest(BaseModel):
    """``POST /insurance/fraud-ppr/stream`` body.

    Single PPR retrieval + single LLM analysis — no agent loop, no
    tool calls. The runner pre-fetches the subgraph server-side, so
    the request only carries the question (+ optional file_id scope).
    """

    query: str = Field(..., min_length=1, max_length=2000)
    file_ids: Optional[List[str]] = Field(
        default=None,
        max_length=50,
        description="Optional: prune passages to these file_ids only (max 50).",
    )
