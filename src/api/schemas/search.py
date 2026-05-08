"""Advanced clause-search DTOs.

Three axes — `granularity` × `channels` × `filters`, no LLM call.
Granularity controls the OUTPUT shape (the ranking is always
page-level under the hood); passage / table_row slice candidates from
the page hits using the inventory atom stores.
"""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


GranularityLiteral = Literal["page", "passage", "table_row"]
ChannelLiteral = Literal["semantic", "bm25", "graph_ppr", "regex"]


class SearchFilters(BaseModel):
    """All filters compose as AND. ``None`` / empty = no constraint."""

    file_ids: Optional[List[str]] = Field(default=None, max_length=64)
    page_range: Optional[List[int]] = Field(
        default=None,
        min_length=2, max_length=2,
        description="Inclusive [start, end] page-number gate, 1-based.",
    )
    suffix: Optional[List[str]] = Field(
        default=None, max_length=8,
        description="Filename-suffix whitelist (e.g. ['.pdf','.docx']).",
    )

    @model_validator(mode="after")
    def _validate_page_range(self) -> "SearchFilters":
        if self.page_range is not None:
            lo, hi = self.page_range
            if lo < 1 or hi < lo:
                raise ValueError("page_range must satisfy 1 <= start <= end")
        if self.file_ids is not None:
            if any(not f or not f.strip() for f in self.file_ids):
                raise ValueError("file_ids cannot contain empty strings")
        if self.suffix is not None:
            if any(not s.startswith(".") for s in self.suffix):
                raise ValueError("each suffix must start with '.', e.g. '.pdf'")
        return self


class SearchRequest(BaseModel):
    """Body for ``POST /search``."""

    query: str = Field(..., min_length=1, max_length=2000)
    granularity: GranularityLiteral = "page"
    channels: List[ChannelLiteral] = Field(
        ..., min_length=1, max_length=4,
        description="At least one channel must be selected.",
    )
    filters: Optional[SearchFilters] = None
    # RRF tuning — None = use admin-config default.
    rrf_k: Optional[int] = Field(default=None, ge=10, le=200)
    rrf_top_m: Optional[int] = Field(default=None, ge=5, le=200)
    top_n: Optional[int] = Field(
        default=None, ge=1, le=100,
        description="Final hits returned. Default = admin rerank_top_n.",
    )
    rerank: bool = Field(
        default=False,
        description="Run the rerank model over the fused candidates. Off by default — adds latency + cost.",
    )

    @model_validator(mode="after")
    def _no_dup_channels(self) -> "SearchRequest":
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must be unique")
        return self


class SearchHit(BaseModel):
    """One result row. Shape varies subtly by granularity:

    * ``granularity="page"``       — ``passage_id`` / ``table_row_id`` empty.
    * ``granularity="passage"``    — ``passage_id`` set.
    * ``granularity="table_row"``  — ``table_row_id`` set.
    """

    file_id: str
    page_id: str
    page_number: Optional[int] = None
    passage_id: Optional[str] = None
    table_row_id: Optional[str] = None
    score: float
    channel_scores: Dict[str, float] = Field(default_factory=dict)
    channels_hit: List[str] = Field(default_factory=list)
    snippet: str = ""
    rerank_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    granularity: GranularityLiteral
    channels_run: List[ChannelLiteral]
    filters_applied: Dict[str, object] = Field(default_factory=dict)
    hits: List[SearchHit]
    n_total: int
    n_returned: int
    timings_ms: Dict[str, int] = Field(default_factory=dict)
    used_rrf: bool
    used_rerank: bool
    rrf_k: Optional[int] = None
    rrf_top_m: Optional[int] = None
    # When True, the service overfetched the candidate set to
    # compensate for post-RRF page_range / suffix filtering. Compare
    # ``n_pre_filter`` to ``n_total`` to see how much the filter
    # pruned; a high prune ratio suggests widening ``rrf_top_m``.
    post_filter_overfetched: bool = False
    n_pre_filter: int = 0
