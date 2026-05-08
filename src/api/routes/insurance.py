"""Insurance workbench routes.

Six endpoints, three transports:

* ``POST /insurance/regulation-search``       — JSON (Tavily + LLM)
* ``POST /insurance/compare/stream``          — SSE (BaseAgent)
* ``POST /insurance/exclusion-audit/stream``  — SSE (ProofAgent)
* ``POST /insurance/recommend/stream``        — SSE (BaseAgent)
* ``POST /insurance/claim-check/stream``      — SSE (BaseAgent)
* ``POST /insurance/policy-calc/stream``      — SSE (BaseAgent + code_run)

All routes go through ``get_current_user`` (analyst-accessible —
these are the analyst's main tools). Heavy synchronous work (Tavily
+ LLM, agent loop kickoff) is wrapped in ``run_in_threadpool``.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from api.deps import get_current_user
from api.models import User
from api.runners.claim_runner import stream_claim_check
from api.runners.compare_runner import stream_compare
from api.runners.exclusion_runner import stream_exclusion_audit
from api.runners.fraud_ppr_runner import stream_fraud_ppr
from api.runners.policy_calc_runner import stream_policy_calc
from api.runners.recommend_runner import stream_recommend
from api.runners.regulation_runner import run_regulation_search
from api.schemas.insurance import (
    ClaimCheckRequest,
    CompareRequest,
    ExclusionAuditRequest,
    FraudPPRRequest,
    PolicyCalcRequest,
    RecommendRequest,
    RegulationSearchRequest,
    RegulationSearchResponse,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/insurance",
    tags=["insurance"],
    dependencies=[Depends(get_current_user)],
)


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


# ---------------------------------------------------------- regulation


@router.post(
    "/regulation-search",
    response_model=RegulationSearchResponse,
)
async def regulation_search(
    body: RegulationSearchRequest,
    request: Request,
) -> RegulationSearchResponse:
    """Single-call Tavily + LLM regulatory summary card."""
    state = request.app.state
    tavily = state.tavily_client
    if not tavily.available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tavily client not configured (TAVILY_API_KEY missing).",
        )
    return await run_in_threadpool(
        run_regulation_search,
        query=body.query,
        jurisdiction=body.jurisdiction,
        llm=state.rag_pipeline.llm,
        tavily=tavily,
        config=state.config,
        max_results=body.max_results,
        days=body.days,
    )


# ---------------------------------------------------------- compare


@router.post("/compare/stream")
async def compare_stream(
    body: CompareRequest,
    request: Request,
    _user: User = Depends(get_current_user),
) -> StreamingResponse:
    return StreamingResponse(
        stream_compare(
            file_ids=body.file_ids,
            properties=body.properties,
            agent=request.app.state.base_agent,
            config=request.app.state.config,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ---------------------------------------------------------- exclusion audit


@router.post("/exclusion-audit/stream")
async def exclusion_audit_stream(
    body: ExclusionAuditRequest,
    request: Request,
    _user: User = Depends(get_current_user),
) -> StreamingResponse:
    return StreamingResponse(
        stream_exclusion_audit(
            file_id=body.file_id,
            customer=body.customer,
            agent=request.app.state.proof_agent,
            config=request.app.state.config,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ---------------------------------------------------------- recommend


@router.post("/recommend/stream")
async def recommend_stream(
    body: RecommendRequest,
    request: Request,
    _user: User = Depends(get_current_user),
) -> StreamingResponse:
    return StreamingResponse(
        stream_recommend(
            customer=body.customer,
            agent=request.app.state.base_agent,
            config=request.app.state.config,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ---------------------------------------------------------- claim check


@router.post("/claim-check/stream")
async def claim_check_stream(
    body: ClaimCheckRequest,
    request: Request,
    _user: User = Depends(get_current_user),
) -> StreamingResponse:
    return StreamingResponse(
        stream_claim_check(
            file_ids=body.file_ids,
            event=body.event,
            agent=request.app.state.base_agent,
            config=request.app.state.config,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ---------------------------------------------------------- policy calc


@router.post("/policy-calc/stream")
async def policy_calc_stream(
    body: PolicyCalcRequest,
    request: Request,
    _user: User = Depends(get_current_user),
) -> StreamingResponse:
    return StreamingResponse(
        stream_policy_calc(
            file_id=body.file_id,
            policy_params=body.policy_params,
            calc_targets=body.calc_targets,
            agent=request.app.state.base_agent,
            config=request.app.state.config,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ---------------------------------------------------------- fraud-ppr


@router.post("/fraud-ppr/stream")
async def fraud_ppr_stream(
    body: FraudPPRRequest,
    request: Request,
    _user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Single PPR retrieval + streamed LLM analysis.

    Used by GraphPage's anti-fraud mode. The runner pre-fetches the
    PPR subgraph (so the model never calls a tool) and streams its
    triage as token frames; passages cited via [^k] resolve to the
    same CitationDrawer the chat surface uses.
    """
    state = request.app.state
    graph_service = getattr(state, "graph_service", None)
    if graph_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="graph service not initialized",
        )
    return StreamingResponse(
        stream_fraud_ppr(
            query=body.query,
            file_ids=body.file_ids,
            graph_service=graph_service,
            llm=state.rag_pipeline.llm,
            config=state.config,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
