"""Insurance workbench routes.

Seven SSE endpoints powering the user-facing workbenches; the
frontend's Risk-Prediction page tabs four of them so they share a UI
shell while keeping their backend contracts independent:

* ``POST /insurance/compare/stream``          — SSE (BaseAgent)
* ``POST /insurance/exclusion-audit/stream``  — SSE (ProofAgent)
* ``POST /insurance/recommend/stream``        — SSE (BaseAgent)
* ``POST /insurance/claim-check/stream``      — SSE (BaseAgent)
* ``POST /insurance/policy-calc/stream``      — SSE (BaseAgent + code_run)
* ``POST /insurance/fraud-ppr/stream``        — SSE (PPR + LLM, no loop)
* ``POST /insurance/risk-predict/stream``     — SSE (GraphAgent + Sankey side-channel)

All routes go through ``get_current_user`` (analyst-accessible —
these are the analyst's main tools). Agent kickoff is async; the
streamer pumps the SSE bus directly.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from api.deps import get_current_user
from api.models import User
from api.runners.claim_runner import stream_claim_check
from api.runners.compare_runner import stream_compare
from api.runners.exclusion_runner import stream_exclusion_audit
from api.runners.fraud_ppr_runner import stream_fraud_ppr
from api.runners.policy_calc_runner import stream_policy_calc
from api.runners.recommend_runner import stream_recommend
from api.runners.risk_predict_runner import stream_risk_predict
from api.schemas.insurance import (
    ClaimCheckRequest,
    CompareRequest,
    ExclusionAuditRequest,
    FraudPPRRequest,
    PolicyCalcRequest,
    RecommendRequest,
    RiskPredictRequest,
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
            held_policies_file_ids=body.held_policies_file_ids,
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

    Used by the Policy Review workbench's "hidden risk" tab. The
    runner pre-fetches the PPR subgraph (no tool loop) and streams
    its analysis of the surrounding semantic neighborhood as token
    frames; passages cited via [^k] resolve to the same
    CitationDrawer the chat surface uses. The URL path must match
    the persisted trace artifacts under
    ``${STORAGE_PATH}/fraud_ppr/...``.
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


# ---------------------------------------------------------- risk predict


@router.post("/risk-predict/stream")
async def risk_predict_stream(
    body: RiskPredictRequest,
    request: Request,
    _user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Pre-issuance risk prediction (GraphAgent + PPR-anchored Sankey).

    Wraps the graph agent runner: drives a fixed PPR → neighbors →
    read pipeline behind the ``prompt.risk_predict`` system prompt and
    augments the agent's ``final`` SSE event with a ``risk_subgraph``
    payload (3-layer ``customer_fields → risk_factors →
    triggered_clauses`` adjacency the frontend Sankey consumes).
    """
    state = request.app.state
    graph_service = getattr(state, "graph_service", None)
    if graph_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="graph service not initialized",
        )
    return StreamingResponse(
        stream_risk_predict(
            file_id=body.file_id,
            customer=body.customer,
            scenario=body.scenario,
            agent=state.graph_agent,
            graph_service=graph_service,
            config=state.config,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
