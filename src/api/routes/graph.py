"""Knowledge-graph read endpoints.

Five routes, all behind ``get_current_user`` (analyst can hit them —
graph is core analyst tooling, not admin-only):

* ``GET  /graph/overview``                 first-paint counts + central entities
* ``GET  /graph/seed?q=&top_k=``           fuzzy entity search (top bar)
* ``GET  /graph/expand?node_id=&...``      double-click → 1-3 hop neighborhood
* ``GET  /graph/nodes/{hash_id}``          hover card (no hash_id in body)
* ``POST /graph/ppr-subgraph``             RAG PPR drawer (re-runs PPR)

All endpoints reuse the lifespan-built ``GraphService`` singleton on
``request.app.state.graph_service`` so the igraph + faiss artifacts
load exactly once per process.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool

from api.deps import get_current_user
from api.models import User
from api.schemas.graph import (
    GraphOverviewResponse,
    GraphSubgraphResponse,
    NodeDetailResponse,
    PPRSubgraphRequest,
    PPRSubgraphResponse,
    SeedHit,
)
from api.services.graph_service import GraphService, GraphServiceUnavailable


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/graph",
    tags=["graph"],
    dependencies=[Depends(get_current_user)],
)


def _service(request: Request) -> GraphService:
    """Pull the lifespan-built GraphService off ``app.state``."""
    svc = getattr(request.app.state, "graph_service", None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="graph service not initialized",
        )
    return svc


@router.get("/overview", response_model=GraphOverviewResponse)
async def get_overview(request: Request) -> GraphOverviewResponse:
    svc = _service(request)
    try:
        payload = svc.overview()
    except GraphServiceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return GraphOverviewResponse(**payload)


@router.get("/seed", response_model=List[SeedHit])
async def search_seed(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200, description="Entity name to fuzzy-match"),
    top_k: int = Query(10, ge=1, le=50),
) -> List[SeedHit]:
    svc = _service(request)
    try:
        # Embedding call inside seed_search → push to threadpool so we
        # don't block the event loop (chat SSE shares this loop).
        hits = await run_in_threadpool(svc.seed_search, q, top_k)
    except GraphServiceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return [SeedHit(**hit) for hit in hits]


@router.get("/sample", response_model=GraphSubgraphResponse)
async def sample_graph(
    request: Request,
    n: int = Query(100, ge=10, le=500),
) -> GraphSubgraphResponse:
    """Random vertex sample (entity-first) — first-paint canvas filler.

    Cached per-n for the process lifetime so mode switches in the
    GraphPage don't reshuffle the underlying canvas under the user.
    """
    svc = _service(request)
    try:
        payload = await run_in_threadpool(svc.sample, n)
    except GraphServiceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return GraphSubgraphResponse(**payload)


@router.get("/expand", response_model=GraphSubgraphResponse)
async def expand_node(
    request: Request,
    node_id: str = Query(
        ..., min_length=1, max_length=128,
        description="hash_id of the seed vertex",
    ),
    hops: int = Query(1, ge=1, le=3),
    top_k: int = Query(50, ge=1, le=200),
    vertex_type: str = Query(
        "both",
        pattern="^(entity|passage|both)$",
        description="Filter applied to non-seed nodes only.",
    ),
    file_ids: Optional[List[str]] = Query(
        None,
        max_length=50,
        description="Optional: prune passage vertices to these file_ids (max 50).",
    ),
) -> GraphSubgraphResponse:
    svc = _service(request)
    try:
        # BFS + induced-edge walk are CPU-only but on a large graph
        # could spike past 50ms; offload so the event loop stays free
        # for concurrent SSE requests.
        payload = await run_in_threadpool(
            svc.expand,
            node_id,
            hops=hops,
            top_k=top_k,
            vertex_type=vertex_type,
            file_ids=file_ids,
        )
    except GraphServiceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return GraphSubgraphResponse(**payload)


@router.get("/nodes/{hash_id}", response_model=NodeDetailResponse)
async def get_node_detail(
    hash_id: str,
    request: Request,
) -> NodeDetailResponse:
    """Hover-card payload. Response body deliberately omits ``hash_id`` —
    the URL already carries it and the frontend renders pure prose."""
    # 128 keeps headroom over the 40-char ``namespace-md5`` real id;
    # rejects oversized request amplification.
    if len(hash_id) > 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="hash_id length exceeds 128",
        )
    svc = _service(request)
    try:
        # node_detail can trigger first-call cluster-cache compute
        # (disambig.get_clusters may write clusters.json). Threadpool
        # keeps the event loop responsive even on the worst-case first
        # hover after a corpus rebuild.
        payload = await run_in_threadpool(svc.node_detail, hash_id)
    except GraphServiceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return NodeDetailResponse(**payload)


@router.post("/ppr-subgraph", response_model=PPRSubgraphResponse)
async def ppr_subgraph(
    body: PPRSubgraphRequest,
    request: Request,
    _user: User = Depends(get_current_user),
) -> PPRSubgraphResponse:
    """Re-run the PPR channel for ``body.query`` and return the visualizable
    subgraph (seeds + actived entities + passages + induced edges).

    ~300-500ms on a warm channel; the RAG-PPR drawer in the chat UI
    calls this when the user clicks "show subgraph" on a graph_ppr hit.
    """
    svc = _service(request)
    try:
        # PPR is the most expensive call (~300-500ms warm); always
        # threadpool — blocking the event loop here would freeze every
        # in-flight SSE chat stream.
        payload = await run_in_threadpool(
            svc.ppr_subgraph, body.query, file_ids=body.file_ids
        )
    except GraphServiceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return PPRSubgraphResponse(**payload)
