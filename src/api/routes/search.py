"""Advanced clause-search route.

POST /search — no-LLM, channel-selective, RRF-fusing retrieval.
Implementation lives in :mod:`api.services.search`.

Filters use a `suffix` whitelist on the file's stored extension —
we look that up via the lifespan-built FileRecord cache (loaded
from the ``files`` table on demand). RBAC: any logged-in user.
"""
import logging
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_session
from api.models import FileRecord, User
from api.schemas.search import SearchRequest, SearchResponse
from api.services import search as search_svc


logger = logging.getLogger(__name__)


router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
    _user: User = Depends(get_current_user),
) -> SearchResponse:
    # Cross-check channel × granularity early so a bogus combo
    # surfaces as 422 (not as channels-returned-zero).
    reason = search_svc.validate_request(
        channels=body.channels, granularity=body.granularity
    )
    if reason is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=reason
        )

    # Resolve file_id → suffix for the suffix filter. Cheap one-shot
    # SELECT keyed on the file_ids the user filtered to (or all if
    # unfiltered). We don't bother caching across requests because
    # the table fits comfortably in memory and SQLite handles the
    # SELECT in <1ms.
    file_id_to_suffix: Dict[str, str] = {}
    if body.filters and body.filters.suffix:
        rows = (await db.execute(select(FileRecord.file_id, FileRecord.suffix))).all()
        for fid, suf in rows:
            file_id_to_suffix[fid] = (suf or "").lower()

    pipeline = request.app.state.rag_pipeline
    payload = await run_in_threadpool(
        search_svc.run_search,
        pipeline=pipeline,
        query=body.query,
        channels=body.channels,
        granularity=body.granularity,
        file_ids=(body.filters.file_ids if body.filters else None),
        page_range=(
            tuple(body.filters.page_range)
            if body.filters and body.filters.page_range
            else None
        ),
        suffixes=(body.filters.suffix if body.filters else None),
        rrf_k=body.rrf_k,
        rrf_top_m=body.rrf_top_m,
        top_n=body.top_n,
        rerank=body.rerank,
        file_id_to_suffix=file_id_to_suffix,
    )
    return SearchResponse(**payload)
