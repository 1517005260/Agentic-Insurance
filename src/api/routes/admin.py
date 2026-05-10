"""Admin config-center routes.

Surface (all admin-only):

* ``GET    /admin/config``           snapshot + schema (one round-trip
                                     for the table view)
* ``GET    /admin/config/schema``    schema only
* ``PATCH  /admin/config``           batch save; all-or-nothing validation
* ``DELETE /admin/config/{key}``     reset one key to schema default

The PATCH route is the single "Save" button in the admin UI — the
frontend posts the full ``{key: value}`` form once. We validate every
entry before writing any row, so an invalid value never half-applies.

Each successful PATCH/DELETE writes one ``audit_log`` entry per key
with the old/new pair so a postmortem can replay any tweak.

The store mutates :attr:`fastapi.Request.app.state.config` in place;
the next handler that reads ``request.app.state.config.get(...)``
sees the new value. In-flight handlers keep the value they already
materialized.
"""
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_session, require_admin
from api.models import User
from api.schemas.admin import (
    ConfigEntrySchema,
    ConfigPatchRequest,
    ConfigPatchResponse,
    ConfigSnapshotResponse,
)
from config.config_store import ConfigStore
from config.config_store.schema import CONFIG_ENTRIES_BY_KEY


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/admin/config",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


# Sibling router for non-config admin actions. Kept in this module so
# the file stays the single home for /admin surfaces that are not
# user-management; the prefix differs ("/admin" vs "/admin/config") so
# we cannot reuse the same APIRouter instance.
admin_actions_router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


@admin_actions_router.post(
    "/refresh-indexes",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def refresh_indexes(request: Request) -> Response:
    """Force-reload PageStore / InventoryStore / GraphPPRChannel from disk.

    Normally ingest tasks call the same hook themselves on every
    successful parse / reingest / delete, so an admin will only reach
    for this when something wrote to ``local_storage`` out of band
    (manual scripts, restored backups). Returns 204 — caller may verify
    via the next /graph/sample or /files request.

    503 when the singletons aren't wired (lifespan didn't run; rare
    outside test clients without lifespan).
    """
    refresh = getattr(request.app.state, "refresh_indexes", None)
    if refresh is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="refresh_indexes hook not initialized",
        )
    # Hold the global ingest lock for the duration of the refresh —
    # otherwise a concurrent ingest writing to the very stores we're
    # reloading could leave the in-memory snapshot pointing at a
    # mid-write parquet (faiss + parquet are written non-atomically
    # within an ``add()`` block, see EmbeddingStore.save). With the
    # lock, the operator's manual refresh waits for any in-flight
    # ingest to finish, then re-reads a consistent on-disk state.
    import asyncio

    from api.services.files import INGEST_LOCK

    async with INGEST_LOCK:
        await asyncio.get_running_loop().run_in_executor(None, refresh)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _store(request: Request) -> ConfigStore:
    """Pull the lifespan-built ConfigStore off ``app.state``."""
    store = getattr(request.app.state, "config", None)
    if store is None:
        # Lifespan hasn't run (test client without lifespan or a
        # boot ordering bug). Fail loudly rather than silently
        # serving defaults from a fresh store.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="config store not initialized",
        )
    return store


def _schema_payload() -> list[ConfigEntrySchema]:
    """Serialize the registered entries in declaration order."""
    return [ConfigEntrySchema(**e.to_public_dict()) for e in CONFIG_ENTRIES_BY_KEY.values()]


@router.get("", response_model=ConfigSnapshotResponse)
async def get_config(request: Request) -> ConfigSnapshotResponse:
    store = _store(request)
    return ConfigSnapshotResponse(
        snapshot=store.snapshot(),
        schema=_schema_payload(),
    )


@router.get("/schema", response_model=list[ConfigEntrySchema])
async def get_schema() -> list[ConfigEntrySchema]:
    return _schema_payload()


@router.patch("", response_model=ConfigPatchResponse)
async def patch_config(
    body: ConfigPatchRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(require_admin),
) -> ConfigPatchResponse:
    store = _store(request)
    if not body.updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="updates must contain at least one key",
        )
    try:
        # ConfigStore.patch is responsible for the whole transaction:
        # validation → app_config UPSERT → audit_log INSERT → commit
        # → in-memory snapshot mutation. Centralizing it there means
        # a SQL failure can never leave the in-memory store ahead of
        # the persisted state.
        diffs = await store.patch(db, updates=body.updates, user_id=user.id)
    except ValueError as exc:
        # Schema validation failure — surface as 422 with the offending
        # key in the body so the admin UI can highlight that row.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return ConfigPatchResponse(diffs=diffs, snapshot=store.snapshot())


@router.delete(
    "/{key}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def reset_config(
    key: str,
    request: Request,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(require_admin),
) -> Response:
    store = _store(request)
    try:
        await store.reset(db, key=key, user_id=user.id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)
