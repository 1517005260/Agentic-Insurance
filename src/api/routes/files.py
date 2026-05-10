"""File CRUD routes.

Permission model (matches the project-wide RBAC):

* analyst — list, get, download original
* admin   — everything analyst can do, plus upload, delete, reingest

Background work (parse + index, delete, reingest) is dispatched via
detached ``asyncio.create_task`` (NOT FastAPI ``BackgroundTasks``):
the latter wraps the work into the response lifecycle, which means a
long ingest stays attached to the upload-POST coroutine and starves
the response loop's ability to service short requests like
``/auth/me`` or ``/chat/sessions/...``. With detached tasks the bg
work runs independently of any caller's HTTP handler, and
``register_bus`` is the only handle the SSE route needs.
"""
import asyncio
import logging
from datetime import datetime
from typing import Coroutine, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_session, require_admin
from api.models import AuditLog, FileRecord, IngestJob, User
from api.runners.events import EventType
from api.runners.ingestion_runner import wait_for_bus
from api.services.files import (
    begin_ingest_job,
    cleanup_committed_upload,
    commit_upload_to_disk,
    create_pending_record,
    find_file,
    list_files,
    prepare_upload,
    run_delete,
    run_parse_index,
    run_reingest,
)
from api.services.preview import page_count as pdf_page_count, render_pdf_page_to_cache
from api.sse import format_event
from config.settings import upload_path

logger = logging.getLogger(__name__)


def _spawn_ingest_task(coro: Coroutine) -> None:
    """Detach an ingest coroutine from the response lifecycle.

    ``asyncio.create_task`` on the running loop, with a callback that
    logs any unhandled exception so a crashed bg task can't silently
    rot. We deliberately do NOT use FastAPI's ``BackgroundTasks``: it
    runs after the response *body* finishes, but the body for an
    ingest 202 is empty, so the work would still be tied to the
    response task and can starve the loop's request handling.
    """
    task = asyncio.create_task(coro)

    def _log_crash(t: "asyncio.Task") -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.exception(
                "ingest bg task crashed", exc_info=(type(exc), exc, exc.__traceback__)
            )

    task.add_done_callback(_log_crash)

router = APIRouter(prefix="/files", tags=["files"])


# ----------------------------------------------------------- response shapes

class FileOut(BaseModel):
    file_id: str
    display_name: str
    original_filename: str
    suffix: str
    byte_size: int
    sha256: str
    page_count: Optional[int]
    status: str
    error_msg: Optional[str]
    uploaded_by: Optional[int]
    uploaded_at: datetime
    indexed_at: Optional[datetime]

    @classmethod
    def from_row(cls, row: FileRecord) -> "FileOut":
        return cls(
            file_id=row.file_id,
            display_name=row.display_name,
            original_filename=row.original_filename,
            suffix=row.suffix,
            byte_size=row.byte_size,
            sha256=row.sha256,
            page_count=row.page_count,
            status=row.status,
            error_msg=row.error_msg,
            uploaded_by=row.uploaded_by,
            uploaded_at=row.uploaded_at,
            indexed_at=row.indexed_at,
        )


class IngestJobOut(BaseModel):
    id: int
    file_id: str
    kind: str
    status: str
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    error_msg: Optional[str]
    log_tail: Optional[str]
    created_at: datetime


# --------------------------------------------------------------- list / get

@router.get("", response_model=list[FileOut])
async def list_all(
    db: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> list[FileOut]:
    rows = await list_files(db)
    return [FileOut.from_row(r) for r in rows]


@router.get("/{file_id}", response_model=FileOut)
async def get_one(
    file_id: str,
    db: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> FileOut:
    rec = await find_file(db, file_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found")
    return FileOut.from_row(rec)


@router.get("/{file_id}/jobs", response_model=list[IngestJobOut])
async def list_jobs(
    file_id: str,
    db: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> list[IngestJobOut]:
    rec = await find_file(db, file_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found")
    res = await db.execute(
        select(IngestJob)
        .where(IngestJob.file_id == file_id)
        .order_by(IngestJob.id.desc())
    )
    return [
        IngestJobOut(
            id=j.id,
            file_id=j.file_id,
            kind=j.kind,
            status=j.status,
            started_at=j.started_at,
            finished_at=j.finished_at,
            error_msg=j.error_msg,
            log_tail=j.log_tail,
            created_at=j.created_at,
        )
        for j in res.scalars()
    ]


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


async def _replay_terminal_job(job: IngestJob):
    """Synthesize a single ``final`` + ``done`` for a job already at rest.

    Used when the SSE caller arrives after ``run_*`` finished and the
    bus has been unregistered. Lets the frontend collapse the timeline
    to the final state without a separate REST round-trip.
    """
    yield format_event(
        EventType.FINAL,
        {
            "file_id": job.file_id,
            "status": "ready" if job.status == "done" else "failed",
            "job_status": job.status,
            "error": job.error_msg,
            "log_tail": job.log_tail,
        },
    )
    yield format_event(EventType.DONE, {})


async def _replay_terminal_file(rec: FileRecord):
    """Same shape as :func:`_replay_terminal_job` but driven by the file row.

    Fallback when the ``ingest_jobs`` row leaked into ``running``
    despite the file row reaching ``ready`` / ``failed`` (job-close is
    best-effort in the service layer; a transient SQLite write failure
    can leave the bookkeeping inconsistent).
    """
    yield format_event(
        EventType.FINAL,
        {
            "file_id": rec.file_id,
            "status": rec.status,
            "error": rec.error_msg,
            "page_count": rec.page_count,
        },
    )
    yield format_event(EventType.DONE, {})


@router.get("/{file_id}/jobs/stream")
async def stream_jobs(
    file_id: str,
    db: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> StreamingResponse:
    """Subscribe to the latest ingest job's stage progress.

    Three terminal cases:

    1. Latest job is ``running`` and a bus is registered — stream live
       ``stage`` / ``final`` / ``done`` frames from the bg task.
    2. Latest job is ``running`` but the bus has not registered yet
       (race against ``BackgroundTasks`` scheduling) — short poll up
       to 2 s, then stream live frames.
    3. Latest job is already terminal (``done`` / ``failed``) — synthesize
       a single ``final`` + ``done`` from the row so the client always
       sees a closed stream.

    The "wait timed out AND row is still nonterminal" path refetches the
    row before falling back to terminal replay so we don't synthesize a
    spurious ``failed`` for a job whose bg task is just slow to schedule.

    Multi-subscriber: the bus is constructed with
    ``replay_buffered=True`` so each ``stream()`` call gets its own
    queue, seeded with the full event history. The FilesPage
    minimized-ingest chip and any extra browser tabs all see the
    same timeline.
    """
    rec = await find_file(db, file_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found")
    res = await db.execute(
        select(IngestJob)
        .where(IngestJob.file_id == file_id)
        .order_by(IngestJob.id.desc())
        .limit(1)
    )
    latest: Optional[IngestJob] = res.scalar_one_or_none()
    if latest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no jobs for file",
        )

    if latest.status in ("done", "failed"):
        return StreamingResponse(
            _replay_terminal_job(latest),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    # Snapshot the id and release the read transaction BEFORE polling
    # the bus. ``db.refresh()`` on stale ORM objects re-reads from the
    # SAME open transaction, which under SQLite WAL means we keep
    # seeing the snapshot taken at the start of this request — the
    # background task may have committed ``ready``/``done`` 1.5 s ago
    # but ``refresh()`` would still report ``running``, and we'd 503.
    # Issuing fresh selects via ``session_scope()`` reads the latest
    # committed state, which is what the polling client expects.
    latest_job_id = latest.id
    await db.rollback()

    bus = await wait_for_bus(latest_job_id, timeout=2.0)
    if bus is None:
        # Bus never registered. Three real causes:
        #   (a) the bg task finished + unregistered between our row
        #       query and the wait — refetch and replay terminal.
        #   (b) job-close telemetry raised but the file row already
        #       made it to ``ready`` / ``failed`` (job-close is
        #       best-effort by design — see service _close_job
        #       try/except); the job row stays ``running`` but the
        #       file is at rest. Trust the file row in that case.
        #   (c) ``_spawn_ingest_task`` hasn't yielded to the loop yet
        #       (rare but possible under load burst) — return 503 so
        #       the client retries with backoff rather than seeing a
        #       fake "failed" from the still-pending row.
        from api.db import session_scope as _scope
        async with _scope() as fresh_db:
            latest_fresh = await fresh_db.get(IngestJob, latest_job_id)
            rec_fresh = await fresh_db.get(FileRecord, file_id)
        if latest_fresh is not None and latest_fresh.status in ("done", "failed"):
            return StreamingResponse(
                _replay_terminal_job(latest_fresh),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        if rec_fresh is not None and rec_fresh.status in ("ready", "failed"):
            # File row is terminal but job row leaked. Synthesize
            # from the file row directly so the client sees a closed
            # stream without polling /jobs forever.
            return StreamingResponse(
                _replay_terminal_file(rec_fresh),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ingest job {latest_job_id} pending dispatch; retry shortly",
            headers={"Retry-After": "2"},
        )

    # Bus is multi-consumer with replay (see api/services/files.py
    # constructing with replay_buffered=True). Multiple subscribers
    # are legal: each ``bus.stream()`` call gets its own queue,
    # seeded with the full event history so a late client (e.g. the
    # FilesPage minimized-ingest chip reopening the drawer) sees
    # exactly what the original consumer saw.
    return StreamingResponse(
        bus.stream(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get("/{file_id}/download")
async def download_original(
    file_id: str,
    db: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> FileResponse:
    """Stream back the cached upload (so the frontend's react-pdf can render
    it for the citation drawer)."""
    rec = await find_file(db, file_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found")
    path = upload_path(file_id, rec.suffix)
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="original blob no longer cached",
        )
    media_type = "application/pdf" if rec.suffix == ".pdf" else "application/octet-stream"
    return FileResponse(path=path, media_type=media_type, filename=rec.original_filename)


@router.get("/{file_id}/preview")
async def page_preview(
    file_id: str,
    page: int = 1,
    db: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> FileResponse:
    """Render-on-demand JPG of any 1-based ``page`` of the cached PDF.

    Default ``page=1`` powers the FilesPage card thumbnail; the
    full-file preview drawer scrolls through ``?page=N`` for each
    visible page. Output is cached under
    ``local_storage/preview/<file_id>/p_NNNN.jpg`` and re-served on
    subsequent hits. Re-ingest invalidates the cache (see
    ``purge_file_artifacts``).

    The previous implementation served PaddleOCR's layout-detection
    visualization (``layout_det_res_0.jpg``) which had bounding boxes
    overlaid — visually it looked like a CV debug screenshot rather
    than a page thumbnail. We now render via pypdfium2 (PDFium, the
    same engine Chromium uses).
    """
    # file_id flows from URL into a filesystem join; reject any token
    # that could escape the storage root before touching disk.
    if "/" in file_id or "\\" in file_id or ".." in file_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid file_id",
        )
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="page must be >= 1",
        )
    rec = await find_file(db, file_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found")
    src = upload_path(file_id, rec.suffix)
    if not src.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="original blob no longer cached",
        )

    import asyncio
    loop = asyncio.get_running_loop()
    rendered = await loop.run_in_executor(
        None,
        lambda: render_pdf_page_to_cache(src, file_id=file_id, page_number=page),
    )
    if rendered is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"page {page} could not be rendered",
        )
    # ``private`` so reverse proxies / CDNs do not cache this auth-gated
    # response into a shared cache that other users could hit.
    return FileResponse(
        path=rendered,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.get("/{file_id}/page-count")
async def page_count_endpoint(
    file_id: str,
    db: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> dict[str, int]:
    """Cheap page-count probe for the full-file preview drawer.

    Reads the PDF's xref via pypdfium2 — no rendering, returns within a
    few ms even for hundred-page documents. Falls back to the
    ``files.page_count`` row column when the PDF is unreadable so the
    UI still gets a usable bound.
    """
    if "/" in file_id or "\\" in file_id or ".." in file_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid file_id",
        )
    rec = await find_file(db, file_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found")
    src = upload_path(file_id, rec.suffix)
    n = pdf_page_count(src) if src.is_file() else 0
    if n <= 0 and rec.page_count:
        n = int(rec.page_count)
    return {"page_count": n}


# ------------------------------------------------------------------ upload

@router.post(
    "",
    response_model=FileOut,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_admin)],
)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    display_name: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> FileOut:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="upload missing filename",
        )

    # Read once. Safe for the demo since PDFs are <50MB; for larger
    # uploads switch to streamed sha256 + chunked write.
    blob = await file.read()
    if not blob:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="upload is empty",
        )

    # Pure metadata derivation. ``prepare_upload`` does NOT touch disk
    # — that order matters: a duplicate-upload check that runs AFTER the
    # write would let cleanup-on-409 unlink the existing file's cached
    # blob (the row owned by some other request would suddenly have no
    # source on disk).
    staged = prepare_upload(filename=file.filename, blob=blob)

    existing = await find_file(db, staged.file_id)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"file already exists (status={existing.status}); call reingest to rebuild",
        )

    # No visible duplicate — write the original to disk now. If a concurrent
    # identical request already inserted the row in an uncommitted transaction,
    # create_pending_record() below is still the authoritative ownership check.
    commit_upload_to_disk(staged, blob)
    cleanup_upload_on_error = True
    try:
        try:
            rec = await create_pending_record(
                db,
                staged=staged,
                display_name=display_name or file.filename,
                original_filename=file.filename,
                uploaded_by=user.id,
            )
        except FileExistsError as exc:
            cleanup_upload_on_error = False
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="file already exists; call reingest to rebuild",
            ) from exc
        # Reserve the IngestJob row in the same transaction; bg task
        # only flips it to 'running' / 'done' / 'failed'. Conditional
        # check uses status='pending' (the row we just inserted).
        job_id = await begin_ingest_job(
            db,
            file_id=rec.file_id,
            kind="parse_index",
            next_status="pending",
            allowed_current=["pending"],
        )
        if job_id is None:  # impossible since we just created the row
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="could not reserve ingest job",
            )
        db.add(
            AuditLog(
                user_id=user.id,
                action="file.upload",
                target=rec.file_id,
                payload_json=None,
            )
        )
        # Commit BEFORE scheduling the background task. Starlette runs
        # ``BackgroundTasks`` before FastAPI tears down ``get_session``, so a
        # deferred end-of-yield commit would still hold the write lock when
        # the bg task tries its first INSERT — and despite ``busy_timeout``
        # SQLite raises "database is locked" immediately under aiosqlite's
        # async dispatch. Commit here, the dep's tail-commit becomes a no-op.
        await db.commit()
    except Exception:
        if cleanup_upload_on_error:
            cleanup_committed_upload(staged)
        raise

    cfg = getattr(request.app.state, "config", None)
    linear_config = cfg.materialize_linear_rag_config() if cfg is not None else None
    parse_workers = cfg.ingest_parallel_workers() if cfg is not None else 1
    _spawn_ingest_task(
        run_parse_index(
            rec.file_id,
            staged.path,
            job_id=job_id,
            linear_config=linear_config,
            parse_workers=parse_workers,
        )
    )
    return FileOut.from_row(rec)


# -------------------------------------------------------- reingest / delete

@router.post(
    "/{file_id}/reingest",
    response_model=FileOut,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_admin)],
)
async def reingest(
    file_id: str,
    request: Request,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> FileOut:
    rec = await find_file(db, file_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found")
    # Atomic CAS: only flip ready/failed → indexing. Two concurrent
    # reingest+delete races see exactly one winner; the loser gets 409
    # without ever scheduling a bg task.
    job_id = await begin_ingest_job(
        db,
        file_id=file_id,
        kind="reingest",
        next_status="indexing",
        allowed_current=["ready", "failed"],
    )
    if job_id is None:
        # Re-fetch for the accurate "busy" detail.
        await db.refresh(rec)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"file is busy (status={rec.status}); wait for it to finish",
        )
    db.add(
        AuditLog(
            user_id=user.id,
            action="file.reingest",
            target=file_id,
            payload_json=None,
        )
    )
    # See upload() for why we commit before scheduling rather than relying
    # on get_session's tail-commit.
    await db.commit()
    await db.refresh(rec)
    cfg = getattr(request.app.state, "config", None)
    linear_config = cfg.materialize_linear_rag_config() if cfg is not None else None
    parse_workers = cfg.ingest_parallel_workers() if cfg is not None else 1
    _spawn_ingest_task(
        run_reingest(
            file_id,
            job_id=job_id,
            linear_config=linear_config,
            parse_workers=parse_workers,
        )
    )
    return FileOut.from_row(rec)


@router.delete(
    "/{file_id}",
    response_model=FileOut,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_admin)],
)
async def delete_one(
    file_id: str,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> FileOut:
    rec = await find_file(db, file_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found")
    # Atomic CAS — same idea as reingest. Allow delete from any terminal
    # state (ready / failed) so an operator can clean up after a stuck
    # ingest. Disallow during in-progress transitions.
    job_id = await begin_ingest_job(
        db,
        file_id=file_id,
        kind="delete",
        next_status="deleting",
        allowed_current=["ready", "failed"],
    )
    if job_id is None:
        await db.refresh(rec)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"file is busy (status={rec.status}); wait for it to finish",
        )
    db.add(
        AuditLog(
            user_id=user.id,
            action="file.delete",
            target=file_id,
            payload_json=None,
        )
    )
    # See upload() for why we commit before scheduling rather than relying
    # on get_session's tail-commit.
    await db.commit()
    await db.refresh(rec)
    _spawn_ingest_task(run_delete(file_id, job_id=job_id))
    return FileOut.from_row(rec)
