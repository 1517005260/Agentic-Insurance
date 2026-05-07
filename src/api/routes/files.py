"""File CRUD routes.

Permission model (matches the project-wide RBAC):

* analyst — list, get, download original
* admin   — everything analyst can do, plus upload, delete, reingest

Background work (parse + index, delete, reingest) is dispatched via
``BackgroundTasks`` after the request returns 202; clients poll the
file status to know when it's done.
"""
import logging
from datetime import datetime
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_session, require_admin
from api.models import AuditLog, FileRecord, IngestJob, User
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
from config.settings import upload_path

logger = logging.getLogger(__name__)

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


# ------------------------------------------------------------------ upload

@router.post(
    "",
    response_model=FileOut,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_admin)],
)
async def upload(
    background: BackgroundTasks,
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

    background.add_task(run_parse_index, rec.file_id, staged.path, job_id=job_id)
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
    background: BackgroundTasks,
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
    background.add_task(run_reingest, file_id, job_id=job_id)
    return FileOut.from_row(rec)


@router.delete(
    "/{file_id}",
    response_model=FileOut,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_admin)],
)
async def delete_one(
    file_id: str,
    background: BackgroundTasks,
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
    background.add_task(run_delete, file_id, job_id=job_id)
    return FileOut.from_row(rec)
