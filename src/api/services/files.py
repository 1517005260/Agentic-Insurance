"""File lifecycle service.

Owns the transitions ``files.status`` walks through:

    pending → parsing → indexing → ready
                                  ↘ failed
                              ↘ failed
                       ↘ failed
    ready  → indexing (re-ingest) → ready
    ready  → deleting → (row removed)

Job creation + status transition are **atomic in the route's DB
transaction**: ``begin_ingest_job`` issues a conditional ``UPDATE files
SET status=:next WHERE file_id=:id AND status IN (:allowed)`` and only
inserts the ``IngestJob`` row when the update flipped a row. Two
concurrent reingest+delete on the same file therefore see exactly one
winner, before any background task is scheduled.

Background tasks (``run_parse_index`` / ``run_reingest`` / ``run_delete``)
take the pre-created ``job_id`` and serialize through ``INGEST_LOCK`` —
the lock owns disk-level mutation of the global faiss / bm25 / graph
artifacts, while the SQL state-machine is owned by the route.

Fresh uploads use content-addressed final paths. A request owns that
blob only when it successfully inserts the matching ``files`` row in the
same route transaction; until then, another concurrent request may be
the real owner for the same ``file_id``. Duplicate losers are therefore
allowed to overwrite the final path with byte-identical content, but
they must never unlink it during cleanup.
"""
import asyncio
import hashlib
import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.db import session_scope
from api.models import AuditLog, FileRecord, IngestJob
from config.settings import upload_path, uploads_root
from ingestion.index import purge_file_artifacts
from pipeline.parse_and_index import parse_and_index

logger = logging.getLogger(__name__)


# Single global writer lock. Anything that mutates the shared faiss /
# bm25 / graph stores must be inside ``async with INGEST_LOCK:``.
#
# WARNING: ``asyncio.Lock`` is process-local. The web layer assumes a
# SINGLE uvicorn worker (``uvicorn api.main:app`` with no ``--workers``
# flag). Multi-worker deployment WILL corrupt the global indexes
# because two workers can hold the lock simultaneously. SQLite + WAL
# also tolerates only one writer; both constraints push us to one
# process. Switch to an OS-level filelock (``filelock`` package) +
# Postgres before scaling out.
INGEST_LOCK = asyncio.Lock()


# ---------------------------------------------------------- intake helpers

@dataclass
class StagedUpload:
    """Result of staging an uploaded blob to disk + computing its file_id."""
    file_id: str
    sha256: str
    byte_size: int
    suffix: str
    path: Path


def _derive_file_id(filename: str, sha256_hex: str) -> str:
    """Mirror ``PdfParser._derive_file_id`` so the row PK matches what
    the parser will produce. Same convention everywhere = no surprises."""
    stem = Path(filename).stem.replace(" ", "_")
    return f"{stem}_{sha256_hex[:16]}"


def prepare_upload(*, filename: str, blob: bytes) -> StagedUpload:
    """Pure metadata derivation — sha256 / file_id / suffix.

    Does NOT touch disk. The route uses this to compute ``file_id``
    cheaply before the duplicate-guard query so a rejected duplicate
    never overwrites the existing file's cached blob (the bug we got
    when stage-then-check ran in the other order).

    The returned ``path`` is the *intended* target on disk; the file is
    not present until ``commit_upload_to_disk`` runs.
    """
    if not blob:
        raise ValueError("uploaded file is empty")
    sha = hashlib.sha256(blob).hexdigest()
    suffix = Path(filename).suffix.lower() or ""
    file_id = _derive_file_id(filename, sha)
    return StagedUpload(
        file_id=file_id,
        sha256=sha,
        byte_size=len(blob),
        suffix=suffix,
        path=upload_path(file_id, suffix),
    )


def commit_upload_to_disk(staged: StagedUpload, blob: bytes) -> None:
    """Atomic-write the blob to ``uploads/<file_id><suffix>``.

    A unique ``mkstemp`` ``.part`` file in the target dir avoids the
    shared-tmp collision two concurrent uploads of identical content
    would otherwise hit on a literal ``<file_id>.tmp`` name.
    """
    up_root = uploads_root()
    up_root.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{staged.file_id}.", suffix=".part", dir=str(up_root)
    )
    tmp_path = Path(tmp_name)
    try:
        with open(fd, "wb") as f:
            f.write(blob)
        tmp_path.replace(staged.path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def cleanup_committed_upload(staged: StagedUpload) -> None:
    """Best-effort unlink of a blob this request just wrote.

    Called only AFTER ``commit_upload_to_disk`` and only on a route-
    level failure that would otherwise leak the file. Caller is
    responsible for never invoking this on a duplicate-upload path —
    the duplicate's blob belongs to the EXISTING row, not us.
    """
    try:
        staged.path.unlink(missing_ok=True)
    except Exception:
        logger.exception("orphan-upload cleanup failed: %s", staged.path)


# ----------------------------------------------------------- DB helpers

async def find_file(db: AsyncSession, file_id: str) -> Optional[FileRecord]:
    return await db.get(FileRecord, file_id)


async def list_files(db: AsyncSession) -> list[FileRecord]:
    rows = await db.execute(
        select(FileRecord).order_by(FileRecord.uploaded_at.desc())
    )
    return list(rows.scalars())


# Upload ownership/race walkthrough:
#
# Approach:
# - The pre-insert find_file() is a fast duplicate guard only. It can miss a
#   winner whose transaction has inserted but not committed.
# - commit_upload_to_disk() may therefore run in two identical requests. The
#   second write is byte-identical and harmless.
# - create_pending_record() is the ownership boundary. If its INSERT flushes,
#   this request owns uploads/<file_id><suffix> and route-level cleanup may
#   unlink it on later failure. If the INSERT raises IntegrityError for the
#   files.file_id primary key, another request owns the blob and cleanup is
#   forbidden.
#
# Walkthrough:
# - winner inserts -> loser INSERT raises IntegrityError -> loser maps to 409
#   without cleanup -> winner blob survives -> winner background task ingests.
# - winner inserts + commits -> later loser find_file() sees it -> 409 before
#   commit_upload_to_disk(), so no blob write or cleanup is involved.
# - both miss find_file() -> both commit_upload_to_disk() -> first INSERT wins
#   -> second IntegrityError -> loser exits cleanly without unlinking the
#   shared final path.
async def create_pending_record(
    db: AsyncSession,
    *,
    staged: StagedUpload,
    display_name: str,
    original_filename: str,
    uploaded_by: Optional[int],
) -> FileRecord:
    """Insert a fresh ``files`` row in ``pending`` state.

    Caller usually performs a duplicate-guard query before writing the
    blob, but that SELECT is only an optimization. The authoritative
    ownership boundary is this INSERT: once it flushes, this request owns
    the final upload path and may clean it up if a later step fails.
    If the primary-key INSERT loses to a concurrent identical upload, we
    raise ``FileExistsError`` and the caller must not unlink the blob.
    """
    existing = await find_file(db, staged.file_id)
    if existing is not None:
        raise FileExistsError(
            f"file_id {staged.file_id} already exists (status={existing.status})"
        )

    rec = FileRecord(
        file_id=staged.file_id,
        display_name=display_name,
        original_filename=original_filename,
        suffix=staged.suffix,
        byte_size=staged.byte_size,
        sha256=staged.sha256,
        status="pending",
        uploaded_by=uploaded_by,
    )
    db.add(rec)
    try:
        await db.flush()
    except IntegrityError as exc:
        if _is_file_id_integrity_race(exc):
            raise FileExistsError(
                f"file_id {staged.file_id} already exists"
            ) from exc
        raise
    return rec


def _is_file_id_integrity_race(exc: IntegrityError) -> bool:
    """Return True when an INSERT lost the ``files.file_id`` PK race."""
    message = str(getattr(exc, "orig", exc)).lower()
    return "files.file_id" in message or (
        "unique constraint" in message and "file_id" in message
    )


async def begin_ingest_job(
    db: AsyncSession,
    *,
    file_id: str,
    kind: str,
    next_status: str,
    allowed_current: Iterable[str],
) -> Optional[int]:
    """Atomically flip ``files.status`` and reserve an ``IngestJob`` row.

    Returns the new ``job_id`` on success, or ``None`` if the row's
    current status was outside ``allowed_current`` (caller maps that to
    HTTP 409). The whole sequence runs in the route's transaction so a
    racing reingest+delete sees exactly one winner.

    Caller MUST commit the route session before scheduling the
    background task — see api/routes/files.py for the pattern (and
    ``run_reingest`` log on why deferred-yield commits deadlock here).
    """
    allowed = list(allowed_current)
    res = await db.execute(
        update(FileRecord)
        .where(FileRecord.file_id == file_id, FileRecord.status.in_(allowed))
        .values(status=next_status, error_msg=None)
    )
    if (res.rowcount or 0) == 0:
        return None

    job = IngestJob(
        file_id=file_id,
        kind=kind,
        status="pending",
    )
    db.add(job)
    await db.flush()
    return job.id


# ----------------------------------------------------- background tasks

async def _set_status(
    file_id: str,
    *,
    status: Optional[str] = None,
    error_msg: Optional[str] = None,
    page_count: Optional[int] = None,
    indexed_at: Optional[datetime] = None,
) -> None:
    """Targeted update of a ``files`` row. ``None`` fields are left untouched."""
    async with session_scope() as db:
        rec = await db.get(FileRecord, file_id)
        if rec is None:
            logger.warning("set_status: file row vanished mid-job: %s", file_id)
            return
        if status is not None:
            rec.status = status
        if error_msg is not None:
            rec.error_msg = error_msg
        if page_count is not None:
            rec.page_count = page_count
        if indexed_at is not None:
            rec.indexed_at = indexed_at


async def _start_job(job_id: int) -> None:
    """Flip a pre-reserved IngestJob from ``pending`` to ``running``."""
    async with session_scope() as db:
        job = await db.get(IngestJob, job_id)
        if job is None:
            logger.warning("_start_job: job %d disappeared", job_id)
            return
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)


async def _close_job(
    job_id: int,
    *,
    ok: bool,
    error_msg: Optional[str] = None,
    log_tail: Optional[str] = None,
) -> None:
    async with session_scope() as db:
        job = await db.get(IngestJob, job_id)
        if job is None:
            return
        job.status = "done" if ok else "failed"
        job.finished_at = datetime.now(timezone.utc)
        if error_msg is not None:
            job.error_msg = error_msg
        if log_tail is not None:
            # Cap at ~4KB so a runaway log doesn't bloat the row.
            job.log_tail = log_tail[-4096:]


async def run_parse_index(file_id: str, source_path: Path, *, job_id: int) -> None:
    """Parse + index a freshly-uploaded file.

    The route already created the ``files`` row in ``pending`` and the
    matching ``IngestJob`` row in ``pending``. We own the transition
    through ``parsing`` → ``indexing`` → ``ready`` / ``failed`` plus the
    job's ``running`` → ``done`` / ``failed``.
    """
    async with INGEST_LOCK:
        try:
            await _start_job(job_id)
            await _set_status(file_id, status="parsing")
            # parse + 4-builder index. Builders run serially per project
            # policy on small hosts (8GB WSL OOM otherwise).
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: parse_and_index(
                    source_path,
                    file_id=file_id,
                    overwrite=True,
                    parallel_builders=False,
                ),
            )
            if not result.ok:
                raise RuntimeError(result.error or "ingest failed")

            page_count = result.parse.total_pages if result.parse else None
            await _set_status(
                file_id,
                status="ready",
                page_count=page_count,
                indexed_at=datetime.now(timezone.utc),
            )
            log_tail = "\n".join(
                f"{r.index_name}: items={r.item_count} skipped={r.skipped_reason or '-'}"
                for r in result.indexes
            )
            # Job-close is best-effort: a telemetry write failure here
            # must NOT flip the file back to 'failed' — the indexes are
            # already correct on disk.
            try:
                await _close_job(job_id, ok=True, log_tail=log_tail)
            except Exception:
                logger.exception("job-close telemetry failed (file already ready): %d", job_id)
            logger.info("ingest ok: file_id=%s pages=%s", file_id, page_count)
        except Exception as exc:
            logger.exception("ingest failed: file_id=%s", file_id)
            # Roll back any partial index state. Some builders may have
            # already written rows tagged with ``file_id`` before the
            # one that raised — leaving them behind would silently
            # poison every retrieval channel. ``keep_upload=True`` so
            # the operator can retry without re-uploading the original.
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: purge_file_artifacts(file_id, keep_upload=True),
                )
            except Exception:
                logger.exception(
                    "post-failure purge raised (file %s left in failed state with possible partial indexes)",
                    file_id,
                )
            await _set_status(file_id, status="failed", error_msg=f"{type(exc).__name__}: {exc}")
            await _close_job(job_id, ok=False, error_msg=str(exc))


async def run_reingest(file_id: str, *, job_id: int) -> None:
    """Wipe-and-rebuild for a file we already have on disk.

    Requires the cached upload (``uploads/<file_id><suffix>``); if it's
    missing we fail loud rather than try to recover from paddle_ocr cache
    only — that's a separate "salvage" path the operator runs by hand.

    NB: ``purge_file_artifacts(keep_upload=True)`` — without it, we'd
    delete the source blob mid-flight and the subsequent ``parse_and_index``
    would fail with FileNotFoundError.
    """
    async with INGEST_LOCK:
        try:
            await _start_job(job_id)

            async with session_scope() as db:
                rec = await db.get(FileRecord, file_id)
                if rec is None:
                    raise FileNotFoundError(f"no files row: {file_id}")
                src = upload_path(file_id, rec.suffix)
                if not src.is_file():
                    raise FileNotFoundError(
                        f"cached upload missing: {src}; cannot re-ingest"
                    )

            # Drop indexes first so parse_and_index can rebuild from a
            # clean slate. ``keep_upload=True`` preserves the source blob;
            # purge_file_artifacts is otherwise idempotent.
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: purge_file_artifacts(file_id, keep_upload=True),
            )

            result = await loop.run_in_executor(
                None,
                lambda: parse_and_index(
                    src,
                    file_id=file_id,
                    overwrite=True,
                    parallel_builders=False,
                ),
            )
            if not result.ok:
                raise RuntimeError(result.error or "reingest failed")

            page_count = result.parse.total_pages if result.parse else None
            await _set_status(
                file_id,
                status="ready",
                page_count=page_count,
                indexed_at=datetime.now(timezone.utc),
            )
            try:
                await _close_job(job_id, ok=True)
            except Exception:
                logger.exception("job-close telemetry failed (file already ready): %d", job_id)
        except Exception as exc:
            logger.exception("reingest failed: file_id=%s", file_id)
            # We've already purged the old indexes (line above the try).
            # If the rebuild raised partway, partial new rows may exist;
            # purge again so the file is consistently absent from every
            # store. Upload blob is preserved so the operator can retry.
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: purge_file_artifacts(file_id, keep_upload=True),
                )
            except Exception:
                logger.exception(
                    "post-failure purge raised (file %s left in failed state with possible partial indexes)",
                    file_id,
                )
            await _set_status(file_id, status="failed", error_msg=f"{type(exc).__name__}: {exc}")
            await _close_job(job_id, ok=False, error_msg=str(exc))


async def run_delete(file_id: str, *, job_id: int) -> None:
    """Cascade-delete one file: indexes first, then files row.

    Order matters: we wipe disk before the row, so a crash in the middle
    leaves status='deleting' on a row whose indexes are already partly
    gone — operator can re-trigger delete and the idempotent purge will
    finish the job. The opposite order would orphan disk artifacts with
    no DB pointer, requiring ``indexed_file_ids()`` reconciliation.

    On the rare DB-only failure AFTER the on-disk purge succeeded, the
    ``files`` row stays in ``deleting`` (operator intervention). The
    ``IngestJob`` row records the exact failure for debugging.
    """
    async with INGEST_LOCK:
        try:
            await _start_job(job_id)
            # Pull the suffix BEFORE purge so the uploads-dir delete is
            # exact (no chance of clobbering a sibling's blob if file_ids
            # share a prefix).
            async with session_scope() as db:
                rec = await db.get(FileRecord, file_id)
                upload_suffix = rec.suffix if rec is not None else None

            loop = asyncio.get_running_loop()
            counts = await loop.run_in_executor(
                None,
                lambda: purge_file_artifacts(
                    file_id, upload_suffix=upload_suffix
                ),
            )
            log_tail = ", ".join(f"{k}={v}" for k, v in counts.items())

            # Persist the per-step counts as an audit trail BEFORE the
            # cascade kills the IngestJob row. AuditLog has no FK to
            # files (target is just a text ref), so it survives.
            async with session_scope() as db:
                db.add(
                    AuditLog(
                        action="file.delete.complete",
                        target=file_id,
                        payload_json=json.dumps(counts, ensure_ascii=False),
                    )
                )
                rec = await db.get(FileRecord, file_id)
                if rec is not None:
                    await db.delete(rec)
            # The files-row delete cascaded the IngestJob row away, so
            # ``_close_job`` will be a no-op — best-effort, don't crash
            # the bg task on it.
            try:
                await _close_job(job_id, ok=True, log_tail=log_tail)
            except Exception:
                pass
            logger.info("delete ok: file_id=%s counts=%s", file_id, counts)
        except Exception as exc:
            logger.exception("delete failed: file_id=%s", file_id)
            # Don't overwrite status='deleting' (already set by route).
            await _set_status(file_id, error_msg=f"{type(exc).__name__}: {exc}")
            await _close_job(job_id, ok=False, error_msg=str(exc))


# ----------------------------------------------- crash-recovery helpers

# Statuses that mean "a worker was midway through this when the process
# stopped". On lifespan startup these are converted to ``failed`` so the
# operator can intervene; without this sweep they'd appear stuck forever.
_STALE_FILE_STATUSES = ("pending", "parsing", "indexing", "deleting")
_STALE_JOB_STATUSES = ("pending", "running")


async def reconcile_after_restart() -> dict[str, int]:
    """Mark mid-flight rows as failed on app startup.

    Returns counts of (files, jobs) that were rewritten so the lifespan
    handler can log a one-line audit line. Idempotent: a clean restart
    flips zero rows.
    """
    async with session_scope() as db:
        files_res = await db.execute(
            update(FileRecord)
            .where(FileRecord.status.in_(_STALE_FILE_STATUSES))
            .values(status="failed", error_msg="process restarted mid-job")
        )
        jobs_res = await db.execute(
            update(IngestJob)
            .where(IngestJob.status.in_(_STALE_JOB_STATUSES))
            .values(
                status="failed",
                error_msg="process restarted mid-job",
                finished_at=datetime.now(timezone.utc),
            )
        )
    return {"files": files_res.rowcount or 0, "jobs": jobs_res.rowcount or 0}


async def sweep_orphan_uploads() -> dict[str, int]:
    """Delete ``uploads/<file_id><suffix>`` blobs whose file_id has no DB row.

    The route writes the blob to disk BEFORE inserting the ``files`` row
    (and committing). A process crash inside that window leaves an
    orphan original on disk with no SQL pointer; normal exception
    cleanup misses these. Run this on lifespan startup to catch them.

    Returns ``{"scanned": N, "removed": M, "skipped_part_files": K}``.

    Conservative: skips ``.part`` mkstemp leftovers (those are stale
    partial writes — separate concern; another orphan sweep cleans them
    if needed). Only looks at files whose stem-without-final-ext is a
    plausible file_id; partial writes with random tmp names are left
    alone.
    """
    from config.settings import uploads_root

    up_root = uploads_root()
    scanned = 0
    removed = 0
    skipped_parts = 0
    if not up_root.exists():
        return {"scanned": 0, "removed": 0, "skipped_part_files": 0}

    async with session_scope() as db:
        known_ids = set(
            (await db.execute(select(FileRecord.file_id))).scalars().all()
        )

    for entry in up_root.iterdir():
        if not entry.is_file():
            continue
        # Skip in-flight tmp files (.<file_id>.<rand>.part). Those are
        # the mkstemp staging path used by commit_upload_to_disk; if
        # they're stale, a separate sweep can age them out.
        if entry.name.startswith(".") and entry.suffix == ".part":
            skipped_parts += 1
            continue
        scanned += 1
        # Inverse of upload_path(file_id, suffix): strip the final ext.
        candidate_id = entry.with_suffix("").name or entry.name
        if candidate_id in known_ids:
            continue
        try:
            entry.unlink()
            removed += 1
            logger.warning("orphan upload removed (no DB row): %s", entry.name)
        except OSError:
            logger.exception("orphan upload unlink failed: %s", entry)

    return {
        "scanned": scanned,
        "removed": removed,
        "skipped_part_files": skipped_parts,
    }


__all__ = [
    "INGEST_LOCK",
    "StagedUpload",
    "prepare_upload",
    "commit_upload_to_disk",
    "cleanup_committed_upload",
    "find_file",
    "list_files",
    "create_pending_record",
    "begin_ingest_job",
    "run_parse_index",
    "run_reingest",
    "run_delete",
    "reconcile_after_restart",
    "sweep_orphan_uploads",
]
