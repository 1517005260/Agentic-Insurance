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
from typing import Any, Callable, Iterable, Mapping, Optional

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.db import session_scope
from api.models import AuditLog, FileRecord, IngestJob
from api.runners.events import EventBus
from api.runners.ingestion_runner import register_bus, unregister_bus
from config.settings import paddle_ocr_root, upload_path, uploads_root
from ingestion.index.maintenance import purge_file_artifacts
from pipeline.parse_and_index import index_parsed, parse_only

logger = logging.getLogger(__name__)


# Module-level refresh hook. Lifespan registers a closure that reloads
# ``PageStore`` / ``InventoryStore`` / ``GraphPPRChannel`` from disk;
# the ingest tasks call it inside ``INGEST_LOCK`` after a successful
# parse / reingest / delete so the long-lived singletons pick up new
# (or removed) on-disk artifacts. Without this the singletons reflect
# the lifespan-boot snapshot forever and tools that read them
# (toc / graph_explore / read_page / GraphService) silently see empty
# state for any file uploaded after boot.
#
# Stored as a list to make register/unregister atomic in pure Python
# without a lock — there is at most one hook in practice.
_REFRESH_HOOKS: list[Callable[[], None]] = []


def register_refresh_hook(hook: Callable[[], None]) -> None:
    """Register the lifespan-built ``refresh_indexes`` closure."""
    _REFRESH_HOOKS.clear()
    _REFRESH_HOOKS.append(hook)


def unregister_refresh_hook() -> None:
    _REFRESH_HOOKS.clear()


def _run_refresh_hook() -> None:
    """Invoke the registered refresh hook. Best-effort: log + swallow on
    exception so a singleton-reload bug never poisons the SSE final
    frame for an otherwise-successful ingest."""
    for hook in list(_REFRESH_HOOKS):
        try:
            hook()
        except Exception:
            logger.exception("refresh_indexes hook raised; singletons may be stale")


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


# Parse-stage semaphore — bounded concurrency for the OCR pre-pass
# (paddle is a remote service that handles its own queue, no shared
# in-process state). Capacity is sourced from the ``ingest.parallel_workers``
# admin entry on first use; admin patches require a **process restart**
# to take effect — see :func:`_get_parse_sem` for the rationale and
# the safety implications of the alternative.
_PARSE_SEM: Optional[asyncio.Semaphore] = None
_PARSE_SEM_CAP: int = 0
# Once-only flag so the "config changed but cap is frozen" warning doesn't
# spam the log every request after an admin patch. Reset on process boot.
_PARSE_SEM_DRIFT_WARNED: bool = False


def _get_parse_sem(workers: int) -> asyncio.Semaphore:
    """Return the process-wide parse semaphore; create once.

    We can't construct ``asyncio.Semaphore`` at module import — it has
    to bind to the event loop that will call ``acquire``. The first
    request through this function snapshots ``workers`` (from
    ConfigStore) and builds the semaphore against the running loop.

    **Capacity is frozen for the process lifetime.** ``Semaphore`` has
    no public ``resize()`` API, and silently swapping the semaphore
    object on an admin patch would temporarily *bypass* the limit:
    requests already inside the old semaphore keep running while new
    requests queue on a separate, fresh slot pool. Admins who need a
    new cap restart the worker. The route layer still reads the live
    config value each request — that's harmless, the lookup just
    short-circuits on the second-and-later call.
    """
    global _PARSE_SEM, _PARSE_SEM_CAP, _PARSE_SEM_DRIFT_WARNED
    if _PARSE_SEM is None:
        _PARSE_SEM = asyncio.Semaphore(max(1, workers))
        _PARSE_SEM_CAP = max(1, workers)
    elif _PARSE_SEM_CAP != workers and not _PARSE_SEM_DRIFT_WARNED:
        # Admin changed ingest.parallel_workers after process boot.
        # Honour the boot-time value and warn ONCE so operators see
        # they need to restart to pick up the new cap (without log
        # spam on every subsequent ingest request).
        logger.warning(
            "ingest.parallel_workers changed (boot=%d, live=%d) — restart "
            "the worker to apply; continuing with boot-time semaphore "
            "(this warning fires only once per process)",
            _PARSE_SEM_CAP,
            workers,
        )
        _PARSE_SEM_DRIFT_WARNED = True
    return _PARSE_SEM


def _purge_paddle_only(file_id: str) -> None:
    """Remove just ``paddle_ocr/<file_id>/`` — safe without INGEST_LOCK.

    Used on parse-stage failure where the global faiss / bm25 / graph
    stores have not been touched yet. ``purge_file_artifacts`` would
    cascade into those stores and require the lock; here we only need
    to drop the per-file OCR cache so a retry doesn't hit a stale
    paddle output. Idempotent (rmtree silently no-ops on missing dir).
    """
    import shutil

    target = paddle_ocr_root() / file_id
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)


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


def _aggregate_timings(events_seen: list[Mapping[str, Any]]) -> dict[str, int]:
    """Pluck ``elapsed_ms`` out of buffered ``stage:done`` events."""
    return {e["stage"]: int(e["elapsed_ms"]) for e in events_seen if "elapsed_ms" in e and "stage" in e}


async def run_parse_index(
    file_id: str,
    source_path: Path,
    *,
    job_id: int,
    linear_config: Optional[Any] = None,
    parse_workers: int = 1,
) -> None:
    """Parse + index a freshly-uploaded file.

    The route already created the ``files`` row in ``pending`` and the
    matching ``IngestJob`` row in ``pending``. We own the transition
    through ``parsing`` → ``indexing`` → ``ready`` / ``failed`` plus the
    job's ``running`` → ``done`` / ``failed``.

    Stage progress is fanned out via an :class:`EventBus` registered
    under ``job_id`` so ``GET /files/{id}/jobs/stream`` can subscribe.
    The pipeline emits ``parse`` + ``page_assets`` + 4 builder stages;
    the bus's ``final`` carries the aggregated ``timings_ms`` map. Bus
    is closed exactly once — clean ``done`` on success, ``error`` +
    ``done`` on failure — and the ``finally`` clears the registry slot.

    Concurrency model (per ``ingest.parallel_workers`` admin knob):
    parse stages run under a process-wide ``Semaphore`` of capacity
    ``parse_workers`` so up to N PDFs OCR in parallel. Index writes
    still go through the global ``INGEST_LOCK`` because faiss / graph
    stores have no internal locking. ``parse_workers=1`` reproduces
    the original fully-serial behaviour.

    ``linear_config`` (LinearRAGConfig; ``None`` falls back to dataclass
    defaults) carries admin-tuned literal-backfill knobs into the
    GraphIndexBuilder. The route materializes one snapshot per request
    from ConfigStore so concurrent admin PATCHes can't half-apply.
    """
    loop = asyncio.get_running_loop()
    # ``replay_buffered=True`` makes the bus multi-consumer with full
    # event replay on subscribe — required so a "minimized" upload
    # dialog can be reopened later via the FilesPage chip and still
    # see the stage timeline. Disconnects no longer flip is_closed,
    # so the bg task keeps running regardless of who is watching.
    bus = EventBus(loop=loop, replay_buffered=True)
    register_bus(job_id, bus)
    bus_closed = False
    stage_dones: list[Mapping[str, Any]] = []

    def _emit(event: str, data: Mapping[str, Any]) -> None:
        if event == "stage" and data.get("phase") == "done":
            stage_dones.append(dict(data))
        bus.push(event, data)

    parse_sem = _get_parse_sem(parse_workers)

    async def _on_failure(exc: BaseException, *, stage: str) -> None:
        """Cleanup branched on which stage raised.

        ``stage="parse"`` — global faiss / bm25 / graph stores have not
        been touched yet (parse only writes per-file paddle output), so
        we drop just ``paddle_ocr/<file_id>/`` without touching the
        global stores. Critical: we are NOT inside INGEST_LOCK at this
        point and ``purge_file_artifacts`` would race a concurrent
        index-stage write of another file.

        ``stage="index"`` — caller is already inside INGEST_LOCK, so
        the full ``purge_file_artifacts`` is safe AND required (the
        builders may have written partial rows). After purge, refresh
        the long-lived singletons (PageStore / GraphPPRChannel) so they
        don't keep serving entries for a file that no longer exists on
        disk.

        Each step is best-effort so a downstream IO error never masks
        the original SSE error frame the outer block emits.
        """
        if stage == "parse":
            try:
                await loop.run_in_executor(
                    None, lambda: _purge_paddle_only(file_id)
                )
            except Exception:
                logger.exception(
                    "parse-failure paddle-purge raised (file %s)", file_id
                )
        else:  # stage == "index" — caller holds INGEST_LOCK
            try:
                await loop.run_in_executor(
                    None,
                    lambda: purge_file_artifacts(file_id, keep_upload=True),
                )
            except Exception:
                logger.exception(
                    "post-failure purge raised (file %s left with possible partial indexes)",
                    file_id,
                )
            # Even on failure, the global stores may carry partial rows;
            # refresh singletons so PageStore / graph channels don't keep
            # serving the half-written file.
            try:
                await loop.run_in_executor(None, _run_refresh_hook)
            except Exception:
                logger.exception(
                    "post-failure singleton refresh raised (file %s)", file_id
                )

        try:
            await _set_status(
                file_id, status="failed", error_msg=f"{type(exc).__name__}: {exc}"
            )
        except Exception:
            logger.exception("failed-state set raised; SSE will still emit error frame")
        try:
            await _close_job(job_id, ok=False, error_msg=str(exc))
        except Exception:
            logger.exception("job-close on failure raised; SSE will still emit error frame")

    final_emitted = False
    try:
        # ----- parse stage: concurrent under PARSE_SEM (bounded) -----
        async with parse_sem:
            await _start_job(job_id)
            await _set_status(file_id, status="parsing")
            try:
                parse_result = await loop.run_in_executor(
                    None,
                    lambda: parse_only(
                        source_path,
                        file_id=file_id,
                        overwrite=True,
                        on_event=_emit,
                    ),
                )
            except Exception as exc:
                logger.exception("parse failed: file_id=%s", file_id)
                await _on_failure(exc, stage="parse")
                raise

        # ----- index stage: serial under INGEST_LOCK -----
        async with INGEST_LOCK:
            await _set_status(file_id, status="indexing")
            try:
                # Builders run serially per project policy on small hosts
                # (8GB WSL OOM otherwise). The pipeline's own on_event
                # fires page_assets + 4 builder stage frames.
                result = await loop.run_in_executor(
                    None,
                    lambda: index_parsed(
                        parse_result,
                        parallel_builders=False,
                        on_event=_emit,
                        linear_config=linear_config,
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
                # Refresh long-lived singletons (PageStore / InventoryStore /
                # GraphPPRChannel) inside INGEST_LOCK before emitting the
                # SSE final frame. Without this the SSE confirms ``ready``
                # while toc / graph_explore / read_page still see empty
                # state until a backend restart. Hook is best-effort —
                # see ``_run_refresh_hook`` for the swallow rationale.
                await loop.run_in_executor(None, _run_refresh_hook)
                log_tail = "\n".join(
                    f"{r.index_name}: items={r.item_count} skipped={r.skipped_reason or '-'}"
                    for r in result.indexes
                )
                # Job-close is best-effort: a telemetry write failure
                # must NOT flip the file back to 'failed' — the indexes
                # are already correct on disk.
                try:
                    await _close_job(job_id, ok=True, log_tail=log_tail)
                except Exception:
                    logger.exception(
                        "job-close telemetry failed (file already ready): %d", job_id
                    )
                _emit(
                    "final",
                    {
                        "file_id": file_id,
                        "status": "ready",
                        "page_count": page_count,
                        "timings_ms": _aggregate_timings(stage_dones),
                        "stages": [
                            {"name": r.index_name, "items": r.item_count, "skipped_reason": r.skipped_reason}
                            for r in result.indexes
                        ],
                    },
                )
                final_emitted = True
                logger.info("ingest ok: file_id=%s pages=%s", file_id, page_count)
            except Exception as exc:
                logger.exception("ingest failed: file_id=%s", file_id)
                # Bookkeeping is best-effort here too; the SSE-final +
                # error frame are owned by the outer ``finally`` so a
                # raise from _set_status / _close_job / purge does NOT
                # convert a failure stream into a clean ``done``.
                await _on_failure(exc, stage="index")
                # Re-raise so the outer block emits final+error+done.
                raise
    except Exception as exc:
        if not final_emitted:
            try:
                _emit(
                    "final",
                    {
                        "file_id": file_id,
                        "status": "failed",
                        "timings_ms": _aggregate_timings(stage_dones),
                        "error": str(exc),
                    },
                )
            except Exception:
                logger.exception("final emit on failure raised")
        try:
            bus.close(error=str(exc), error_type=type(exc).__name__)
        except Exception:
            logger.exception("bus.close(error=...) raised")
        bus_closed = True
    finally:
        if not bus_closed:
            try:
                bus.close()
            except Exception:
                logger.exception("bus.close() raised")
        unregister_bus(job_id)


async def run_reingest(
    file_id: str,
    *,
    job_id: int,
    linear_config: Optional[Any] = None,
    parse_workers: int = 1,
) -> None:
    """Wipe-and-rebuild for a file we already have on disk.

    Requires the cached upload (``uploads/<file_id><suffix>``); if it's
    missing we fail loud rather than try to recover from paddle_ocr cache
    only — that's a separate "salvage" path the operator runs by hand.

    NB: ``purge_file_artifacts(keep_upload=True)`` — without it, we'd
    delete the source blob mid-flight and the subsequent parse step
    would fail with FileNotFoundError.

    Concurrency: same model as :func:`run_parse_index` — parse runs
    under PARSE_SEM (bounded N), purge + index run under INGEST_LOCK
    (serial). Doing purge under the lock means a fresh ingest racing a
    reingest never observes a half-empty index.

    Same SSE lifecycle as :func:`run_parse_index`: the ``finally`` is
    the single source of truth for bus close, so DB write failures on
    the error path cannot mask the SSE error frame.
    """
    loop = asyncio.get_running_loop()
    # ``replay_buffered=True`` makes the bus multi-consumer with full
    # event replay on subscribe — required so a "minimized" upload
    # dialog can be reopened later via the FilesPage chip and still
    # see the stage timeline. Disconnects no longer flip is_closed,
    # so the bg task keeps running regardless of who is watching.
    bus = EventBus(loop=loop, replay_buffered=True)
    register_bus(job_id, bus)
    bus_closed = False
    stage_dones: list[Mapping[str, Any]] = []

    def _emit(event: str, data: Mapping[str, Any]) -> None:
        if event == "stage" and data.get("phase") == "done":
            stage_dones.append(dict(data))
        bus.push(event, data)

    parse_sem = _get_parse_sem(parse_workers)

    async def _on_failure(exc: BaseException, *, stage: str) -> None:
        """Same branched cleanup as :func:`run_parse_index._on_failure`.

        Reingest's parse-stage purge only needs to drop the per-file
        paddle output (the global stores are still pristine — purge
        already ran in a previous reingest's index stage, but we have
        not yet entered this run's INGEST_LOCK). Index-stage failure
        re-purges the global stores AND refreshes singletons.
        """
        if stage == "parse":
            try:
                await loop.run_in_executor(
                    None, lambda: _purge_paddle_only(file_id)
                )
            except Exception:
                logger.exception(
                    "reingest parse-failure paddle-purge raised (file %s)", file_id
                )
        else:  # stage == "index" — caller holds INGEST_LOCK
            try:
                await loop.run_in_executor(
                    None,
                    lambda: purge_file_artifacts(file_id, keep_upload=True),
                )
            except Exception:
                logger.exception(
                    "post-failure purge raised (file %s left with possible partial indexes)",
                    file_id,
                )
            try:
                await loop.run_in_executor(None, _run_refresh_hook)
            except Exception:
                logger.exception(
                    "post-failure singleton refresh raised (file %s)", file_id
                )

        try:
            await _set_status(
                file_id, status="failed", error_msg=f"{type(exc).__name__}: {exc}"
            )
        except Exception:
            logger.exception("failed-state set raised; SSE will still emit error frame")
        try:
            await _close_job(job_id, ok=False, error_msg=str(exc))
        except Exception:
            logger.exception("job-close on failure raised; SSE will still emit error frame")

    final_emitted = False
    try:
        # ----- pre-parse purge: serial under INGEST_LOCK -----
        # The purge MUST run **before** parse: ``purge_file_artifacts``
        # deletes ``paddle_ocr/<file_id>/``, so running it after parse
        # would wipe the freshly-OCR'd ``meta.json`` and the next
        # ``build_page_assets`` would FileNotFoundError immediately.
        async with INGEST_LOCK:
            await _start_job(job_id)
            try:
                # Source lookup goes BEFORE purge so a missing row /
                # cached upload still routes through ``_on_failure``
                # — otherwise the outer except would emit SSE error
                # frames but never mark file/job failed, leaving
                # status='indexing' / job='running' forever.
                async with session_scope() as db:
                    rec = await db.get(FileRecord, file_id)
                    if rec is None:
                        raise FileNotFoundError(f"no files row: {file_id}")
                    src = upload_path(file_id, rec.suffix)
                    if not src.is_file():
                        raise FileNotFoundError(
                            f"cached upload missing: {src}; cannot re-ingest"
                        )

                _emit("stage", {"stage": "purge", "phase": "start"})
                t_purge = datetime.now(timezone.utc)
                try:
                    await loop.run_in_executor(
                        None,
                        lambda: purge_file_artifacts(file_id, keep_upload=True),
                    )
                except Exception as exc:
                    _emit(
                        "stage",
                        {
                            "stage": "purge",
                            "phase": "done",
                            "elapsed_ms": int((datetime.now(timezone.utc) - t_purge).total_seconds() * 1000),
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    raise
                _emit(
                    "stage",
                    {
                        "stage": "purge",
                        "phase": "done",
                        "elapsed_ms": int((datetime.now(timezone.utc) - t_purge).total_seconds() * 1000),
                    },
                )
                # Drop the old singletons NOW so any concurrent query
                # released between this lock and the next index lock
                # sees a consistently empty file_id.
                await loop.run_in_executor(None, _run_refresh_hook)
            except Exception as exc:
                logger.exception("reingest pre-parse purge failed: file_id=%s", file_id)
                await _on_failure(exc, stage="index")
                raise

        # ----- parse stage: concurrent under PARSE_SEM -----
        async with parse_sem:
            try:
                parse_result = await loop.run_in_executor(
                    None,
                    lambda: parse_only(
                        src,
                        file_id=file_id,
                        overwrite=True,
                        on_event=_emit,
                    ),
                )
            except Exception as exc:
                logger.exception("reingest parse failed: file_id=%s", file_id)
                await _on_failure(exc, stage="parse")
                raise

        # ----- index stage: serial under INGEST_LOCK -----
        async with INGEST_LOCK:
            try:
                # Purge already ran in the pre-parse block; builders
                # now read paddle_ocr/<file_id>/ that was just produced
                # by parse_only and write the new indexes.
                result = await loop.run_in_executor(
                    None,
                    lambda: index_parsed(
                        parse_result,
                        parallel_builders=False,
                        on_event=_emit,
                        linear_config=linear_config,
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
                # See run_parse_index for the rationale: reingest also
                # mutates the shared on-disk artifacts (faiss / graphml /
                # page_assets), so the singletons must reload before the
                # next query reads them.
                await loop.run_in_executor(None, _run_refresh_hook)
                try:
                    await _close_job(job_id, ok=True)
                except Exception:
                    logger.exception(
                        "job-close telemetry failed (file already ready): %d", job_id
                    )
                _emit(
                    "final",
                    {
                        "file_id": file_id,
                        "status": "ready",
                        "page_count": page_count,
                        "timings_ms": _aggregate_timings(stage_dones),
                        "stages": [
                            {"name": r.index_name, "items": r.item_count, "skipped_reason": r.skipped_reason}
                            for r in result.indexes
                        ],
                    },
                )
                final_emitted = True
            except Exception as exc:
                logger.exception("reingest failed: file_id=%s", file_id)
                # ``_on_failure(stage="index")`` re-purges (we already
                # purged once at the top of the lock block) so a
                # partially-rebuilt index is consistently absent from
                # every store, then refreshes singletons. Upload blob
                # is preserved so the operator can retry.
                await _on_failure(exc, stage="index")
                raise
    except Exception as exc:
        if not final_emitted:
            try:
                _emit(
                    "final",
                    {
                        "file_id": file_id,
                        "status": "failed",
                        "timings_ms": _aggregate_timings(stage_dones),
                        "error": str(exc),
                    },
                )
            except Exception:
                logger.exception("final emit on failure raised")
        try:
            bus.close(error=str(exc), error_type=type(exc).__name__)
        except Exception:
            logger.exception("bus.close(error=...) raised")
        bus_closed = True
    finally:
        if not bus_closed:
            try:
                bus.close()
            except Exception:
                logger.exception("bus.close() raised")
        unregister_bus(job_id)


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
            # Drop the deleted file's pages / sections / graph vertices
            # from the singletons. Without this a query right after delete
            # could still surface the file's old pages from the in-memory
            # PageStore even though they are gone from disk.
            await loop.run_in_executor(None, _run_refresh_hook)
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
    "register_refresh_hook",
    "unregister_refresh_hook",
]
