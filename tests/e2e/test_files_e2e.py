"""End-to-end file lifecycle checks for the FastAPI file API.

Design choices:

* The app runs in-process with ``httpx.AsyncClient`` + ``ASGITransport`` and
  FastAPI's lifespan context, so startup DB init, admin seeding, and restart
  reconciliation are exercised without spawning uvicorn.
* The fixture deletes only ``local_storage/db/app.db*`` before each run. The
  paid PaddleOCR corpus cache under ``local_storage/paddle_ocr/<file_id>/`` is
  treated as read-only.
* Cache-reuse coverage reconstructs ``ParseResult`` from ``meta.json`` and
  calls ``_ingest_one(..., builders=[])`` for two corpus files. This proves the
  route-independent cache path without calling OCR or remote embedding APIs.
* Destructive delete uses a disposable ``e2e_delete_*`` clone of one cached
  file, with probe rows in dense, visual, graph-passage, and BM25 stores. This
  lets the test assert full cleanup, including ``paddle_ocr/<id>/``, without
  deleting the real paid OCR directories.
* Full OCR upload and the current reingest route are gated behind ``RUN_OCR=1``.
  Without it, the suite still exercises the fast cache path and records the
  known route design gap with an xfail.
"""
import asyncio
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient


pytestmark = pytest.mark.anyio

RUN_OCR = os.environ.get("RUN_OCR") == "1"


@dataclass(frozen=True)
class CorpusPdf:
    filename: str
    file_id: str

    @property
    def suffix(self) -> str:
        return ".pdf"


@dataclass(frozen=True)
class DeleteProbe:
    file_id: str
    passage_vertex: str
    unique_entity_vertex: str
    shared_entity_vertex: str
    survivor_passage_vertex: str


CORPUS = (
    CorpusPdf(
        filename="AXA安盛「盛利II-至尊」保费回赠及预缴利率 截止至12月31日（英文版）.pdf",
        file_id="AXA安盛「盛利II-至尊」保费回赠及预缴利率_截止至12月31日（英文版）_4a5deaa25d7dda9a",
    ),
    CorpusPdf(
        filename="盛利2-至尊 小册子（英文版）.pdf",
        file_id="盛利2-至尊_小册子（英文版）_d2fe8002daf75382",
    ),
    CorpusPdf(
        filename="盛利2-至尊 宣传彩页（英文版）.pdf",
        file_id="盛利2-至尊_宣传彩页（英文版）_3f375b5635323f41",
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _storage_root(root: Path) -> Path:
    return root / "local_storage"


def _db_dir(root: Path) -> Path:
    return _storage_root(root) / "db"


def _clean_db_files(root: Path) -> None:
    db_dir = _db_dir(root)
    db_dir.mkdir(parents=True, exist_ok=True)
    for path in db_dir.glob("app.db*"):
        path.unlink(missing_ok=True)


async def _dispose_engine_if_loaded() -> None:
    db_mod = sys.modules.get("api.db")
    if db_mod is not None:
        async with _threadsafe_callback_heartbeat():
            await db_mod.engine.dispose()


async def _heartbeat() -> None:
    while True:
        await asyncio.sleep(0.01)


@asynccontextmanager
async def _threadsafe_callback_heartbeat() -> AsyncIterator[None]:
    """Keep the selector from sleeping forever under restricted sandboxes.

    ``aiosqlite`` completes DB work on a thread and wakes the event loop via
    ``call_soon_threadsafe``. Some local sandboxes do not wake the selector
    promptly, so this heartbeat gives the loop a short timer to come back and
    process thread-safe callbacks. On a normal event loop it is just a cheap
    no-op timer.
    """
    task = asyncio.create_task(_heartbeat())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def _force_runtime_settings() -> None:
    root = _repo_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    venv_site = (
        root
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if venv_site.is_dir() and str(venv_site) not in sys.path:
        sys.path.append(str(venv_site))
    _preload_venv_sqlalchemy(venv_site)
    settings_mod = sys.modules.get("config.settings")
    if settings_mod is not None:
        # Keep STORAGE_PATH relative — meta.json's batch_dir entries are
        # cwd-relative strings written by PdfParser, and PageAssetBuilder
        # does ``batch_dir.relative_to(output_dir)`` which only works when
        # both sides share the same absolute/relative form. Tests chdir
        # to project root in the fixture so this resolves consistently
        # with production.
        settings_mod.STORAGE_PATH = Path("./local_storage")
        settings_mod.ALLOW_INSECURE_JWT = True
    main_mod = sys.modules.get("api.main")
    if main_mod is not None:
        main_mod.ALLOW_INSECURE_JWT = True


def _preload_venv_sqlalchemy(venv_site: Path) -> None:
    """Keep SQLAlchemy and aiosqlite from the same environment.

    The bare ``pytest`` on this machine is Anaconda-based, while the app
    dependencies live in ``.venv``. Anaconda SQLAlchemy can hang when paired
    with venv ``aiosqlite``, so preload both from the project venv. We avoid
    prepending the whole venv permanently because pytest plugins may already
    have imported Anaconda Pydantic.
    """
    if not venv_site.is_dir():
        return
    loaded = sys.modules.get("sqlalchemy")
    if loaded is not None and str(venv_site) in str(getattr(loaded, "__file__", "")):
        return
    if loaded is not None:
        for name in [m for m in sys.modules if m == "sqlalchemy" or m.startswith("sqlalchemy.")]:
            sys.modules.pop(name, None)

    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(venv_site))
        importlib.import_module("aiosqlite")
        importlib.import_module("sqlalchemy")
        importlib.import_module("sqlalchemy.ext.asyncio")
    finally:
        sys.path[:] = original_path


def _import_app_module():
    _force_runtime_settings()
    return importlib.import_module("api.main")


class AppHarness:
    def __init__(self, root: Path):
        self.root = root

    @asynccontextmanager
    async def client(self) -> AsyncIterator[tuple[AsyncClient, dict[str, str]]]:
        app_mod = _import_app_module()
        app = app_mod.app
        async with _threadsafe_callback_heartbeat():
            async with app.router.lifespan_context(app):
                transport = ASGITransport(app=app, raise_app_exceptions=True)
                async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                    resp = await client.post(
                        "/auth/login",
                        data={"username": "admin", "password": "admin123"},
                    )
                    assert resp.status_code == 200, resp.text
                    token = resp.json()["access_token"]
                    yield client, {"Authorization": f"Bearer {token}"}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def app_harness(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AppHarness]:
    root = _repo_root()
    # Force cwd to project root so STORAGE_PATH=./local_storage resolves
    # consistently with the relative batch_dir entries inside meta.json.
    monkeypatch.chdir(root)
    monkeypatch.setenv("ALLOW_INSECURE_JWT", "1")
    monkeypatch.setenv("STORAGE_PATH", "./local_storage")
    monkeypatch.setenv("DEFAULT_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("DEFAULT_ADMIN_PASSWORD", "admin123")
    _force_runtime_settings()

    initial_upload_names = _snapshot_upload_names(root)
    await _dispose_engine_if_loaded()
    _clean_db_files(root)

    try:
        yield AppHarness(root)
    finally:
        await _dispose_engine_if_loaded()
        _purge_e2e_artifacts(root)
        _restore_upload_dir(root, initial_upload_names)
        _clean_db_files(root)


def _snapshot_upload_names(root: Path) -> set[str]:
    uploads = _storage_root(root) / "uploads"
    if not uploads.is_dir():
        return set()
    return {p.name for p in uploads.iterdir() if p.is_file()}


def _restore_upload_dir(root: Path, initial_names: set[str]) -> None:
    uploads = _storage_root(root) / "uploads"
    if not uploads.is_dir():
        return
    for path in uploads.iterdir():
        if path.is_file() and path.name not in initial_names:
            path.unlink(missing_ok=True)


def _purge_e2e_artifacts(root: Path) -> None:
    storage = _storage_root(root)
    candidate_ids: set[str] = set()
    for folder in (
        storage / "page_assets",
        storage / "inventory",
        storage / "inventory_atoms" / "passages",
        storage / "inventory_atoms" / "table_rows",
    ):
        if folder.is_dir():
            candidate_ids.update(p.stem for p in folder.glob("e2e_*.json"))
    paddle = storage / "paddle_ocr"
    if paddle.is_dir():
        candidate_ids.update(p.name for p in paddle.iterdir() if p.name.startswith("e2e_"))
    uploads = storage / "uploads"
    if uploads.is_dir():
        for path in uploads.iterdir():
            if path.is_file() and path.name.startswith("e2e_"):
                candidate_ids.add(path.with_suffix("").name)

    if not candidate_ids:
        return
    try:
        from ingestion.index.maintenance import purge_file_artifacts
    except Exception:
        return
    for file_id in sorted(candidate_ids):
        try:
            purge_file_artifacts(file_id)
        except Exception:
            # Teardown should not hide the real test failure. The next run still
            # cleans by prefix and the artifacts are disposable.
            pass


def _corpus_pdf_path(root: Path, item: CorpusPdf) -> Path:
    path = root / item.filename
    assert path.is_file(), f"missing test corpus PDF: {path}"
    return path


def _paddle_cache_dir(item: CorpusPdf) -> Path:
    from config.settings import paddle_ocr_root

    path = paddle_ocr_root() / item.file_id
    assert (path / "meta.json").is_file(), f"missing PaddleOCR cache: {path}"
    assert (path / "combined.md").is_file(), f"missing PaddleOCR combined.md: {path}"
    return path


def _load_parse_result(file_dir: Path):
    from ingestion.paddle_ocr.parser import BatchOutput, ParseResult

    meta_path = file_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return ParseResult(
        file_id=meta["file_id"],
        source_path=meta.get("source_path"),
        file_type=meta["file_type"],
        total_pages=meta["total_pages"],
        output_dir=str(file_dir),
        combined_markdown_path=str(file_dir / "combined.md"),
        meta_path=str(meta_path),
        batches=[BatchOutput(**b) for b in meta.get("batches", [])],
    )


def _cache_page_count(item: CorpusPdf) -> int:
    meta = json.loads((_paddle_cache_dir(item) / "meta.json").read_text(encoding="utf-8"))
    return int(meta["total_pages"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


async def _exercise_cache_reuse(item: CorpusPdf) -> None:
    from pipeline.parse_and_index import _ingest_one

    parse = _load_parse_result(_paddle_cache_dir(item))
    pages, indexes = _ingest_one(parse, [], parallel_builders=False)
    assert len(pages) == parse.total_pages
    assert indexes == []


def _copy_upload(root: Path, item: CorpusPdf) -> None:
    from config.settings import upload_path

    src = _corpus_pdf_path(root, item)
    dst = upload_path(item.file_id, item.suffix)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        assert _sha256(dst) == _sha256(src), f"existing upload differs: {dst}"
        return
    shutil.copy2(src, dst)


async def _seed_ready_corpus(root: Path, uploaded_by: int) -> None:
    from api.db import session_scope
    from api.models import FileRecord
    from api.services.files import prepare_upload

    for item in CORPUS:
        _copy_upload(root, item)

    async with session_scope() as db:
        for item in CORPUS:
            src = _corpus_pdf_path(root, item)
            blob = src.read_bytes()
            staged = prepare_upload(filename=item.filename, blob=blob)
            assert staged.file_id == item.file_id

            rec = await db.get(FileRecord, item.file_id)
            if rec is None:
                rec = FileRecord(file_id=item.file_id)
                db.add(rec)
            rec.display_name = item.filename
            rec.original_filename = item.filename
            rec.suffix = item.suffix
            rec.byte_size = staged.byte_size
            rec.sha256 = staged.sha256
            rec.page_count = _cache_page_count(item)
            rec.status = "ready"
            rec.error_msg = None
            rec.uploaded_by = uploaded_by
            rec.indexed_at = datetime.now(timezone.utc)


async def _file_row(file_id: str) -> dict[str, Any] | None:
    from api.db import session_scope
    from api.models import FileRecord

    async with session_scope() as db:
        rec = await db.get(FileRecord, file_id)
        if rec is None:
            return None
        return {
            "file_id": rec.file_id,
            "display_name": rec.display_name,
            "original_filename": rec.original_filename,
            "suffix": rec.suffix,
            "byte_size": rec.byte_size,
            "uploaded_by": rec.uploaded_by,
            "uploaded_at": rec.uploaded_at,
            "indexed_at": rec.indexed_at,
            "status": rec.status,
            "error_msg": rec.error_msg,
            "page_count": rec.page_count,
            "sha256": rec.sha256,
        }


async def _set_file_status(file_id: str, status: str, error_msg: str | None = None) -> None:
    from api.db import session_scope
    from api.models import FileRecord

    async with session_scope() as db:
        rec = await db.get(FileRecord, file_id)
        assert rec is not None
        rec.status = status
        rec.error_msg = error_msg


async def _seed_user(
    *,
    username: str,
    password: str,
    role: str,
    is_active: int = 1,
) -> int:
    from sqlalchemy import select

    from api.auth import hash_password
    from api.db import session_scope
    from api.models import User

    async with session_scope() as db:
        existing = (
            await db.execute(select(User).where(User.username == username))
        ).scalar_one_or_none()
        if existing is not None:
            existing.password_hash = hash_password(password)
            existing.role = role
            existing.is_active = is_active
            await db.flush()
            return int(existing.id)
        user = User(
            username=username,
            password_hash=hash_password(password),
            role=role,
            is_active=is_active,
        )
        db.add(user)
        await db.flush()
        return int(user.id)


async def _set_user_active(user_id: int, is_active: int) -> None:
    from api.db import session_scope
    from api.models import User

    async with session_scope() as db:
        user = await db.get(User, user_id)
        assert user is not None
        user.is_active = is_active


async def _login_headers(client: AsyncClient, username: str, password: str) -> dict[str, str]:
    resp = await client.post("/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def _file_ids_in_db() -> set[str]:
    from sqlalchemy import select

    from api.db import session_scope
    from api.models import FileRecord

    async with session_scope() as db:
        rows = await db.execute(select(FileRecord.file_id).order_by(FileRecord.file_id))
        return set(rows.scalars())


async def _audit_log_count() -> int:
    from sqlalchemy import func, select

    from api.db import session_scope
    from api.models import AuditLog

    async with session_scope() as db:
        return int((await db.execute(select(func.count()).select_from(AuditLog))).scalar_one())


async def _stale_state_counts() -> dict[str, int]:
    from sqlalchemy import func, select

    from api.db import session_scope
    from api.models import FileRecord, IngestJob

    async with session_scope() as db:
        file_count = (
            await db.execute(
                select(func.count())
                .select_from(FileRecord)
                .where(FileRecord.status.in_(("pending", "parsing", "indexing", "deleting")))
            )
        ).scalar_one()
        job_count = (
            await db.execute(
                select(func.count())
                .select_from(IngestJob)
                .where(IngestJob.status.in_(("pending", "running")))
            )
        ).scalar_one()
        return {"files": int(file_count), "jobs": int(job_count)}


async def _create_ingest_job(
    *,
    file_id: str,
    kind: str,
    status: str,
) -> int:
    from api.db import session_scope
    from api.models import IngestJob

    async with session_scope() as db:
        job = IngestJob(
            file_id=file_id,
            kind=kind,
            status=status,
            started_at=datetime.now(timezone.utc) if status == "running" else None,
        )
        db.add(job)
        await db.flush()
        return int(job.id)


async def _ingest_job_rows(file_id: str, *, kind: str | None = None) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from api.db import session_scope
    from api.models import IngestJob

    async with session_scope() as db:
        stmt = select(IngestJob).where(IngestJob.file_id == file_id).order_by(IngestJob.id)
        if kind is not None:
            stmt = stmt.where(IngestJob.kind == kind)
        rows = (await db.execute(stmt)).scalars().all()
        return [
            {
                "id": row.id,
                "file_id": row.file_id,
                "kind": row.kind,
                "status": row.status,
                "started_at": row.started_at,
                "finished_at": row.finished_at,
                "error_msg": row.error_msg,
                "log_tail": row.log_tail,
                "created_at": row.created_at,
            }
            for row in rows
        ]


async def _finish_ingest_job(job_id: int, *, file_id: str, status: str = "ready") -> None:
    from api.db import session_scope
    from api.models import FileRecord, IngestJob

    async with session_scope() as db:
        rec = await db.get(FileRecord, file_id)
        assert rec is not None
        rec.status = status
        job = await db.get(IngestJob, job_id)
        assert job is not None
        job.status = "done"
        job.finished_at = datetime.now(timezone.utc)


def _uploads_for_file_id(file_id: str) -> list[Path]:
    from config.settings import uploads_root

    root = uploads_root()
    if not root.exists():
        return []
    prefix = f"{file_id}."
    return [
        p
        for p in root.iterdir()
        if p.is_file() and (p.name == file_id or p.name.startswith(prefix))
    ]


def _assert_artifacts_present(file_id: str) -> None:
    from config.settings import (
        inventory_atoms_root,
        inventory_path,
        page_assets_path,
        paddle_ocr_root,
    )

    assert page_assets_path(file_id).is_file()
    assert inventory_path(file_id).is_file()
    assert (inventory_atoms_root("passages") / f"{file_id}.json").is_file()
    assert (inventory_atoms_root("table_rows") / f"{file_id}.json").is_file()
    assert _uploads_for_file_id(file_id)
    assert (paddle_ocr_root() / file_id).is_dir()


def _assert_artifacts_absent(file_id: str) -> None:
    from config.settings import (
        inventory_atoms_root,
        inventory_path,
        page_assets_path,
        paddle_ocr_root,
    )

    assert not page_assets_path(file_id).exists()
    assert not inventory_path(file_id).exists()
    assert not (inventory_atoms_root("passages") / f"{file_id}.json").exists()
    assert not (inventory_atoms_root("table_rows") / f"{file_id}.json").exists()
    assert not _uploads_for_file_id(file_id)
    assert not (paddle_ocr_root() / file_id).exists()


def _bm25_count(file_id: str) -> int:
    from config.settings import bm25_root

    meta_path = bm25_root() / "meta.json"
    if not meta_path.is_file():
        return 0
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return int((meta.get("file_counts") or {}).get(file_id, 0))


def _store_file_count(store_dir: Path, namespace: str, file_id: str) -> int:
    from storage import EmbeddingStore

    store = EmbeddingStore(store_dir, namespace=namespace)
    return sum(1 for fid in store.meta_column("file_id") if fid == file_id)


def _index_counts(file_id: str) -> dict[str, int]:
    from config.settings import faiss_dense_dir, faiss_graph_passage_dir, faiss_visual_dir

    return {
        "dense": _store_file_count(faiss_dense_dir(), "dense", file_id),
        "visual": _store_file_count(faiss_visual_dir(), "visual", file_id),
        "graph_passage": _store_file_count(faiss_graph_passage_dir(), "passage", file_id),
        "bm25": _bm25_count(file_id),
    }


def _assert_no_index_rows(file_id: str) -> None:
    assert _index_counts(file_id) == {
        "dense": 0,
        "visual": 0,
        "graph_passage": 0,
        "bm25": 0,
    }


def _stable_unit_vector(label: str, dim: int) -> np.ndarray:
    raw = hashlib.sha256(label.encode("utf-8")).digest()
    values = np.frombuffer((raw * ((dim // len(raw)) + 1))[:dim], dtype=np.uint8).astype(
        np.float32
    )
    values = values - values.mean()
    norm = np.linalg.norm(values)
    if norm == 0:
        values[0] = 1.0
        norm = 1.0
    return values / norm


def _add_probe_embedding_row(store_dir: Path, namespace: str, file_id: str) -> str:
    from storage import EmbeddingStore

    store = EmbeddingStore(store_dir, namespace=namespace)
    dim = int(store.dim or 8)
    text = f"e2e probe {namespace} {file_id}"
    hash_id = store.hash_for(text)
    if store.has(hash_id):
        return hash_id
    extra: dict[str, list[Any]] = {"file_id": [file_id], "page_id": ["p_0001"]}
    if namespace == "visual":
        extra["image_path"] = [f"local_storage/paddle_ocr/{file_id}/raw/probe.jpg"]
    if namespace == "passage":
        extra["page_number"] = [1]
    store.add(
        [hash_id],
        [text],
        np.asarray([_stable_unit_vector(f"{namespace}:{file_id}", dim)], dtype=np.float32),
        extra_metadata=extra,
    )
    # add() leaves the store dirty in-memory only; the delete-probe path
    # below opens a fresh EmbeddingStore that re-reads from disk, so we
    # must persist before returning.
    store.save()
    return hash_id


def _graphml_path() -> Path:
    from config.settings import faiss_graph_dir

    return faiss_graph_dir() / "LinearRAG.graphml"


def _read_graphml():
    import igraph as ig

    path = _graphml_path()
    assert path.is_file(), f"missing LinearRAG graphml: {path.resolve()}"
    return ig.Graph.Read_GraphML(str(path))


def _graph_names() -> set[str]:
    graph = _read_graphml()
    return {v["name"] for v in graph.vs if "name" in v.attributes()}


def _graph_counts() -> tuple[int, int]:
    graph = _read_graphml()
    return graph.vcount(), graph.ecount()


def _first_surviving_passage_vertex(excluded_file_id: str) -> str:
    graph = _read_graphml()
    for vertex in graph.vs:
        attrs = vertex.attributes()
        name = attrs.get("name")
        if not isinstance(name, str) or excluded_file_id in name:
            continue
        if attrs.get("vertex_type") == "passage" or name.startswith("passage-"):
            return name
    raise AssertionError("LinearRAG graph has no surviving passage vertex to share")


def _add_graph_delete_probe(file_id: str, passage_vertex: str) -> DeleteProbe:
    graph = _read_graphml()
    existing = {v["name"] for v in graph.vs if "name" in v.attributes()}
    survivor = _first_surviving_passage_vertex(file_id)
    unique_entity = f"entity-e2e-unique-{hashlib.sha256(file_id.encode()).hexdigest()[:16]}"
    shared_entity = f"entity-e2e-shared-{hashlib.sha256(file_id.encode()).hexdigest()[:16]}"

    for name, vertex_type, content in (
        (passage_vertex, "passage", f"delete-only probe passage for {file_id}"),
        (unique_entity, "entity", f"delete-only entity for {file_id}"),
        (shared_entity, "entity", f"shared entity for {file_id}"),
    ):
        if name not in existing:
            graph.add_vertex(name=name, vertex_type=vertex_type, content=content)
            existing.add(name)

    name_to_idx = {v["name"]: v.index for v in graph.vs if "name" in v.attributes()}
    for left, right in (
        (unique_entity, passage_vertex),
        (shared_entity, passage_vertex),
        (shared_entity, survivor),
    ):
        if graph.get_eid(name_to_idx[left], name_to_idx[right], error=False) == -1:
            graph.add_edge(
                name_to_idx[left],
                name_to_idx[right],
                edge_type="entity_passage",
                weight=1.0,
            )

    if "id" in graph.vs.attributes():
        del graph.vs["id"]
    graph.write_graphml(str(_graphml_path()))
    return DeleteProbe(
        file_id=file_id,
        passage_vertex=passage_vertex,
        unique_entity_vertex=unique_entity,
        shared_entity_vertex=shared_entity,
        survivor_passage_vertex=survivor,
    )


def _replace_file_id(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_file_id(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _replace_file_id(v, old, new) for k, v in value.items()}
    return value


def _copy_json_artifact(src: Path, dst: Path, old_file_id: str, new_file_id: str) -> None:
    assert src.is_file(), f"missing source artifact: {src}"
    data = json.loads(src.read_text(encoding="utf-8"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        json.dumps(_replace_file_id(data, old_file_id, new_file_id), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def _seed_disposable_delete_target(root: Path, source: CorpusPdf) -> DeleteProbe:
    from api.db import session_scope
    from api.models import FileRecord
    from config.settings import (
        faiss_dense_dir,
        faiss_graph_passage_dir,
        faiss_visual_dir,
        inventory_atoms_root,
        inventory_path,
        page_assets_path,
        paddle_ocr_root,
        upload_path,
    )
    from ingestion.index.bm25_tantivy import BM25IndexBuilder
    from storage.page_store import PageAsset

    file_id = f"e2e_delete_{hashlib.sha256(source.file_id.encode('utf-8')).hexdigest()[:16]}"
    src_pdf = _corpus_pdf_path(root, source)
    dst_upload = upload_path(file_id, source.suffix)
    dst_upload.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_pdf, dst_upload)

    src_cache = _paddle_cache_dir(source)
    dst_cache = paddle_ocr_root() / file_id
    if dst_cache.exists():
        shutil.rmtree(dst_cache)
    shutil.copytree(src_cache, dst_cache)

    _copy_json_artifact(
        page_assets_path(source.file_id),
        page_assets_path(file_id),
        source.file_id,
        file_id,
    )
    _copy_json_artifact(
        inventory_path(source.file_id),
        inventory_path(file_id),
        source.file_id,
        file_id,
    )
    _copy_json_artifact(
        inventory_atoms_root("passages") / f"{source.file_id}.json",
        inventory_atoms_root("passages") / f"{file_id}.json",
        source.file_id,
        file_id,
    )
    _copy_json_artifact(
        inventory_atoms_root("table_rows") / f"{source.file_id}.json",
        inventory_atoms_root("table_rows") / f"{file_id}.json",
        source.file_id,
        file_id,
    )

    pages_json = json.loads(page_assets_path(file_id).read_text(encoding="utf-8"))
    BM25IndexBuilder().build(file_id, [PageAsset.from_dict(p) for p in pages_json])
    _add_probe_embedding_row(faiss_dense_dir(), "dense", file_id)
    _add_probe_embedding_row(faiss_visual_dir(), "visual", file_id)
    passage_vertex = _add_probe_embedding_row(faiss_graph_passage_dir(), "passage", file_id)
    graph_probe = _add_graph_delete_probe(file_id, passage_vertex)

    async with session_scope() as db:
        rec = await db.get(FileRecord, file_id)
        if rec is None:
            rec = FileRecord(file_id=file_id)
            db.add(rec)
        rec.display_name = f"e2e delete probe for {source.filename}"
        rec.original_filename = source.filename
        rec.suffix = source.suffix
        rec.byte_size = src_pdf.stat().st_size
        rec.sha256 = _sha256(src_pdf)
        rec.page_count = _cache_page_count(source)
        rec.status = "ready"
        rec.error_msg = None
        rec.indexed_at = datetime.now(timezone.utc)

    assert all(count > 0 for count in _index_counts(file_id).values())
    _assert_artifacts_present(file_id)
    return graph_probe


def _fingerprint_path(path: Path) -> tuple[Any, ...]:
    resolved = path.resolve()
    if path.is_file():
        return ("file", str(resolved), path.stat().st_size, _sha256(path))
    if path.is_dir():
        rows = []
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            rows.append(
                (
                    child.relative_to(path).as_posix(),
                    str(child.resolve()),
                    child.stat().st_size,
                    _sha256(child),
                )
            )
        return ("dir", str(resolved), tuple(rows))
    return ("missing", str(resolved))


def _normalize_snapshot_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _normalize_snapshot_value(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(k): _normalize_snapshot_value(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize_snapshot_value(v) for v in value]
    return value


def _embedding_meta_rows(store_dir: Path, namespace: str, file_id: str) -> list[dict[str, Any]]:
    from storage import EmbeddingStore

    store = EmbeddingStore(store_dir, namespace=namespace)
    if "file_id" not in store._meta.columns:
        return []
    rows = store._meta.loc[store._meta["file_id"] == file_id]
    if "hash_id" in rows.columns:
        rows = rows.sort_values("hash_id")
    return [
        {str(k): _normalize_snapshot_value(v) for k, v in row.items()}
        for row in rows.to_dict(orient="records")
    ]


def _index_meta_snapshot(file_id: str) -> dict[str, list[dict[str, Any]]]:
    from config.settings import faiss_dense_dir, faiss_graph_passage_dir, faiss_visual_dir

    return {
        "dense": _embedding_meta_rows(faiss_dense_dir(), "dense", file_id),
        "visual": _embedding_meta_rows(faiss_visual_dir(), "visual", file_id),
        "graph_passage": _embedding_meta_rows(
            faiss_graph_passage_dir(), "passage", file_id
        ),
    }


def _paddle_meta_fingerprint(item: CorpusPdf) -> tuple[str, int, str]:
    path = _paddle_cache_dir(item) / "meta.json"
    return (str(path.resolve()), path.stat().st_mtime_ns, _sha256(path))


def _paddle_meta_fingerprints(items: Sequence[CorpusPdf]) -> dict[str, tuple[str, int, str]]:
    return {item.file_id: _paddle_meta_fingerprint(item) for item in items}


async def _snapshot_file_state(file_ids: list[str]) -> dict[str, dict[str, Any]]:
    from config.settings import (
        inventory_atoms_root,
        inventory_path,
        page_assets_path,
        paddle_ocr_root,
    )

    out: dict[str, dict[str, Any]] = {}
    for file_id in file_ids:
        artifacts = {
            "page_assets": _fingerprint_path(page_assets_path(file_id)),
            "inventory": _fingerprint_path(inventory_path(file_id)),
            "passage_atoms": _fingerprint_path(
                inventory_atoms_root("passages") / f"{file_id}.json"
            ),
            "table_row_atoms": _fingerprint_path(
                inventory_atoms_root("table_rows") / f"{file_id}.json"
            ),
            "paddle_ocr": _fingerprint_path(paddle_ocr_root() / file_id),
            "uploads": tuple(
                (p.name, p.stat().st_size, _sha256(p))
                for p in sorted(_uploads_for_file_id(file_id))
            ),
        }
        row = await _file_row(file_id)
        assert row is not None
        out[file_id] = {
            "artifacts": artifacts,
            "db_row": _normalize_snapshot_value(row),
            "index_counts": _index_counts(file_id),
            "index_meta": _index_meta_snapshot(file_id),
        }
    return out


def _terminal_write(config: pytest.Config, message: str) -> None:
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(message)
    else:
        print(message, flush=True)


async def _assert_strict_teardown_state(
    *,
    expected_ids: set[str],
    config: pytest.Config,
) -> None:
    from ingestion.index.maintenance import indexed_file_ids

    indexed_ids = indexed_file_ids()
    db_ids = await _file_ids_in_db()
    _terminal_write(
        config,
        f"strict file state: indexed={len(indexed_ids)} db={len(db_ids)} "
        f"expected={len(expected_ids)}",
    )
    assert indexed_ids == expected_ids
    assert db_ids == expected_ids


async def _poll_file_status(
    client: AsyncClient,
    headers: dict[str, str],
    file_id: str,
    expected: str,
    *,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last: dict[str, Any] | None = None
    while asyncio.get_running_loop().time() < deadline:
        resp = await client.get(f"/files/{file_id}", headers=headers)
        if resp.status_code == 200:
            last = resp.json()
            if last["status"] == expected:
                return last
        await asyncio.sleep(0.2)
    raise AssertionError(f"{file_id} did not reach {expected}; last={last}")


async def _poll_file_deleted(
    client: AsyncClient,
    headers: dict[str, str],
    file_id: str,
    *,
    timeout_s: float = 10.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_status = None
    while asyncio.get_running_loop().time() < deadline:
        resp = await client.get(f"/files/{file_id}", headers=headers)
        last_status = resp.status_code
        if resp.status_code == 404:
            return
        await asyncio.sleep(0.2)
    raise AssertionError(f"{file_id} was not deleted; last_status={last_status}")


async def test_files_lifecycle_fast_e2e(
    app_harness: AppHarness,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    async with app_harness.client() as (client, headers):
        me = await client.get("/auth/me", headers=headers)
        assert me.status_code == 200
        assert me.json()["username"] == "admin"
        assert me.json()["role"] == "admin"
        admin_id = int(me.json()["id"])

        wrong_password = await client.post(
            "/auth/login",
            data={"username": "admin", "password": "wrong-password"},
        )
        unknown_user = await client.post(
            "/auth/login",
            data={"username": "missing-e2e-user", "password": "wrong-password"},
        )
        assert wrong_password.status_code == 401
        assert unknown_user.status_code == 401
        assert wrong_password.json()["detail"] == "invalid username or password"
        assert unknown_user.json()["detail"] == wrong_password.json()["detail"]

        await _seed_user(username="e2e_analyst", password="analyst123", role="analyst")
        analyst_headers = await _login_headers(client, "e2e_analyst", "analyst123")
        disabled_id = await _seed_user(
            username="e2e_disabled",
            password="disabled123",
            role="analyst",
        )
        disabled_headers = await _login_headers(client, "e2e_disabled", "disabled123")
        await _set_user_active(disabled_id, 0)
        disabled_me = await client.get("/auth/me", headers=disabled_headers)
        assert disabled_me.status_code == 401

        cold_files = await client.get("/files", headers=headers)
        assert cold_files.status_code == 200
        assert cold_files.json() == []

        # Cache reuse for the two non-OCR corpus files. This rebuilds page,
        # inventory, and atom JSONs from paid PaddleOCR cache without OCR.
        cache_reuse_items = (CORPUS[1], CORPUS[2])
        corpus_meta_before = _paddle_meta_fingerprints(CORPUS)
        cache_meta_before = _paddle_meta_fingerprints(cache_reuse_items)
        await _exercise_cache_reuse(CORPUS[1])
        await _exercise_cache_reuse(CORPUS[2])
        await _seed_ready_corpus(app_harness.root, uploaded_by=admin_id)

        listed = await client.get("/files", headers=headers)
        assert listed.status_code == 200
        rows = listed.json()
        assert {row["file_id"] for row in rows} == {item.file_id for item in CORPUS}
        assert all(row["status"] == "ready" for row in rows)
        for item in CORPUS:
            row = await _file_row(item.file_id)
            assert row is not None
            assert row["status"] == "ready"
            assert row["page_count"] == _cache_page_count(item)
            _assert_artifacts_present(item.file_id)

        auth_probe = CORPUS[0]
        auth_blob = _corpus_pdf_path(app_harness.root, auth_probe).read_bytes()

        def auth_upload_parts() -> dict[str, tuple[str, bytes, str]]:
            return {"file": (f"e2e_auth_{auth_probe.filename}", auth_blob, "application/pdf")}

        no_token_upload = await client.post("/files", files=auth_upload_parts())
        no_token_reingest = await client.post(f"/files/{auth_probe.file_id}/reingest")
        no_token_delete = await client.delete(f"/files/{auth_probe.file_id}")
        assert no_token_upload.status_code == 401
        assert no_token_reingest.status_code == 401
        assert no_token_delete.status_code == 401

        analyst_upload = await client.post(
            "/files",
            headers=analyst_headers,
            files=auth_upload_parts(),
        )
        analyst_reingest = await client.post(
            f"/files/{auth_probe.file_id}/reingest",
            headers=analyst_headers,
        )
        analyst_delete = await client.delete(
            f"/files/{auth_probe.file_id}",
            headers=analyst_headers,
        )
        assert analyst_upload.status_code == 403
        assert analyst_reingest.status_code == 403
        assert analyst_delete.status_code == 403

        pending_guard_id = CORPUS[0].file_id
        await _set_file_status(pending_guard_id, "pending")
        pending_reingest = await client.post(
            f"/files/{pending_guard_id}/reingest",
            headers=headers,
        )
        assert pending_reingest.status_code == 409
        pending_row = await _file_row(pending_guard_id)
        assert pending_row is not None
        assert pending_row["status"] == "pending"
        await _set_file_status(pending_guard_id, "ready")

        indexing_guard_id = CORPUS[0].file_id
        await _set_file_status(indexing_guard_id, "indexing")
        indexing_delete = await client.delete(f"/files/{indexing_guard_id}", headers=headers)
        assert indexing_delete.status_code == 409
        indexing_row = await _file_row(indexing_guard_id)
        assert indexing_row is not None
        assert indexing_row["status"] == "indexing"
        await _set_file_status(indexing_guard_id, "ready")

        dup_item = CORPUS[0]
        dup_resp = await client.post(
            "/files",
            headers=headers,
            files={
                "file": (
                    dup_item.filename,
                    _corpus_pdf_path(app_harness.root, dup_item).read_bytes(),
                    "application/pdf",
                )
            },
        )
        assert dup_resp.status_code == 409
        assert (await _file_row(dup_item.file_id))["status"] == "ready"  # type: ignore[index]
        _assert_artifacts_present(dup_item.file_id)

        # Proof of cache-reingest capability while the route still lacks the
        # fast path: another direct _ingest_one call should leave API-visible
        # rows ready without OCR.
        await _exercise_cache_reuse(CORPUS[1])
        assert _paddle_meta_fingerprints(cache_reuse_items) == cache_meta_before
        assert (await client.get(f"/files/{CORPUS[1].file_id}", headers=headers)).json()[
            "status"
        ] == "ready"

        protected_ids = [CORPUS[1].file_id, CORPUS[2].file_id]
        delete_probe = await _seed_disposable_delete_target(app_harness.root, CORPUS[0])
        delete_id = delete_probe.file_id
        protected_before = await _snapshot_file_state(protected_ids)
        graph_counts_before = _graph_counts()
        graph_names_before = _graph_names()
        assert delete_probe.passage_vertex in graph_names_before
        assert delete_probe.unique_entity_vertex in graph_names_before
        assert delete_probe.shared_entity_vertex in graph_names_before
        assert delete_probe.survivor_passage_vertex in graph_names_before
        delete_resp = await client.delete(f"/files/{delete_id}", headers=headers)
        assert delete_resp.status_code == 202, delete_resp.text
        assert delete_resp.json()["status"] == "deleting"
        await _poll_file_deleted(client, headers, delete_id)
        assert await _file_row(delete_id) is None
        _assert_artifacts_absent(delete_id)
        _assert_no_index_rows(delete_id)
        assert await _snapshot_file_state(protected_ids) == protected_before
        graph_counts_after = _graph_counts()
        graph_names_after = _graph_names()
        assert graph_counts_after[0] < graph_counts_before[0]
        assert graph_counts_after[1] < graph_counts_before[1]
        assert delete_probe.passage_vertex not in graph_names_after
        assert delete_probe.unique_entity_vertex not in graph_names_after
        assert delete_probe.shared_entity_vertex in graph_names_after
        assert delete_probe.survivor_passage_vertex in graph_names_after

        restart_id = CORPUS[1].file_id
        await _set_file_status(restart_id, "indexing")
        restart_job_id = await _create_ingest_job(
            file_id=restart_id,
            kind="reingest",
            status="running",
        )

    async with app_harness.client() as (client, headers):
        restarted = await client.get(f"/files/{restart_id}", headers=headers)
        assert restarted.status_code == 200
        assert restarted.json()["status"] == "failed"
        assert restarted.json()["error_msg"] == "process restarted mid-job"
        row = await _file_row(restart_id)
        assert row is not None
        assert row["status"] == "failed"
        assert row["error_msg"] == "process restarted mid-job"
        restart_jobs = [
            row for row in await _ingest_job_rows(restart_id) if row["id"] == restart_job_id
        ]
        assert len(restart_jobs) == 1
        assert restart_jobs[0]["status"] == "failed"
        assert restart_jobs[0]["error_msg"] == "process restarted mid-job"
        assert restart_jobs[0]["finished_at"] is not None

    clean_audit_before = await _audit_log_count()
    clean_counts_before = await _stale_state_counts()
    async with app_harness.client() as (_client, _headers):
        assert await _stale_state_counts() == {"files": 0, "jobs": 0}
    assert clean_counts_before == {"files": 0, "jobs": 0}
    assert await _stale_state_counts() == clean_counts_before
    assert await _audit_log_count() == clean_audit_before

    async with app_harness.client() as (client, headers):
        cas_id = CORPUS[2].file_id
        await _set_file_status(cas_id, "ready")

        from api.routes import files as files_routes

        async def fake_run_reingest(file_id: str, *, job_id: int) -> None:
            assert file_id == cas_id
            assert job_id > 0
            await asyncio.sleep(0.3)
            await _finish_ingest_job(job_id, file_id=file_id)

        monkeypatch.setattr(files_routes, "run_reingest", fake_run_reingest)
        responses = await asyncio.gather(
            *[
                client.post(f"/files/{cas_id}/reingest", headers=headers)
                for _ in range(5)
            ]
        )
        status_codes = sorted(resp.status_code for resp in responses)
        assert status_codes.count(202) == 1
        assert status_codes.count(409) == 4
        cas_jobs = await _ingest_job_rows(cas_id, kind="reingest")
        assert len(cas_jobs) == 1
        await _poll_file_status(client, headers, cas_id, "ready")
        row = await _file_row(cas_id)
        assert row is not None
        assert row["status"] == "ready"
        _assert_artifacts_present(cas_id)
        assert _paddle_meta_fingerprints(CORPUS) == corpus_meta_before
        await _assert_strict_teardown_state(
            expected_ids={item.file_id for item in CORPUS},
            config=request.config,
        )


async def test_identical_upload_race_preserves_winner_blob(
    app_harness: AppHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api.routes import files as files_routes
    from api.services.files import prepare_upload

    filename = "e2e_race_same.pdf"
    payload = b"%PDF-1.4\n% e2e identical upload race\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
    staged = prepare_upload(filename=filename, blob=payload)
    original_create_pending_record = files_routes.create_pending_record
    create_calls = 0
    create_lock = asyncio.Lock()
    both_at_insert = asyncio.Event()

    async def racing_create_pending_record(*args: Any, **kwargs: Any):
        nonlocal create_calls
        if kwargs.get("staged") == staged:
            async with create_lock:
                create_calls += 1
                if create_calls == 2:
                    both_at_insert.set()
            await asyncio.wait_for(both_at_insert.wait(), timeout=2.0)
        return await original_create_pending_record(*args, **kwargs)

    async def fake_run_parse_index(file_id: str, source_path: Path, *, job_id: int) -> None:
        assert file_id == staged.file_id
        assert source_path.resolve() == staged.path.resolve()
        assert job_id > 0

    monkeypatch.setattr(files_routes, "create_pending_record", racing_create_pending_record)
    monkeypatch.setattr(files_routes, "run_parse_index", fake_run_parse_index)

    def upload_parts() -> dict[str, tuple[str, bytes, str]]:
        return {"file": (filename, payload, "application/pdf")}

    async with app_harness.client() as (client, headers):
        responses = await asyncio.gather(
            client.post("/files", headers=headers, files=upload_parts()),
            client.post("/files", headers=headers, files=upload_parts()),
        )

    status_codes = sorted(resp.status_code for resp in responses)
    assert status_codes == [202, 409]
    row = await _file_row(staged.file_id)
    assert row is not None
    assert (await _file_ids_in_db()) == {staged.file_id}
    jobs = await _ingest_job_rows(staged.file_id)
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "parse_index"
    blobs = _uploads_for_file_id(staged.file_id)
    assert len(blobs) == 1
    assert blobs[0].resolve() == staged.path.resolve()
    assert _sha256(blobs[0]) == hashlib.sha256(payload).hexdigest()


@pytest.mark.skipif(
    not os.environ.get("RUN_OCR"),
    reason="RUN_OCR required for paid full OCR upload",
)
async def test_full_upload_fresh_pdf_path_when_enabled(app_harness: AppHarness) -> None:
    from config.settings import upload_path

    item = CORPUS[0]
    filename = f"e2e_full_ocr_{item.filename}"
    source_pdf = _corpus_pdf_path(app_harness.root, item)
    source_blob = source_pdf.read_bytes()
    async with app_harness.client() as (client, headers):
        resp = await client.post(
            "/files",
            headers=headers,
            files={
                "file": (
                    filename,
                    source_blob,
                    "application/pdf",
                )
            },
        )
        assert resp.status_code == 202, resp.text
        file_id = resp.json()["file_id"]
        ready = await _poll_file_status(client, headers, file_id, "ready", timeout_s=300.0)
        assert ready["page_count"] == _cache_page_count(item)
        assert (await _file_row(file_id))["status"] == "ready"  # type: ignore[index]
        assert _sha256(upload_path(file_id, ".pdf")) == _sha256(source_pdf)
        _assert_artifacts_present(file_id)

        cleanup = await client.delete(f"/files/{file_id}", headers=headers)
        assert cleanup.status_code == 202
        await _poll_file_deleted(client, headers, file_id, timeout_s=60.0)
        _assert_artifacts_absent(file_id)
        _assert_no_index_rows(file_id)


@pytest.mark.xfail(
    not RUN_OCR,
    reason=(
        "reingest route currently calls parse_and_index(overwrite=True) "
        "and does not expose OCR-cache reuse"
    ),
    strict=True,
)
async def test_reingest_route_cache_reuse_gap(app_harness: AppHarness) -> None:
    if not RUN_OCR:
        raise AssertionError("route cache-reuse path is intentionally not exposed yet")

    item = CORPUS[0]
    async with app_harness.client() as (client, headers):
        admin = await client.get("/auth/me", headers=headers)
        await _seed_ready_corpus(app_harness.root, uploaded_by=int(admin.json()["id"]))
        resp = await client.post(f"/files/{item.file_id}/reingest", headers=headers)
        assert resp.status_code == 202, resp.text
        ready = await _poll_file_status(client, headers, item.file_id, "ready", timeout_s=300.0)
        assert ready["page_count"] == _cache_page_count(item)


# ---------------------------------------------------------------------------
# Regression tests for the post-review hardening pass
# ---------------------------------------------------------------------------
# The three tests below cover production-code behavior changes that the
# original suite did not exercise:
#
#   * builder failure must mark the file failed (not ready) and purge
#     any partial rows the surviving builders managed to write
#     (parse_and_index returns ok=False; run_parse_index reacts).
#   * delete must persist purge counts to AuditLog before the cascade
#     removes the IngestJob row, so a successful delete is still
#     auditable after the file row is gone.
#   * exact-suffix purge must NOT cross-delete an unrelated file whose
#     name happens to share a prefix with the target file_id.
# ---------------------------------------------------------------------------


async def test_builder_failure_marks_file_failed_and_purges_partial(
    app_harness: AppHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A builder raising mid-ingest must:
    1) make ``parse_and_index`` return ``ok=False`` (failed flag wired),
    2) flip ``files.status`` to 'failed' (NOT 'ready'),
    3) leave NO rows tagged with that file_id in any of the four indexes
       (post-failure ``purge_file_artifacts`` call), AND
    4) preserve the cached upload blob so the operator can retry.
    """
    from api.services import files as files_service
    from config.settings import upload_path
    import ingestion.index.bm25_tantivy as bm25_module

    item = CORPUS[1]
    fake_file_id = "e2e_builder_fail_target"

    real_index_parsed = files_service.index_parsed

    def fake_index_parsed(parse_result, *args: Any, **kwargs: Any):
        # Mirror the real call's return shape but force one builder to
        # fail. We don't actually run paddle / dense / graph — too slow
        # for a unit-style test. We synthesize a PipelineResult that
        # exercises the (failed=True → ok=False) wiring AND drops a
        # bm25 row tagged with file_id so the post-failure purge has
        # something concrete to clean up. ``parse_only`` has already
        # run upstream by the time the route hands us ``parse_result``;
        # we ignore its content here and return a synthetic failure.
        from ingestion.index.base import IndexBuildResult
        from ingestion.paddle_ocr.parser import ParseResult
        from pipeline.parse_and_index import PipelineResult

        # Simulate ONE successful builder having written rows (bm25 row
        # injection — simplest sentinel; same idea as the existing
        # delete-probe row helper).
        bm25_module.BM25IndexBuilder().build  # ensure import side-effects ok
        from config.settings import bm25_root
        bm25_dir = bm25_root()
        bm25_dir.mkdir(parents=True, exist_ok=True)
        meta_path = bm25_dir / "meta.json"
        existing = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        )
        file_counts = dict(existing.get("file_counts", {}))
        file_counts[fake_file_id] = 7
        meta_path.write_text(
            json.dumps(
                {"fields": ["file_id", "page_id", "text"], "file_counts": file_counts},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        synthetic_parse = parse_result if parse_result is not None else ParseResult(
            file_id=fake_file_id,
            source_path="",
            file_type=0,
            total_pages=0,
            output_dir="",
            combined_markdown_path="",
            meta_path="",
            batches=[],
        )
        return PipelineResult(
            parse=synthetic_parse,
            pages=[],
            indexes=[
                IndexBuildResult(
                    index_name="bm25",
                    file_id=fake_file_id,
                    output_dir=str(bm25_dir),
                    item_count=7,
                ),
                IndexBuildResult(
                    index_name="text_dense",
                    file_id=fake_file_id,
                    output_dir="",
                    skipped_reason="build raised: RuntimeError: simulated dense failure",
                    failed=True,
                ),
            ],
            total_seconds=0.01,
            source=getattr(synthetic_parse, "source_path", ""),
            ok=False,
            error="builder(s) failed: text_dense (build raised: RuntimeError: simulated dense failure)",
        )

    monkeypatch.setattr(files_service, "index_parsed", fake_index_parsed)

    # Use the API end-to-end so the route + bg task wiring is also tested.
    blob = _corpus_pdf_path(app_harness.root, item).read_bytes()
    # Override the deterministic file_id derivation so we don't collide
    # with the real corpus file. We do this by monkeypatching
    # _derive_file_id to a fixed sentinel.
    monkeypatch.setattr(
        files_service, "_derive_file_id",
        lambda filename, sha256_hex: fake_file_id,
    )

    async with app_harness.client() as (client, headers):
        resp = await client.post(
            "/files",
            headers=headers,
            files={"file": ("simulated.pdf", blob, "application/pdf")},
        )
        assert resp.status_code == 202, resp.text
        # Bg task should set status='failed' (not 'ready').
        failed = await _poll_file_status(client, headers, fake_file_id, "failed", timeout_s=15.0)
        assert "text_dense" in (failed.get("error_msg") or "")

        # Upload blob preserved for retry.
        kept = upload_path(fake_file_id, ".pdf")
        assert kept.is_file(), "post-failure purge must NOT delete the upload blob"

        # No partial rows remain in any index store.
        _assert_no_index_rows(fake_file_id)

        # Cleanup the fake file row + upload via DELETE so teardown is clean.
        await client.delete(f"/files/{fake_file_id}", headers=headers)
        await _poll_file_deleted(client, headers, fake_file_id, timeout_s=15.0)
        assert not kept.exists()


async def test_delete_persists_audit_counts_before_cascade(
    app_harness: AppHarness,
) -> None:
    """``run_delete`` writes ``AuditLog(action='file.delete.complete', payload_json=counts)``
    BEFORE the FileRecord delete cascades the IngestJob row away, so the
    counts survive even though the IngestJob row does not."""
    from api.db import session_scope
    from api.models import AuditLog
    from sqlalchemy import select as sa_select

    item = CORPUS[2]
    async with app_harness.client() as (client, headers):
        admin = await client.get("/auth/me", headers=headers)
        # Seed a disposable target so we don't damage the corpus.
        probe = await _seed_disposable_delete_target(app_harness.root, item)
        await _set_file_status(probe.file_id, "ready")

        resp = await client.delete(f"/files/{probe.file_id}", headers=headers)
        assert resp.status_code == 202
        await _poll_file_deleted(client, headers, probe.file_id, timeout_s=30.0)

    async with session_scope() as db:
        rows = (
            await db.execute(
                sa_select(AuditLog).where(
                    AuditLog.action == "file.delete.complete",
                    AuditLog.target == probe.file_id,
                )
            )
        ).scalars().all()
    assert len(rows) == 1, "expected exactly one delete-complete audit row"
    counts = json.loads(rows[0].payload_json or "{}")
    # At minimum the bm25_rebuild count must be present (always written).
    assert "bm25_rebuild" in counts
    # And the per-store row counts that the probe row helper inserted.
    assert counts.get("dense_rows", 0) >= 1 or counts.get("graph_passage_rows", 0) >= 1


async def test_purge_uploads_does_not_cross_delete_prefix_sibling(
    app_harness: AppHarness,
) -> None:
    """Purging file_id ``abc_hash`` must NOT also delete the upload blob
    of an unrelated file whose name begins with ``abc_hash.`` — that's
    the cross-deletion hazard the exact-suffix fix closed."""
    from config.settings import upload_path, uploads_root
    from ingestion.index.maintenance import purge_file_artifacts

    uploads_root().mkdir(parents=True, exist_ok=True)

    target_id = "abc_hash"
    sibling_id = "abc_hash.v2_otherhash"

    target_path = upload_path(target_id, ".pdf")
    sibling_path = upload_path(sibling_id, ".pdf")
    target_path.write_bytes(b"target blob")
    sibling_path.write_bytes(b"sibling blob")

    try:
        # Purge target with EXACT suffix (the API delete path).
        purge_file_artifacts(target_id, upload_suffix=".pdf")
        assert not target_path.exists(), "target blob should be gone"
        assert sibling_path.exists(), (
            "sibling blob with shared prefix must NOT be deleted"
        )
    finally:
        sibling_path.unlink(missing_ok=True)
        target_path.unlink(missing_ok=True)

    # And the orphan-fallback path (no suffix) must also be safe: it uses
    # ``with_suffix("").name == file_id`` so the prefix-sibling shape
    # ``abc_hash.v2_otherhash.pdf`` (stem = "abc_hash.v2_otherhash")
    # never matches the target id "abc_hash".
    target_path.write_bytes(b"target blob v2")
    sibling_path.write_bytes(b"sibling blob v2")
    try:
        purge_file_artifacts(target_id)  # no suffix → orphan path
        assert not target_path.exists()
        assert sibling_path.exists()
    finally:
        sibling_path.unlink(missing_ok=True)
        target_path.unlink(missing_ok=True)


async def test_audit_endpoint_recovers_post_delete_history(
    app_harness: AppHarness,
) -> None:
    """``GET /files/{id}/jobs`` 404s after a successful delete because
    the IngestJob rows were cascaded away. The audit endpoint must
    surface the ``file.delete.complete`` row + payload so the frontend
    can still render delete history."""
    item = CORPUS[0]
    async with app_harness.client() as (client, headers):
        probe = await _seed_disposable_delete_target(app_harness.root, item)
        await _set_file_status(probe.file_id, "ready")

        resp = await client.delete(f"/files/{probe.file_id}", headers=headers)
        assert resp.status_code == 202
        await _poll_file_deleted(client, headers, probe.file_id, timeout_s=30.0)

        # Per-file jobs endpoint can no longer help.
        gone = await client.get(f"/files/{probe.file_id}/jobs", headers=headers)
        assert gone.status_code == 404

        # But the audit endpoint can recover the trail.
        audit_resp = await client.get(
            "/audit",
            headers=headers,
            params={"action": "file.delete.complete", "target": probe.file_id},
        )
        assert audit_resp.status_code == 200
        entries = audit_resp.json()
        assert len(entries) == 1, entries
        entry = entries[0]
        assert entry["action"] == "file.delete.complete"
        assert entry["target"] == probe.file_id
        counts = json.loads(entry["payload_json"] or "{}")
        assert "bm25_rebuild" in counts


async def test_audit_endpoint_requires_admin(
    app_harness: AppHarness,
) -> None:
    """Audit history is admin-scope; an analyst session must get 403."""
    async with app_harness.client() as (client, _admin_headers):
        # Seed inside the client context so lifespan has already run
        # init_db() — otherwise the users table doesn't exist yet.
        await _seed_user(username="auditor_alice", password="alice123", role="analyst")
        analyst_headers = await _login_headers(client, "auditor_alice", "alice123")
        forbidden = await client.get("/audit", headers=analyst_headers)
        assert forbidden.status_code == 403


async def test_orphan_upload_sweep_reaps_blob_with_no_db_row(
    app_harness: AppHarness,
) -> None:
    """Simulate a process crash inside the
    ``commit_upload_to_disk → create_pending_record`` window: a blob
    on disk with no matching ``files`` row. Lifespan's
    ``sweep_orphan_uploads()`` must remove it on next start."""
    from api.services.files import sweep_orphan_uploads
    from config.settings import upload_path, uploads_root

    # Enter the client context so lifespan has ensured init_db() ran;
    # without it, ``sweep_orphan_uploads`` raises "no such table: files".
    async with app_harness.client() as (_client, _headers):
        uploads_root().mkdir(parents=True, exist_ok=True)

        orphan_id = "e2e_orphan_blob_xyz"
        orphan_path = upload_path(orphan_id, ".pdf")
        orphan_path.write_bytes(b"orphan blob from a hypothetical crash")
        assert orphan_path.is_file()

        # ``.part`` sentinel: mkstemp leftover from an in-flight write
        # — sweep must skip it (separate concern from "no DB row").
        part_path = uploads_root() / f".{orphan_id}.tmpx.part"
        part_path.write_bytes(b"in-flight write fragment")

        try:
            counts = await sweep_orphan_uploads()
            assert counts["removed"] >= 1
            assert not orphan_path.exists(), "orphan blob must be unlinked"
            assert part_path.is_file(), ".part files must be preserved by orphan sweep"
            assert counts["skipped_part_files"] >= 1
        finally:
            part_path.unlink(missing_ok=True)
            orphan_path.unlink(missing_ok=True)
