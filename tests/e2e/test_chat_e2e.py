"""E2E for chat sessions + (mocked) streaming.

We split the surface into two layers:

* **Session CRUD** is a real round-trip through FastAPI + SQLite, using
  the same ``ASGITransport`` lifespan pattern as ``test_files_e2e``.
  These cover the happy path, RBAC, mode/agent_kind validation, and
  cascade delete.

* **Streaming + persistence** is exercised against a stub runner so
  the test doesn't pay 2-5 min for a real RAG/agent run on every CI
  invocation. The stub mimics the runner's contract: yields a few
  SSE-encoded frames and resolves the result-future with a payload
  the route persists into ``chat_messages``. The genuine pipeline /
  agent runs are covered by the live smoke scripts in ``scripts/``.
"""
import asyncio
import importlib
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient


pytestmark = pytest.mark.anyio


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


async def _heartbeat() -> None:
    while True:
        await asyncio.sleep(0.01)


@asynccontextmanager
async def _heartbeat_ctx() -> AsyncIterator[None]:
    task = asyncio.create_task(_heartbeat())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _dispose_engine_if_loaded() -> None:
    db_mod = sys.modules.get("api.db")
    if db_mod is not None:
        async with _heartbeat_ctx():
            await db_mod.engine.dispose()


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
        settings_mod.STORAGE_PATH = Path("./local_storage")
        settings_mod.ALLOW_INSECURE_JWT = True


def _preload_venv_sqlalchemy(venv_site: Path) -> None:
    """Mirror the trick from test_files_e2e — keep SQLAlchemy + aiosqlite paired.

    The bare ``pytest`` on this dev box runs from Anaconda; the app
    deps live in ``.venv``. Anaconda SQLAlchemy + venv ``aiosqlite``
    can produce ``ArgumentError`` when ``select(User)`` resolves to a
    Table object — the two installs disagree about which decorator
    decorates which class. Preload from the project venv when test
    files share a session.
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


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def app_harness(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[dict]:
    """Boot the app in-process; clean DB before and after."""
    root = _repo_root()
    monkeypatch.chdir(root)
    monkeypatch.setenv("ALLOW_INSECURE_JWT", "1")
    monkeypatch.setenv("STORAGE_PATH", "./local_storage")
    monkeypatch.setenv("DEFAULT_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("DEFAULT_ADMIN_PASSWORD", "admin123")
    _force_runtime_settings()
    await _dispose_engine_if_loaded()
    _clean_db_files(root)
    try:
        yield {"root": root}
    finally:
        await _dispose_engine_if_loaded()
        _clean_db_files(root)


@asynccontextmanager
async def _client_for(app_harness):
    """Boot lifespan + return a logged-in admin client."""
    app_mod = importlib.import_module("api.main")
    app = app_mod.app
    async with _heartbeat_ctx():
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.post(
                    "/auth/login",
                    data={"username": "admin", "password": "admin123"},
                )
                assert resp.status_code == 200, resp.text
                token = resp.json()["access_token"]
                yield client, app, {"Authorization": f"Bearer {token}"}


# =====================================================================
# Session CRUD
# =====================================================================


async def test_session_lifecycle_crud(app_harness):
    """Create → list → get → patch → delete; verify cascade on messages."""
    async with _client_for(app_harness) as (client, app, headers):
        # Create rag session
        r = await client.post(
            "/chat/sessions",
            headers=headers,
            json={"mode": "rag", "title": "exam-axa"},
        )
        assert r.status_code == 201, r.text
        rag_session = r.json()
        assert rag_session["mode"] == "rag"
        assert rag_session["agent_kind"] is None

        # Create agent session
        r = await client.post(
            "/chat/sessions",
            headers=headers,
            json={"mode": "agent", "agent_kind": "proof", "title": "proof-run"},
        )
        assert r.status_code == 201, r.text
        agent_session = r.json()
        assert agent_session["agent_kind"] == "proof"

        # List — both visible, newest first
        r = await client.get("/chat/sessions", headers=headers)
        assert r.status_code == 200
        listed = r.json()
        assert len(listed) >= 2
        # Most recent is the agent one (created second)
        assert listed[0]["id"] == agent_session["id"]

        # Get detail
        r = await client.get(f"/chat/sessions/{rag_session['id']}", headers=headers)
        assert r.status_code == 200
        detail = r.json()
        assert detail["session"]["id"] == rag_session["id"]
        assert detail["messages"] == []

        # Patch title
        r = await client.patch(
            f"/chat/sessions/{rag_session['id']}",
            headers=headers,
            json={"title": "renamed"},
        )
        assert r.status_code == 200
        assert r.json()["title"] == "renamed"

        # Delete
        r = await client.delete(f"/chat/sessions/{rag_session['id']}", headers=headers)
        assert r.status_code == 204

        # Gone
        r = await client.get(f"/chat/sessions/{rag_session['id']}", headers=headers)
        assert r.status_code == 404


async def test_session_validation_mode_kind(app_harness):
    """SessionCreate enforces mode/agent_kind compatibility before hitting DB."""
    async with _client_for(app_harness) as (client, _app, headers):
        # mode=agent requires agent_kind
        r = await client.post(
            "/chat/sessions",
            headers=headers,
            json={"mode": "agent"},
        )
        assert r.status_code == 422

        # mode=rag forbids agent_kind
        r = await client.post(
            "/chat/sessions",
            headers=headers,
            json={"mode": "rag", "agent_kind": "base"},
        )
        assert r.status_code == 422


async def test_session_rbac_isolation(app_harness):
    """Sessions are scoped to their owning user; cross-user access returns 404."""
    from api.auth import hash_password
    from api.db import session_scope
    from api.models import User

    async with _client_for(app_harness) as (client, _app, admin_headers):
        # Create a second user (analyst) directly in DB.
        async with session_scope() as db:
            db.add(
                User(
                    username="analyst1",
                    password_hash=hash_password("analyst-pwd-123"),
                    role="analyst",
                    is_active=1,
                )
            )

        # Admin creates a session
        r = await client.post(
            "/chat/sessions",
            headers=admin_headers,
            json={"mode": "rag", "title": "admin-only"},
        )
        assert r.status_code == 201, r.text
        admin_session_id = r.json()["id"]

        # Analyst logs in
        r = await client.post(
            "/auth/login",
            data={"username": "analyst1", "password": "analyst-pwd-123"},
        )
        assert r.status_code == 200, r.text
        analyst_headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

        # Analyst can't see admin's session — 404, NOT 403 (no leak).
        r = await client.get(
            f"/chat/sessions/{admin_session_id}", headers=analyst_headers
        )
        assert r.status_code == 404

        # Analyst can't patch / delete it either.
        r = await client.patch(
            f"/chat/sessions/{admin_session_id}",
            headers=analyst_headers,
            json={"title": "hijack"},
        )
        assert r.status_code == 404
        r = await client.delete(
            f"/chat/sessions/{admin_session_id}", headers=analyst_headers
        )
        assert r.status_code == 404

        # Analyst's own list is empty.
        r = await client.get("/chat/sessions", headers=analyst_headers)
        assert r.status_code == 200
        assert r.json() == []


# =====================================================================
# Streaming + persistence (with stub runner)
# =====================================================================


_STUB_FRAMES = [
    b"event: status\ndata: {\"phase\":\"preprocess\"}\n\n",
    b"event: token\ndata: {\"delta\":\"Hello \"}\n\n",
    b"event: token\ndata: {\"delta\":\"world.\"}\n\n",
    b"event: citations\ndata: {\"items\":[]}\n\n",
    b"event: final\ndata: {\"answer_chars\":12}\n\n",
    b"event: done\ndata: {}\n\n",
]


async def _stub_stream(
    *, query, file_ids, pipeline, config=None, tracer=None, result_future=None,
    history=None,
):
    """Mimic stream_rag's contract: yield frames + set result_future."""
    for frame in _STUB_FRAMES:
        yield frame
    if result_future is not None and not result_future.done():
        result_future.set_result(
            {
                "answer": "Hello world.",
                "exit_reason": "ok",
                "citations": [],
                "channels_hit_counts": {"semantic": 0, "bm25": 0, "graph_ppr": 0, "regex": 0},
                "timings_ms": {"preprocess": 1, "retrieve": 2, "rerank": 1, "answer": 1},
                "reranked_count": 0,
                "trace_path": "rag/2026-05-06/000000_stubrun",
            }
        )


async def test_session_message_round_trip_persists_assistant(monkeypatch, app_harness):
    """POST a message → SSE stream → assistant row persisted with metadata."""
    from api.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "stream_rag", _stub_stream)

    async with _client_for(app_harness) as (client, _app, headers):
        # Create session
        r = await client.post(
            "/chat/sessions",
            headers=headers,
            json={"mode": "rag", "title": "stub-rag"},
        )
        sid = r.json()["id"]

        # POST message — must consume the full body so the after-stream
        # persistence hook fires.
        async with client.stream(
            "POST",
            f"/chat/sessions/{sid}/messages",
            headers=headers,
            json={"content": "ping"},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            chunks: list[bytes] = []
            async for chunk in resp.aiter_raw():
                chunks.append(chunk)
            body = b"".join(chunks).decode()
            assert "event: status" in body
            assert "Hello " in body
            assert "event: done" in body

        # Persistence happens in the route's finally block; the route
        # has returned by now. Brief poll for the assistant message to
        # land — same kind of post-stream timing the smoke shell waits
        # for. ``persist_assistant_message`` opens its own session; the
        # GET endpoint reads via the request-scoped session, so we may
        # need a moment for the writer's commit to be visible.
        for _ in range(50):
            r = await client.get(f"/chat/sessions/{sid}", headers=headers)
            messages = r.json()["messages"]
            if len(messages) >= 2:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("assistant message never landed in DB")

        # First is user, second is assistant
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "ping"

        assistant = messages[1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == "Hello world."
        meta = assistant["metadata"]
        assert isinstance(meta, dict)
        assert meta["exit_reason"] == "ok"
        assert meta["trace_path"] == "rag/2026-05-06/000000_stubrun"
        assert "timings_ms" in meta
        assert "channels_hit_counts" in meta


async def test_session_message_session_not_found(app_harness):
    """POSTing to an unknown / not-owned session 404s before any stream open."""
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.post(
            "/chat/sessions/99999/messages",
            headers=headers,
            json={"content": "x"},
        )
        assert r.status_code == 404
