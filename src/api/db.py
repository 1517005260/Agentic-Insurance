"""Async SQLite engine + session lifecycle.

One engine for the whole process; sessions are short-lived and produced
by ``get_session`` (a FastAPI dependency).

SQLite tuning notes:

* ``journal_mode=WAL`` lets readers and a single writer coexist without
  blocking each other — the API serves chat / file lists from many
  connections while one background ingest writes ``files`` / ``ingest_jobs``.
* ``foreign_keys=ON`` is needed per-connection (sqlite default is OFF).
* ``synchronous=NORMAL`` is safe under WAL and ~10× faster than ``FULL``.
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config.settings import app_db_dir, app_db_path


def _build_engine() -> AsyncEngine:
    app_db_dir().mkdir(parents=True, exist_ok=True)
    url = f"sqlite+aiosqlite:///{app_db_path()}"
    engine = create_async_engine(
        url,
        echo=False,
        future=True,
        # aiosqlite uses one connection per AsyncSession; pool_size is
        # irrelevant. We do want pre-ping so a closed sqlite handle (e.g.
        # after laptop sleep) is replaced rather than raising.
        pool_pre_ping=True,
    )

    # Per-connection pragmas. SQLAlchemy fires ``connect`` once per new
    # underlying DBAPI connection — covers both the sessionmaker and the
    # one-off connection used by ``init_db()``.
    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_conn, _record) -> None:  # noqa: ANN001
        cur = dbapi_conn.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=5000")
        finally:
            cur.close()

    return engine


engine: AsyncEngine = _build_engine()
SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


async def init_db() -> None:
    """Create all tables on first startup. Idempotent.

    Bootstrap path only — uses ``Base.metadata.create_all`` which adds
    missing tables but cannot evolve column / constraint changes. Once
    the schema starts changing in production, switch to alembic
    (``alembic init`` + autogenerate); the dependency is already in
    ``pyproject.toml``.

    A tiny ad-hoc dev-mode top-up runs after ``create_all`` to add the
    ``chat_sessions.web`` column to dev DBs that don't have it yet, so
    the operator doesn't have to wipe ``app.db`` by hand. The column
    is ``NOT NULL DEFAULT 0`` so existing rows get a safe value;
    SQLite ignores model-level CHECK constraints on a late-added
    column, so the Pydantic layer rejects forbidden combos at request
    time instead.
    """
    from api.models import Base  # local import to break cyclic init

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_chat_sessions_web_column(conn)


async def _ensure_chat_sessions_web_column(conn) -> None:
    """Add ``chat_sessions.web`` (default 0) if the table lacks it.

    SQLite's ``ALTER TABLE ADD COLUMN`` is the cheap path; we don't
    rebuild the table because re-creating CHECK constraints requires
    a 12-step dance that's overkill for a single dev-machine concern.
    Production migrations should use alembic.
    """
    from sqlalchemy import text

    cols = (
        await conn.execute(text("PRAGMA table_info(chat_sessions)"))
    ).fetchall()
    have_web = any(row[1] == "web" for row in cols)
    if have_web:
        return
    await conn.execute(
        text(
            "ALTER TABLE chat_sessions "
            "ADD COLUMN web INTEGER NOT NULL DEFAULT 0"
        )
    )


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Background-task helper: ``async with session_scope() as db: ...``.

    Routes use the ``get_session`` dependency in ``deps.py`` instead;
    this is for code that runs outside a request (background ingest).
    """
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
