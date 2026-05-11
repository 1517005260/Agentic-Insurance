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
    """Bring the DB schema up to head. Idempotent.

    Three cases:

    1. **Fresh DB** (no app tables) → ``alembic upgrade head`` builds
       everything from migrations.
    2. **Unmanaged DB** (app tables exist but no alembic version row)
       → stamp ``head`` so Alembic treats the schema as current and
       skips CREATE TABLE for rows that already exist.
    3. **Already-managed DB** (alembic_version row at head) →
       ``upgrade head`` is a no-op; no work, no errors.

    Case 2 is detected by "users exists AND alembic_version is empty",
    which also covers the case where a prior failed upgrade left an
    empty ``alembic_version`` table behind.

    Why we DON'T share the lifespan's AsyncConnection with Alembic:
    SQLite migrations run with ``transactional_ddl=False`` (Alembic's
    default for SQLite); wrapping CREATE TABLE inside an outer
    ``engine.begin()`` BEGIN block deadlocks the env.py txn release
    on the very first migration. Instead we read the DB state with
    a non-transactional ``engine.connect()``, then close that handle
    and let Alembic open its own sync engine via the URL override in
    ``alembic/env.py`` (which resolves to the same ``app_db_path()``).
    """
    import asyncio

    from sqlalchemy import inspect as sa_inspect, text

    # Probe schema state without holding a transaction.
    async with engine.connect() as conn:
        existing_tables = await conn.run_sync(
            lambda sync_conn: sa_inspect(sync_conn).get_table_names()
        )
        users_present = "users" in existing_tables
        has_version_row = False
        if "alembic_version" in existing_tables:
            row = (
                await conn.execute(text("SELECT version_num FROM alembic_version"))
            ).first()
            has_version_row = row is not None

    # Hand off to Alembic on its own thread with its own sync engine —
    # never share the async lifespan engine's connections.
    loop = asyncio.get_running_loop()
    if users_present and not has_version_row:
        await loop.run_in_executor(None, _alembic_stamp_head)
    else:
        await loop.run_in_executor(None, _alembic_upgrade_head)


def _alembic_upgrade_head() -> None:
    """Run ``alembic upgrade head``. Sync — call via run_in_executor.

    Alembic opens its own sync engine via ``sqlalchemy.url`` in
    ``alembic.ini`` (overridden in ``alembic/env.py`` to
    ``sqlite:///{app_db_path()}``).
    """
    from alembic import command
    from alembic.config import Config
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    cfg = Config(str(project_root / "alembic.ini"))
    command.upgrade(cfg, "head")


def _alembic_stamp_head() -> None:
    """Mark the DB as already at head without running migrations."""
    from alembic import command
    from alembic.config import Config
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    cfg = Config(str(project_root / "alembic.ini"))
    command.stamp(cfg, "head")


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
