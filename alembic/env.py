"""Alembic env — anchors against ``api.models.Base`` + project-local SQLite.

Tweaks vs. the stock template:

* ``sqlalchemy.url`` is overridden at runtime from
  :func:`config.settings.app_db_path` so the migration tooling lands
  on the same DB the app uses (no second SQLite file under cwd).
* ``target_metadata`` points at :data:`api.models.Base.metadata` for
  autogenerate.
* ``render_as_batch=True`` so SQLite-specific column / constraint
  rewrites use the table-rebuild dance Alembic implements; without it
  any future column drop / type change would 404 on SQLite.
* The PYTHONPATH addition at the top lets ``alembic`` run from any
  cwd without first ``cd``'ing into ``src/``.
"""
from logging.config import fileConfig
from pathlib import Path
import sys

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Make ``src/`` importable regardless of how alembic is invoked
# (``alembic upgrade head`` from project root, in-process from
# api.main lifespan, etc.). Mirrors the project's PYTHONPATH=src
# convention.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.models import Base  # noqa: E402  (after sys.path tweak)
from config.settings import app_db_path  # noqa: E402


config = context.config

# ``disable_existing_loggers=False`` keeps uvicorn / our app loggers
# alive after this env.py imports. Default True silently kills them,
# so the only log output past "Running upgrade ..." is alembic's own —
# the rest of the lifespan (admin seed / config / spaCy preheat /
# "Application startup complete.") looks like a hang from the user's
# terminal even though the server is fully up.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Override sqlalchemy.url so alembic always lands on the canonical DB
# regardless of the placeholder in alembic.ini. Single-source-of-truth
# rule (project memory: STORAGE_PATH anchored to repo root). The
# ALEMBIC_URL_OVERRIDE env escape lets ``alembic revision --autogenerate``
# diff against an empty / scratch DB without touching the live one
# (otherwise autogenerate produces an empty migration because the
# live DB already has every table).
import os as _os
config.set_main_option(
    "sqlalchemy.url",
    _os.environ.get("ALEMBIC_URL_OVERRIDE")
    or f"sqlite:///{app_db_path()}",
)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    When invoked from ``api.db.init_db`` we already hold a live
    connection (the lifespan's async-engine has set the SQLite
    PRAGMAs); reusing it via ``cfg.attributes['connection']``
    avoids opening a second handle that would skip those pragmas.
    Falls back to creating an engine from the URL when invoked
    standalone (CLI ``alembic upgrade head``).
    """
    existing_conn = config.attributes.get("connection")
    if existing_conn is not None:
        context.configure(
            connection=existing_conn,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()
        return

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
