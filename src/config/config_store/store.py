"""Effective-config holder, used by the web layer and experiment scripts.

A ``ConfigStore`` is a snapshot of every registered key. Reads
(:meth:`get`, :meth:`materialize_*`) are O(1) dict lookups against the
in-memory snapshot. Mutations (:meth:`patch`, :meth:`reset`,
:meth:`reload`) rebuild the snapshot from the ``app_config`` table.

Two backends, one runtime API:

* :meth:`from_app_db` — for the FastAPI lifespan: reads the table
  once, falls back to schema defaults for any missing key.
* :meth:`defaults_only` — for experiment scripts that don't have
  ``app.db`` and want the original baseline numbers.

Hot-reload contract: a ``PATCH /admin/config`` route should
``await store.patch(...)`` (which already calls ``reload`` after the
write). The next request enters via FastAPI's worker, sees the new
snapshot. In-flight requests keep the value they already read — they
hold local copies (e.g. ``RAGPipeline`` instantiated with a
materialized ``RAGConfig``), and the store never mutates objects it
already handed out.
"""
import json
import logging
from dataclasses import replace
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from config.rag import RAGConfig

from config.config_store.entry import ConfigEntry
from config.config_store.schema import CONFIG_ENTRIES, CONFIG_ENTRIES_BY_KEY


logger = logging.getLogger(__name__)


# Map a config key onto the field of ``RAGConfig`` it overrides. Keys
# without an entry here don't participate in :meth:`materialize_rag_config`.
_RAG_FIELD_BY_KEY: Dict[str, str] = {
    "rag.rrf_k": "rrf_k",
    "rag.rrf_top_m": "rrf_top_m",
    "rag.rerank_top_n": "rerank_top_n",
    "rag.answer_max_tokens": "answer_max_tokens",
}


# Map (agent kind, override field) → config key. Used by
# :meth:`materialize_agent_kwargs`. ``None`` means the kind doesn't
# expose that override.
_AGENT_KEY_TABLE: Dict[str, Dict[str, str]] = {
    "base": {
        "max_loops": "agent.base.max_loops",
        "max_token_budget": "agent.base.max_token_budget",
        "system_prompt": "prompt.base_agent",
    },
    "proof": {
        "max_loops": "agent.proof.max_loops",
        "max_token_budget": "agent.proof.max_token_budget",
        "system_prompt": "prompt.proof_agent",
    },
    "graph": {
        "max_loops": "agent.graph.max_loops",
        "max_token_budget": "agent.graph.max_token_budget",
        "system_prompt": "prompt.graph_agent",
    },
    "web": {
        "max_loops": "agent.web.max_loops",
        "max_token_budget": "agent.web.max_token_budget",
        "system_prompt": "prompt.web_agent",
    },
}


class ConfigStore:
    """Read/write facade around the ``app_config`` K-V table."""

    def __init__(
        self,
        *,
        values: Dict[str, Any],
        db_factory: Optional[Callable[[], AsyncSession]] = None,
    ) -> None:
        self._values: Dict[str, Any] = dict(values)
        # ``db_factory`` is an async context manager factory (typically
        # the :class:`sessionmaker` itself) used for ``reload`` /
        # ``patch`` / ``reset`` on the web side. ``defaults_only`` /
        # in-memory scripts pass ``None`` and stay read-only.
        self._db_factory = db_factory
        self._entries: Dict[str, ConfigEntry] = CONFIG_ENTRIES_BY_KEY

    # ----------------------------------------------------- constructors ----

    @classmethod
    def defaults_only(cls) -> "ConfigStore":
        """Pure-schema-defaults store; no DB dependency. For experiment scripts."""
        return cls(values={e.key: e.default for e in CONFIG_ENTRIES})

    @classmethod
    async def from_app_db(
        cls,
        db_factory: Callable[[], AsyncSession],
    ) -> "ConfigStore":
        """Read every override row from ``app_config``; layer on schema defaults.

        Missing keys fall back to the schema default — that's the same
        path :meth:`defaults_only` walks for every key, so a fresh
        ``app.db`` is observably identical to "no DB at all".
        """
        store = cls.defaults_only()
        store._db_factory = db_factory
        await store.reload_from_factory()
        return store

    @classmethod
    def from_env(cls, app_db_path: Optional[Any] = None) -> "ConfigStore":
        """Pick the backend by looking at the filesystem.

        ``app.db`` exists → caller almost certainly wants the web-layer
        overrides (e.g. ``rag_eval`` reproducing a tuned production
        run). Otherwise → defaults only.

        We accept an optional ``app_db_path`` kwarg purely so tests can
        point at a temp DB without monkeypatching :mod:`config.settings`.
        """
        from config.settings import app_db_path as default_app_db_path

        path = app_db_path if app_db_path is not None else default_app_db_path()
        if path is None or not path.is_file():
            return cls.defaults_only()
        # We can't run ``await from_app_db`` from a sync entry point,
        # so build the snapshot synchronously by reading the sqlite
        # file directly. Keeps experiment scripts free of asyncio.
        return cls._from_sqlite_path_sync(path)

    @classmethod
    def _from_sqlite_path_sync(cls, app_db_path: Any) -> "ConfigStore":
        """Sync sqlite read — used only by :meth:`from_env` for scripts."""
        import sqlite3

        store = cls.defaults_only()
        try:
            conn = sqlite3.connect(str(app_db_path))
            try:
                cur = conn.execute("SELECT key, value_json FROM app_config")
                for key, value_json in cur.fetchall():
                    entry = store._entries.get(key)
                    if entry is None:
                        # Stale row from a removed key — ignore rather
                        # than crash an experiment script.
                        continue
                    try:
                        value = json.loads(value_json)
                        entry.validate(value)
                    except (ValueError, json.JSONDecodeError) as exc:
                        logger.warning(
                            "config_store: ignoring invalid stored value for %s: %s",
                            key,
                            exc,
                        )
                        continue
                    store._values[key] = value
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            # Most common case: the table exists but rows are nil. Fall
            # through and return the defaults-only snapshot we built above.
            logger.debug("config_store.from_env: sqlite read failed (%s); using defaults", exc)
        return store

    # ----------------------------------------------------------- reads ----

    def get(self, key: str) -> Any:
        try:
            return self._values[key]
        except KeyError as exc:
            raise KeyError(f"config_store: unknown key {key!r}") from exc

    def snapshot(self) -> Dict[str, Any]:
        """Copy of the full effective dict; safe to mutate."""
        return dict(self._values)

    @property
    def schema(self) -> List[ConfigEntry]:
        """Registered entries in declaration order — handy for the admin UI."""
        return list(CONFIG_ENTRIES)

    # ------------------------------------------------- materializers ----

    def materialize_rag_config(self, base: Optional[RAGConfig] = None) -> RAGConfig:
        """``RAGConfig`` with the admin-managed knobs swapped in.

        We only expose four fields through the admin UI; the rest of
        the dataclass (per-channel topks, PPR coefficients,
        ``rerank_doc_max_chars``, …) stay code-managed. Defaulting
        ``base`` to a fresh ``RAGConfig()`` would silently flatten any
        non-admin tuning a caller had baked into their pipeline. The
        web runner threads ``pipeline.config`` here so the pipeline's
        constructor-time customizations survive the override.
        """
        starting = base if base is not None else RAGConfig()
        overrides: Dict[str, Any] = {}
        for key, field in _RAG_FIELD_BY_KEY.items():
            overrides[field] = self._values[key]
        return replace(starting, **overrides)

    def materialize_agent_kwargs(self, kind: str) -> Dict[str, Any]:
        """``{max_loops, max_token_budget, system_prompt}`` for ``BaseAgent.run``.

        Pass the dict straight as ``**kwargs`` to ``agent.run(...)`` —
        the ``run`` method's three new override kwargs each treat
        ``None`` as "use the constructor value", but here we always
        supply the effective config value so the admin overrides win.
        """
        try:
            mapping = _AGENT_KEY_TABLE[kind]
        except KeyError as exc:
            raise ValueError(f"materialize_agent_kwargs: unknown agent kind {kind!r}") from exc
        return {field: self._values[key] for field, key in mapping.items()}

    def citation_preview_chars(self) -> int:
        return int(self._values["citation.preview_chars"])

    def chat_history_turns(self) -> int:
        """Recent (user, assistant) pairs replayed into the next request.

        0 makes every request stateless and lets the chat route skip
        the trace lookup entirely.
        """
        return int(self._values["chat.history_turns"])

    def ingest_parallel_workers(self) -> int:
        """Cap on concurrent parse stages (faiss/graph index write stays serial).

        Powers the ``INGEST_PARSE_SEM`` semaphore in
        :mod:`api.services.files`. 1 = fully-serial parse.
        """
        return int(self._values["ingest.parallel_workers"])

    def materialize_linear_rag_config(
        self, base: Optional["LinearRAGConfig"] = None
    ) -> "LinearRAGConfig":
        """``LinearRAGConfig`` with admin-managed knobs swapped in.

        Same pattern as :meth:`materialize_rag_config` — preserve any
        constructor-time tuning of non-admin fields (embedding client,
        alias thresholds) and only overwrite the literal-backfill +
        GLiNER knobs that the admin UI exposes. The backfill defaults
        double as the query-time PPR gazetteer defaults inside
        :class:`GraphPPRChannel`, so flipping them here propagates to
        both ingest and retrieval.
        """
        from config.linear_rag import LinearRAGConfig
        starting = base if base is not None else LinearRAGConfig()
        return replace(
            starting,
            literal_backfill_enabled=bool(self._values["linear_rag.literal_backfill_enabled"]),
            literal_backfill_min_chars=int(self._values["linear_rag.literal_backfill_min_chars"]),
            literal_backfill_multi_word_only=bool(
                self._values["linear_rag.literal_backfill_multi_word_only"]
            ),
            gliner_model_id=str(self._values["linear_rag.gliner_model_id"]),
            gliner_labels=list(self._values["linear_rag.gliner_labels"]),
            gliner_noise_labels=list(self._values["linear_rag.gliner_noise_labels"]),
            gliner_threshold=float(self._values["linear_rag.gliner_threshold"]),
            junk_max_han_chars=int(self._values["linear_rag.junk_max_han_chars"]),
            ner_max_span_chars=int(self._values["linear_rag.ner_max_span_chars"]),
            acceptance_handler=str(self._values["linear_rag.acceptance_handler"]),
            graphml_flush_every=int(self._values["linear_rag.graphml_flush_every"]),
            cluster_shape_every=int(self._values["linear_rag.cluster_shape_every"]),
            cluster_algorithm=str(self._values["linear_rag.cluster_algorithm"]),
            cluster_leiden_resolution=float(
                self._values["linear_rag.cluster_leiden_resolution"]
            ),
            cluster_leiden_weighted=bool(
                self._values["linear_rag.cluster_leiden_weighted"]
            ),
            alias_propagation_policy=str(self._values["linear_rag.alias_propagation_policy"]),
            alias_prop_const=float(self._values["linear_rag.alias_prop_const"]),
            alias_prop_lo=float(self._values["linear_rag.alias_prop_lo"]),
            alias_prop_hi=float(self._values["linear_rag.alias_prop_hi"]),
            alias_prop_tau_cos=float(self._values["linear_rag.alias_prop_tau_cos"]),
            alias_prop_tau_rerank=float(self._values["linear_rag.alias_prop_tau_rerank"]),
        )

    def graph_explore_kwargs(self) -> Dict[str, Any]:
        """``{entity_lookup_min_sim, entity_lookup_gradient}`` for GraphExploreTool.

        Pass straight as ``**kwargs`` to ``GraphExploreTool(...)`` from
        the agent factory; the tool's constructor accepts these as
        kw-only and falls back to its built-in defaults so direct
        instantiations (experiment scripts) keep working unchanged.
        """
        return {
            "entity_lookup_min_sim": float(self._values["graph_explore.entity_lookup_min_sim"]),
            "entity_lookup_gradient": float(self._values["graph_explore.entity_lookup_gradient"]),
        }

    # ---------------------------------------------------------- mutations ----

    async def patch(
        self,
        db: AsyncSession,
        *,
        updates: Dict[str, Any],
        user_id: Optional[int],
    ) -> Dict[str, Dict[str, Any]]:
        """Validate + UPSERT every entry in ``updates`` atomically; return diffs.

        Either every update lands or none does — schema validation runs
        first against every key, then we UPSERT the ``app_config`` rows
        AND the matching ``audit_log`` entries inside one transaction
        and commit. Only after ``commit()`` succeeds do we mutate the
        in-memory snapshot. This guarantees the contract: a reader
        (web runner or another request) never observes an in-memory
        value that didn't durably land in the DB. A failed commit
        re-raises and leaves both the snapshot and DB at their
        pre-call state.

        Caller contract: ``db`` MUST NOT carry unrelated pending writes
        — this method commits the whole session. The admin route opens
        a fresh ``get_session`` per request, so this holds in practice;
        new call sites should respect it.

        Returns ``{key: {"old": ..., "new": ...}}`` for the route
        response.
        """
        from api.models import AppConfig, AuditLog  # local: algorithm pkg → web orm

        # Validate everything upfront. ``ValueError`` propagates with the
        # offending key in the message — admin route translates to 422.
        validated: Dict[str, Any] = {}
        for key, raw_value in updates.items():
            entry = self._entries.get(key)
            if entry is None:
                raise ValueError(f"unknown config key {key!r}")
            validated[key] = entry.validate(raw_value)

        # Capture pre-image for the audit trail before we touch the DB.
        diffs: Dict[str, Dict[str, Any]] = {
            key: {"old": self._values[key], "new": new}
            for key, new in validated.items()
        }

        for key, value in validated.items():
            value_json = json.dumps(value, ensure_ascii=False)
            existing = await db.get(AppConfig, key)
            if existing is None:
                db.add(
                    AppConfig(
                        key=key,
                        value_json=value_json,
                        updated_by=user_id,
                    )
                )
            else:
                existing.value_json = value_json
                existing.updated_by = user_id
            db.add(
                AuditLog(
                    user_id=user_id,
                    action="config.update",
                    target=key,
                    payload_json=json.dumps(diffs[key], ensure_ascii=False),
                )
            )

        # Commit is the durability barrier. Memory mutation happens
        # AFTER the commit lands, so a SQL failure (e.g. disk full)
        # never leaves the snapshot ahead of the persisted state.
        await db.commit()
        self._values.update(validated)
        return diffs

    async def reset(
        self,
        db: AsyncSession,
        *,
        key: str,
        user_id: Optional[int],
    ) -> Dict[str, Any]:
        """Drop the override row for ``key``; revert to schema default.

        Same commit-before-mutate ordering as :meth:`patch` so the
        in-memory snapshot can never lead the DB.
        """
        from api.models import AppConfig, AuditLog

        entry = self._entries.get(key)
        if entry is None:
            raise ValueError(f"unknown config key {key!r}")

        old = self._values[key]
        change = {"old": old, "new": entry.default}
        await db.execute(delete(AppConfig).where(AppConfig.key == key))
        db.add(
            AuditLog(
                user_id=user_id,
                action="config.reset",
                target=key,
                payload_json=json.dumps(change, ensure_ascii=False),
            )
        )
        await db.commit()
        self._values[key] = entry.default
        return change

    async def reload_from_factory(self) -> None:
        """Open a fresh session via ``self._db_factory`` and reload the snapshot."""
        if self._db_factory is None:
            raise RuntimeError("ConfigStore has no db_factory; reload requires from_app_db()")
        async with self._db_factory() as db:
            await self.reload(db)

    async def reload(self, db: AsyncSession) -> None:
        """Re-read every row from ``app_config``; reset to defaults first."""
        from api.models import AppConfig

        # Reset snapshot to defaults so a deleted row reverts cleanly.
        new_values = {e.key: e.default for e in CONFIG_ENTRIES}

        rows = (await db.execute(select(AppConfig))).scalars().all()
        for row in rows:
            entry = self._entries.get(row.key)
            if entry is None:
                logger.warning(
                    "config_store.reload: ignoring stale row for unknown key %s", row.key
                )
                continue
            try:
                value = json.loads(row.value_json)
                entry.validate(value)
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning(
                    "config_store.reload: ignoring invalid value for %s: %s",
                    row.key,
                    exc,
                )
                continue
            new_values[row.key] = value
        self._values = new_values
