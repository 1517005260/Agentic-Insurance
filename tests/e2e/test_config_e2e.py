"""E2E for the admin config-center routes.

Covers the four endpoints (`GET`, `GET /schema`, `PATCH`, `DELETE`),
RBAC, validation failures, audit-log sidecar rows, and that a PATCH
changes what the next request sees through ``app.state.config``.

Reuses the lifespan harness from ``test_chat_e2e``: shared helpers
mean we only define what's specific to the config surface here.
"""
import importlib
import json
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select


# Reuse the harness wiring from test_chat_e2e — same lifespan boot,
# same DB-cleanup contract. Importing the helpers keeps this file
# focused on the config surface. ``tests/e2e/`` has no ``__init__.py``
# so pytest puts the directory on sys.path; bare-name import works.
from test_chat_e2e import (  # noqa: E402
    _client_for,
    _force_runtime_settings,
    app_harness,  # pytest fixture re-used as-is
    anyio_backend,
)


pytestmark = pytest.mark.anyio


@asynccontextmanager
async def _admin_and_analyst(app_harness) -> AsyncIterator[tuple]:
    """Boot the app, yield (client, app, admin_headers, analyst_headers)."""
    from api.auth import hash_password
    from api.db import session_scope
    from api.models import User

    async with _client_for(app_harness) as (client, app, admin_headers):
        async with session_scope() as db:
            db.add(
                User(
                    username="analyst1",
                    password_hash=hash_password("analyst-pwd-123"),
                    role="analyst",
                    is_active=1,
                )
            )

        r = await client.post(
            "/auth/login",
            data={"username": "analyst1", "password": "analyst-pwd-123"},
        )
        assert r.status_code == 200, r.text
        analyst_headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        yield client, app, admin_headers, analyst_headers


# =====================================================================
# Read endpoints + RBAC
# =====================================================================


async def test_get_config_returns_54_keys_and_schema(app_harness):
    """GET /admin/config returns the full snapshot + schema.

    The 54 keys break down by group as: linear_rag (22) + prompt (13) +
    agent (8) + rag (4) + graph_explore (2) + tavily (2) + chat (1) +
    citation (1) + ingest (1). This test pins the total so adding or
    removing a key without updating ``config_store/schema.py`` fails loudly.
    """
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.get("/admin/config", headers=headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body.keys()) == {"snapshot", "schema"}
        assert len(body["snapshot"]) == 54
        assert len(body["schema"]) == 54

        keys_in_schema = {entry["key"] for entry in body["schema"]}
        assert keys_in_schema == set(body["snapshot"].keys())

        # Defaults match the live algorithm sources at boot time.
        from config.rag import RAGConfig

        assert body["snapshot"]["rag.rrf_k"] == RAGConfig().rrf_k


async def test_get_schema_only(app_harness):
    """The schema-only endpoint mirrors the schema half of the full GET."""
    async with _client_for(app_harness) as (client, _app, headers):
        full = (await client.get("/admin/config", headers=headers)).json()
        only = (await client.get("/admin/config/schema", headers=headers)).json()
        assert full["schema"] == only


async def test_admin_only_rbac(app_harness):
    """Analysts get 403 across the board; the snapshot must not leak."""
    async with _admin_and_analyst(app_harness) as (
        client,
        _app,
        _admin_headers,
        analyst_headers,
    ):
        for method, path, payload in [
            ("get", "/admin/config", None),
            ("get", "/admin/config/schema", None),
            ("patch", "/admin/config", {"updates": {"rag.rrf_k": 50}}),
            ("delete", "/admin/config/rag.rrf_k", None),
        ]:
            req = getattr(client, method)
            kwargs = {"headers": analyst_headers}
            if payload is not None:
                kwargs["json"] = payload
            r = await req(path, **kwargs)
            assert r.status_code == 403, (method, path, r.text)


# =====================================================================
# PATCH — happy path, validation, audit, hot reload
# =====================================================================


async def test_patch_single_key_persists_and_audits(app_harness):
    """PATCH writes the row, mutates app.state.config, and audits the diff."""
    from api.db import session_scope
    from api.models import AppConfig, AuditLog

    async with _client_for(app_harness) as (client, app, headers):
        before = (await client.get("/admin/config", headers=headers)).json()
        old = before["snapshot"]["rag.rrf_k"]
        new = old + 5

        r = await client.patch(
            "/admin/config",
            headers=headers,
            json={"updates": {"rag.rrf_k": new}},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["diffs"] == {"rag.rrf_k": {"old": old, "new": new}}
        assert body["snapshot"]["rag.rrf_k"] == new

        # Subsequent GET sees the new value (hot reload via in-place mutation).
        again = (await client.get("/admin/config", headers=headers)).json()
        assert again["snapshot"]["rag.rrf_k"] == new

        # And the runner store on app.state is the same instance.
        assert app.state.config.get("rag.rrf_k") == new

        # DB row + audit_log entry both landed.
        async with session_scope() as db:
            row = await db.get(AppConfig, "rag.rrf_k")
            assert row is not None
            assert json.loads(row.value_json) == new

            audit_rows = (
                await db.execute(
                    select(AuditLog).where(AuditLog.action == "config.update")
                )
            ).scalars().all()
            assert len(audit_rows) == 1
            assert audit_rows[0].target == "rag.rrf_k"
            assert json.loads(audit_rows[0].payload_json) == {
                "old": old,
                "new": new,
            }


async def test_patch_batch_all_or_nothing(app_harness):
    """A bad value in a batch must abort the whole patch — no partial write."""
    from api.db import session_scope
    from api.models import AppConfig

    async with _client_for(app_harness) as (client, _app, headers):
        before = (await client.get("/admin/config", headers=headers)).json()
        good_old = before["snapshot"]["rag.rrf_top_m"]

        r = await client.patch(
            "/admin/config",
            headers=headers,
            json={
                "updates": {
                    "rag.rrf_top_m": 50,            # valid
                    "rag.rerank_top_n": 9999,      # > max=30
                }
            },
        )
        assert r.status_code == 422, r.text
        assert "rag.rerank_top_n" in r.text

        # Snapshot unchanged.
        after = (await client.get("/admin/config", headers=headers)).json()
        assert after["snapshot"]["rag.rrf_top_m"] == good_old

        # No app_config row was inserted for the valid half either.
        async with session_scope() as db:
            assert (await db.get(AppConfig, "rag.rrf_top_m")) is None


async def test_patch_unknown_key_422(app_harness):
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.patch(
            "/admin/config",
            headers=headers,
            json={"updates": {"made.up.key": 1}},
        )
        assert r.status_code == 422
        assert "made.up.key" in r.text


async def test_patch_empty_body_400(app_harness):
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.patch(
            "/admin/config",
            headers=headers,
            json={"updates": {}},
        )
        assert r.status_code == 400


# =====================================================================
# DELETE — reset to default
# =====================================================================


async def test_delete_resets_to_schema_default(app_harness):
    """DELETE drops the override row and the snapshot reverts."""
    from api.db import session_scope
    from api.models import AppConfig, AuditLog

    async with _client_for(app_harness) as (client, app, headers):
        before = (await client.get("/admin/config", headers=headers)).json()
        default_value = before["snapshot"]["rag.rrf_k"]

        # PATCH then DELETE.
        await client.patch(
            "/admin/config",
            headers=headers,
            json={"updates": {"rag.rrf_k": default_value + 7}},
        )
        r = await client.delete("/admin/config/rag.rrf_k", headers=headers)
        assert r.status_code == 204, r.text

        snap = (await client.get("/admin/config", headers=headers)).json()
        assert snap["snapshot"]["rag.rrf_k"] == default_value
        assert app.state.config.get("rag.rrf_k") == default_value

        async with session_scope() as db:
            assert (await db.get(AppConfig, "rag.rrf_k")) is None
            audit_rows = (
                await db.execute(
                    select(AuditLog).where(AuditLog.action == "config.reset")
                )
            ).scalars().all()
            assert len(audit_rows) == 1
            assert audit_rows[0].target == "rag.rrf_k"


async def test_delete_unknown_key_404(app_harness):
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.delete("/admin/config/made.up.key", headers=headers)
        assert r.status_code == 404


# =====================================================================
# materialize_* sees the live overrides
# =====================================================================


async def test_materialize_reflects_patch(app_harness):
    """After PATCH, materialize_rag_config + materialize_agent_kwargs see new values."""
    async with _client_for(app_harness) as (client, app, headers):
        await client.patch(
            "/admin/config",
            headers=headers,
            json={
                "updates": {
                    "rag.rerank_top_n": 5,
                    "agent.proof.max_loops": 7,
                    "prompt.proof_agent": "test-prompt-override",
                }
            },
        )
        rag_cfg = app.state.config.materialize_rag_config()
        assert rag_cfg.rerank_top_n == 5

        proof_kw = app.state.config.materialize_agent_kwargs("proof")
        assert proof_kw["max_loops"] == 7
        assert proof_kw["system_prompt"] == "test-prompt-override"

        # base / graph stay at their defaults (we only patched proof).
        base_kw = app.state.config.materialize_agent_kwargs("base")
        assert base_kw["max_loops"] == 24


async def test_materialize_rag_config_preserves_non_admin_fields(app_harness):
    """A custom pipeline.config base must survive the materialize() override.

    The runner threads ``pipeline.config`` as the base so non-admin
    fields (e.g. per-channel topks, rerank_doc_max_chars) keep their
    constructor-time values when the config store overlays the four
    admin knobs. Without this, any tuning baked into ``RAGPipeline()``
    silently reverts on every web request.
    """
    from config.rag import RAGConfig

    async with _client_for(app_harness) as (client, app, headers):
        await client.patch(
            "/admin/config",
            headers=headers,
            json={"updates": {"rag.rerank_top_n": 6}},
        )
        custom_base = RAGConfig(
            rerank_doc_max_chars=999,
            semantic_topk_per_subpath=77,
        )
        materialized = app.state.config.materialize_rag_config(base=custom_base)
        # Admin knob applied …
        assert materialized.rerank_top_n == 6
        # … but the custom base's non-admin fields survived.
        assert materialized.rerank_doc_max_chars == 999
        assert materialized.semantic_topk_per_subpath == 77


async def test_runner_threads_config_into_agent_run(monkeypatch, app_harness):
    """stream_agent must pass max_loops/max_token_budget/system_prompt to agent.run.

    Stub the agent singleton with a recorder; PATCH a tunable; fire
    the smoke endpoint; assert the recorder saw the patched value.
    """
    import asyncio

    from agentic.agent.base import BaseAgent
    from api.runners.agent_runner import stream_agent

    captured: dict = {}

    class _Recorder(BaseAgent):
        def __init__(self) -> None:           # bypass the real ctor — no LLM client
            self.system_prompt = "ctor-prompt"
            self.max_loops = 99
            self.max_token_budget = 99_999

        def run(self, query, tracer=None, on_event=None, **kwargs):
            captured.update(kwargs)
            captured["query"] = query
            return {
                "answer": "stub",
                "exit_reason": "natural",
                "loops": 1,
                "total_cost": 0.0,
                "input_tokens_total": 0,
                "cached_tokens_total": 0,
                "output_tokens_total": 0,
            }

    async with _client_for(app_harness) as (client, app, headers):
        # Stash the patched value for "base" agent overrides.
        await client.patch(
            "/admin/config",
            headers=headers,
            json={
                "updates": {
                    "agent.base.max_loops": 5,
                    "agent.base.max_token_budget": 33_000,
                    "prompt.base_agent": "patched-base-prompt",
                }
            },
        )

        # Drive stream_agent directly (the smoke /agent/stream route
        # path requires the real LLM-bound agent; we want to assert
        # plumbing, not LLM behavior).
        result_future: "asyncio.Future" = asyncio.get_running_loop().create_future()
        recorder = _Recorder()
        gen = stream_agent(
            query="ping",
            kind="base",
            agent=recorder,
            config=app.state.config,
            tracer=None,
            result_future=result_future,
        )
        async for _chunk in gen:
            pass
        await asyncio.wait_for(result_future, timeout=5)

        assert captured["max_loops"] == 5
        assert captured["max_token_budget"] == 33_000
        assert captured["system_prompt"] == "patched-base-prompt"
        assert captured["query"] == "ping"


async def test_in_flight_runner_keeps_pre_patch_snapshot(monkeypatch, app_harness):
    """A PATCH that lands mid-run must NOT mutate the value the runner is using.

    The runner snapshots ``materialize_agent_kwargs`` at call entry, so
    a concurrent PATCH only affects the *next* request. Verify by
    starting a stub agent that blocks on an event, PATCHing while it
    blocks, then unblocking and asserting the recorded value is the
    pre-patch one.
    """
    import asyncio

    from agentic.agent.base import BaseAgent
    from api.runners.agent_runner import stream_agent

    enter_event = asyncio.Event()
    release_event = asyncio.Event()
    captured: dict = {}
    loop = asyncio.get_running_loop()

    class _Blocker(BaseAgent):
        def __init__(self) -> None:
            self.system_prompt = "ctor"
            self.max_loops = 99
            self.max_token_budget = 99_999

        def run(self, query, tracer=None, on_event=None, **kwargs):
            captured.update(kwargs)
            # Tell the test we observed our overrides; then block until
            # the test fires the release event.
            loop.call_soon_threadsafe(enter_event.set)
            fut = asyncio.run_coroutine_threadsafe(release_event.wait(), loop)
            fut.result(timeout=5)
            return {
                "answer": "stub",
                "exit_reason": "natural",
                "loops": 1,
                "total_cost": 0.0,
                "input_tokens_total": 0,
                "cached_tokens_total": 0,
                "output_tokens_total": 0,
            }

    async with _client_for(app_harness) as (client, app, headers):
        # Pre-patch snapshot value.
        await client.patch(
            "/admin/config",
            headers=headers,
            json={"updates": {"agent.base.max_loops": 6}},
        )

        result_future: "asyncio.Future" = loop.create_future()
        async def _drain() -> None:
            async for _ in stream_agent(
                query="x",
                kind="base",
                agent=_Blocker(),
                config=app.state.config,
                result_future=result_future,
            ):
                pass

        drain_task = asyncio.create_task(_drain())
        await asyncio.wait_for(enter_event.wait(), timeout=5)

        # Now PATCH mid-flight. The blocked runner must keep the value
        # it snapshotted before the worker started.
        await client.patch(
            "/admin/config",
            headers=headers,
            json={"updates": {"agent.base.max_loops": 9}},
        )

        release_event.set()
        await asyncio.wait_for(drain_task, timeout=5)

        assert captured["max_loops"] == 6, (
            f"runner observed mid-flight PATCH (got {captured['max_loops']!r})"
        )
        # And the next call sees the new value.
        assert app.state.config.get("agent.base.max_loops") == 9
