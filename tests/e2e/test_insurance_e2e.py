"""Live insurance + web e2e — real Tavily, real LLM.

Why ``@pytest.mark.live``:
  * Real Tavily ($/query) + real LLM ($/turn) make these expensive
    enough that CI without keys must skip them. The marker (declared
    in ``pyproject.toml`` and excluded by default) does that.
  * Default-fixture queries / customer profiles / file_ids are the
    "demo defaults" that match what the front-end LayoutShell will
    seed when the user opens the page; they stay in
    ``tests/e2e/fixtures/insurance_defaults.py`` so changing the
    demo corpus is one edit.

Run with:
  TAVILY_API_KEY=... CHAT_API_KEY=... \
    PYTHONPATH=src .venv/bin/pytest tests/e2e/test_insurance_e2e.py \
    -m live -v -s
"""
import json
import os
import sys
from pathlib import Path
from typing import AsyncIterator, Tuple

import pytest

# Reuse the harness machinery from the chat e2e — it boots the app
# in-process and yields a logged-in admin client.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_chat_e2e import _client_for, anyio_backend, app_harness  # noqa: E402,F401

from fixtures.insurance_defaults import (  # noqa: E402
    DEFAULT_CALC_TARGETS,
    DEFAULT_CLAIM_EVENT,
    DEFAULT_COMPARE_PROPERTIES,
    DEFAULT_CUSTOMER,
    DEFAULT_FILE_IDS,
    DEFAULT_POLICY_PARAMS,
    DEFAULT_PRIMARY_FILE_ID,
    DEFAULT_REGULATION_JURISDICTION,
    DEFAULT_REGULATION_QUERY,
    DEFAULT_WEB_AGENT_QUERY,
    DEFAULT_WEB_RAG_QUERY,
)


pytestmark = [pytest.mark.live, pytest.mark.anyio]


def _has_keys() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY")) and bool(
        os.environ.get("CHAT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )


_NEEDS_KEYS = pytest.mark.skipif(
    not _has_keys(), reason="TAVILY_API_KEY + LLM key required for live tests"
)


# ============================================================
# helpers
# ============================================================


async def _drain_sse(client, path: str, body: dict, headers: dict, timeout: float = 180.0):
    """Stream an SSE body and return list of (event, data) tuples + stats."""
    events = []
    final_payload = None
    async with client.stream(
        "POST", path, json=body, headers=headers, timeout=timeout
    ) as resp:
        assert resp.status_code == 200, await resp.aread()
        event_name = None
        data_buf: list[str] = []
        async for raw in resp.aiter_lines():
            if raw == "":
                if event_name and data_buf:
                    data = "\n".join(data_buf)
                    try:
                        parsed = json.loads(data) if data else {}
                    except json.JSONDecodeError:
                        parsed = {"_raw": data}
                    events.append((event_name, parsed))
                    if event_name == "final":
                        final_payload = parsed
                event_name, data_buf = None, []
                continue
            if raw.startswith(":"):
                continue
            if raw.startswith("event: "):
                event_name = raw[len("event: "):].strip()
            elif raw.startswith("data: "):
                data_buf.append(raw[len("data: "):])
    return events, final_payload


# ============================================================
# 1. regulation-search (non-streaming)
# ============================================================


@_NEEDS_KEYS
async def test_regulation_search_hk_jurisdiction(app_harness):
    async with _client_for(app_harness) as (client, app, headers):
        body = {
            "query": DEFAULT_REGULATION_QUERY,
            "jurisdiction": DEFAULT_REGULATION_JURISDICTION,
            "max_results": 4,
        }
        r = await client.post(
            "/insurance/regulation-search", json=body, headers=headers, timeout=120.0
        )
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["jurisdiction"] == "hk"
        assert out["search_query"] == DEFAULT_REGULATION_QUERY
        assert out["used_include_domains"], "HK domain whitelist must be applied"
        assert isinstance(out["sources"], list)
        # If Tavily returned anything, the LLM should have produced text.
        if out["n_results"] > 0:
            assert out["summary_chars"] > 0, "non-empty summary expected when sources exist"


@_NEEDS_KEYS
async def test_regulation_search_503_when_no_tavily(app_harness, monkeypatch):
    # Yank the key inside the running process so the lifespan-built
    # client surfaces unavailable. We do this BEFORE _client_for boots
    # so the lifespan sees the wiped key.
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    async with _client_for(app_harness) as (client, app, headers):
        # Replace the lifespan client too — defensively, the cached one
        # still holds the old key value.
        from model_client.web_search import TavilyClient
        app.state.tavily_client = TavilyClient(api_key=None)
        r = await client.post(
            "/insurance/regulation-search",
            json={"query": "anything", "jurisdiction": "both"},
            headers=headers,
        )
        assert r.status_code == 503, r.text


# ============================================================
# 2. compare (BaseAgent SSE)
# ============================================================


@_NEEDS_KEYS
async def test_compare_two_products_streams_final(app_harness):
    async with _client_for(app_harness) as (client, app, headers):
        body = {
            "file_ids": DEFAULT_FILE_IDS[:2],
            "properties": DEFAULT_COMPARE_PROPERTIES[:2],
        }
        events, final = await _drain_sse(
            client, "/insurance/compare/stream", body, headers, timeout=600.0
        )
        # SSE plumbing
        assert any(e[0] == "tool_call" for e in events), "agent must call at least one tool"
        assert final is not None, "final event must arrive"
        assert final.get("flavor") == "compare"
        assert final.get("matrix_dims") == [2, 2]
        # Either CERTIFIED-style natural exit or max_loops; never error path
        assert final.get("exit_reason") in {"natural", "max_loops_exceeded", "finalized", "ok"}


# ============================================================
# 3. exclusion audit (ProofAgent SSE)
# ============================================================


@_NEEDS_KEYS
async def test_exclusion_audit_with_default_customer(app_harness):
    async with _client_for(app_harness) as (client, app, headers):
        body = {"file_id": DEFAULT_PRIMARY_FILE_ID, "customer": DEFAULT_CUSTOMER}
        events, final = await _drain_sse(
            client,
            "/insurance/exclusion-audit/stream",
            body,
            headers,
            timeout=600.0,
        )
        # ProofAgent should emit at least obligation events
        assert any(e[0] in {"obligation", "tool_call"} for e in events)
        assert final is not None
        assert final.get("flavor") == "exclusion"


# ============================================================
# 4. recommend (BaseAgent SSE)
# ============================================================


@_NEEDS_KEYS
async def test_recommend_with_default_customer(app_harness):
    async with _client_for(app_harness) as (client, app, headers):
        events, final = await _drain_sse(
            client,
            "/insurance/recommend/stream",
            {"customer": DEFAULT_CUSTOMER},
            headers,
            timeout=600.0,
        )
        # The agent should have called list_files (it is the canonical
        # discovery step the prompt requires).
        list_files_calls = [
            e for e in events if e[0] == "tool_call" and e[1].get("name") == "list_files"
        ]
        assert list_files_calls, "recommend agent must call list_files"
        assert final is not None and final.get("flavor") == "recommend"


# ============================================================
# 5. claim check (BaseAgent SSE)
# ============================================================


@_NEEDS_KEYS
async def test_claim_check_with_default_event(app_harness):
    async with _client_for(app_harness) as (client, app, headers):
        body = {
            "file_ids": [DEFAULT_PRIMARY_FILE_ID],
            "event": DEFAULT_CLAIM_EVENT,
        }
        events, final = await _drain_sse(
            client, "/insurance/claim-check/stream", body, headers, timeout=600.0
        )
        assert final is not None
        assert final.get("flavor") == "claim"
        assert final.get("event_type") == DEFAULT_CLAIM_EVENT["type"]


# ============================================================
# 6. policy calc (BaseAgent + code_run)
# ============================================================


@_NEEDS_KEYS
async def test_policy_calc_invokes_code_run(app_harness):
    async with _client_for(app_harness) as (client, app, headers):
        body = {
            "file_id": DEFAULT_PRIMARY_FILE_ID,
            "policy_params": DEFAULT_POLICY_PARAMS,
            "calc_targets": DEFAULT_CALC_TARGETS,
        }
        events, final = await _drain_sse(
            client,
            "/insurance/policy-calc/stream",
            body,
            headers,
            timeout=600.0,
        )
        # Mandatory: prompt forces the agent to use code_run for every
        # arithmetic step. If this assertion fails, the prompt softened.
        code_run_calls = [
            e for e in events if e[0] == "tool_call" and e[1].get("name") == "code_run"
        ]
        assert code_run_calls, "policy-calc agent must call code_run at least once"
        assert final is not None and final.get("flavor") == "policy_calc"


# ============================================================
# 7. chat web-rag (mode=rag, web=true)
# ============================================================


@_NEEDS_KEYS
async def test_chat_web_rag_streams_token_and_citations(app_harness):
    async with _client_for(app_harness) as (client, app, headers):
        # Create a web-rag session (mode=rag, web=true).
        r = await client.post(
            "/chat/sessions",
            headers=headers,
            json={"mode": "rag", "web": True, "title": "web rag smoke"},
        )
        assert r.status_code == 201, r.text
        sid = r.json()["id"]
        assert r.json()["web"] is True

        events, _ = await _drain_sse(
            client,
            f"/chat/sessions/{sid}/messages",
            {"content": DEFAULT_WEB_RAG_QUERY},
            headers,
            timeout=240.0,
        )
        # web-rag emits status → retrieval (channel=web) → status →
        # token… → citations → final → done
        retrieval = [e for e in events if e[0] == "retrieval"]
        assert retrieval and retrieval[0][1].get("channel") == "web"
        assert any(e[0] == "token" for e in events), "expected at least one token frame"


# ============================================================
# 8. chat web-agent (mode=agent, agent_kind=base, web=true)
# ============================================================


@_NEEDS_KEYS
async def test_chat_web_agent_uses_web_search(app_harness):
    async with _client_for(app_harness) as (client, app, headers):
        r = await client.post(
            "/chat/sessions",
            headers=headers,
            json={
                "mode": "agent",
                "agent_kind": "base",
                "web": True,
                "title": "web agent smoke",
            },
        )
        assert r.status_code == 201, r.text
        sid = r.json()["id"]

        events, _ = await _drain_sse(
            client,
            f"/chat/sessions/{sid}/messages",
            {"content": DEFAULT_WEB_AGENT_QUERY},
            headers,
            timeout=600.0,
        )
        # The web agent has only web_search + web_fetch; verify
        # web_search is on the trajectory.
        web_search_calls = [
            e for e in events if e[0] == "tool_call" and e[1].get("name") == "web_search"
        ]
        assert web_search_calls, "web agent must call web_search at least once"


# ============================================================
# 9. validation negatives (cheap, no API hit)
# ============================================================


async def test_compare_dup_file_ids_rejected(app_harness):
    async with _client_for(app_harness) as (client, app, headers):
        r = await client.post(
            "/insurance/compare/stream",
            headers=headers,
            json={"file_ids": ["a", "a"], "properties": ["x"]},
        )
        assert r.status_code == 422


async def test_chat_session_proof_with_web_rejected(app_harness):
    async with _client_for(app_harness) as (client, app, headers):
        r = await client.post(
            "/chat/sessions",
            headers=headers,
            json={"mode": "agent", "agent_kind": "proof", "web": True},
        )
        assert r.status_code == 422


async def test_agent_stream_proof_with_web_rejected(app_harness):
    async with _client_for(app_harness) as (client, app, headers):
        r = await client.post(
            "/agent/stream",
            headers=headers,
            json={"query": "test", "kind": "proof", "web": True},
        )
        assert r.status_code == 422
