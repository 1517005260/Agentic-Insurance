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
    DEFAULT_HIDDEN_RISK_QUERY,
    DEFAULT_POLICY_PARAMS,
    DEFAULT_PRIMARY_FILE_ID,
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
# 1. compare (BaseAgent SSE)
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
# 2. recommend / needs-analysis (BaseAgent SSE)
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
        # discovery step the prompt requires in open-corpus mode).
        list_files_calls = [
            e for e in events if e[0] == "tool_call" and e[1].get("name") == "list_files"
        ]
        assert list_files_calls, "recommend agent must call list_files"
        assert final is not None and final.get("flavor") == "recommend"
        assert final.get("held_policies_count", 0) == 0


@_NEEDS_KEYS
async def test_recommend_gap_analysis_with_held_policies(app_harness):
    """Held-policy mode: the runner should switch to gap-analysis prompt.

    Beyond "some read happened" (which would also pass when the agent
    reads candidate products), assert at least one ``read`` targets a
    held file_id — that's what proves the gap-analysis branch ran the
    "summarize existing coverage first" step the new prompt mandates.
    """
    held = DEFAULT_FILE_IDS[:1]
    held_set = set(held)
    async with _client_for(app_harness) as (client, app, headers):
        events, final = await _drain_sse(
            client,
            "/insurance/recommend/stream",
            {"customer": DEFAULT_CUSTOMER, "held_policies_file_ids": held},
            headers,
            timeout=600.0,
        )
        read_calls = [
            e for e in events
            if e[0] == "tool_call" and e[1].get("name") == "read"
        ]
        assert read_calls, "gap analysis must read at least one policy"
        # ReadTool addresses a unit either via `unit_ids` (where each
        # unit_id is `<file_id>/<page_id>`) or via `file_ids` (file
        # allow-list). Check both shapes for any held file_id reference.
        held_targeted = False
        for _, payload in read_calls:
            args = payload.get("args") or {}
            for fid in args.get("file_ids") or []:
                if isinstance(fid, str) and fid in held_set:
                    held_targeted = True
                    break
            if held_targeted:
                break
            for uid in args.get("unit_ids") or []:
                if not isinstance(uid, str):
                    continue
                if any(uid.startswith(f"{fid}/") for fid in held_set):
                    held_targeted = True
                    break
            if held_targeted:
                break
        assert held_targeted, (
            f"gap analysis must read at least one held file_id "
            f"({held_set}), but read_calls targeted: "
            f"{[c[1].get('args') for c in read_calls]}"
        )
        # Belt-and-braces: the citations event must list a passage
        # whose file_id is in the held set — proves the read actually
        # surfaced a real page rather than a not_found stub. The
        # workbench scaffold always emits ``citations`` before
        # ``final``, so absence here is itself a regression.
        cite_evs = [e for e in events if e[0] == "citations"]
        assert cite_evs, "stream_workbench_agent must emit a citations event"
        items = (cite_evs[-1][1].get("items") or [])
        assert any(
            isinstance(it, dict) and it.get("file_id") in held_set
            for it in items
        ), "citations must include at least one held file_id"
        assert final is not None and final.get("flavor") == "recommend"
        assert final.get("held_policies_count") == len(held)


# ============================================================
# 4. claim check (BaseAgent SSE)
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
# 5. policy calc (BaseAgent + code_run)
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
# 6. hidden-risk (PPR + LLM, no agent loop)
# ============================================================


@_NEEDS_KEYS
async def test_hidden_risk_with_query_streams_token_and_final(app_harness):
    """``/insurance/fraud-ppr/stream`` is now powering the Policy-Review
    hidden-risk tab (URL preserved for trace flavor / persisted history).

    Smoke: SSE bus emits at least one token frame and a terminal final
    that carries the documented schema (mode + subgraph_counts + flavor).
    """
    body = {
        "query": DEFAULT_HIDDEN_RISK_QUERY,
        "file_ids": [DEFAULT_PRIMARY_FILE_ID],
    }
    async with _client_for(app_harness) as (client, app, headers):
        events, final = await _drain_sse(
            client, "/insurance/fraud-ppr/stream", body, headers, timeout=240.0
        )
        # Allow no_seeds / no_graph paths to skip token frames (the prompt
        # in those modes is a one-shot abstain). Otherwise we require the
        # bus pumped at least one token before close.
        if final and final.get("mode") == "ppr":
            assert any(e[0] == "token" for e in events), "ppr mode must stream tokens"
        assert final is not None
        assert final.get("flavor") == "fraud_ppr"
        assert "subgraph_counts" in final
        assert {"seeds", "actived_entities", "passages", "edges"} <= set(
            final["subgraph_counts"].keys()
        )


# ============================================================
# 6b. risk predict (GraphAgent + Sankey side-channel)
# ============================================================


@_NEEDS_KEYS
async def test_risk_predict_emits_risk_subgraph(app_harness):
    """Pre-issuance risk prediction must:
      - drive the GraphAgent through at least one graph_explore call
        (visible as a ``graph_subgraph`` SSE frame), and
      - augment the ``final`` event with a ``risk_subgraph`` payload
        that carries the documented 4-key shape.
    """
    body = {
        "file_id": DEFAULT_PRIMARY_FILE_ID,
        "customer": DEFAULT_CUSTOMER,
        "scenario": "客户考虑在投保 6 个月内出境长期旅游",
    }
    async with _client_for(app_harness) as (client, app, headers):
        events, final = await _drain_sse(
            client, "/insurance/risk-predict/stream", body, headers, timeout=600.0
        )
        # The wrapper relies on stream_agent's GRAPH_SUBGRAPH passthrough;
        # at least one graph_explore call must have happened for the
        # canvas to populate.
        assert any(e[0] == "graph_subgraph" for e in events), (
            "risk-predict must trigger at least one graph_explore call"
        )
        assert final is not None
        assert final.get("flavor") == "risk_predict"
        rs = final.get("risk_subgraph")
        assert isinstance(rs, dict), "final must carry risk_subgraph dict"
        assert {
            "customer_fields", "risk_factors", "triggered_clauses", "edges",
        } <= set(rs.keys())
        # Customer profile contributes age + gender + occupation at minimum,
        # so the column-1 strip must never be empty.
        assert len(rs["customer_fields"]) >= 3
        # triggered_clauses MUST NOT carry ``sup`` — agent's read order
        # (citations[].sup) is a separate namespace from PPR rank, and
        # cross-linking would resolve clicks to wrong passages. Frontend
        # joins on (file_id, page_id) instead.
        for clause in rs["triggered_clauses"]:
            assert "sup" not in clause, (
                f"risk_subgraph.triggered_clauses must not carry sup; got {clause}"
            )
            assert {"id", "file_id", "page_id"} <= set(clause.keys())


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


async def test_chat_session_graph_with_web_rejected(app_harness):
    async with _client_for(app_harness) as (client, app, headers):
        r = await client.post(
            "/chat/sessions",
            headers=headers,
            json={"mode": "agent", "agent_kind": "graph", "web": True},
        )
        assert r.status_code == 422


async def test_agent_stream_graph_with_web_rejected(app_harness):
    async with _client_for(app_harness) as (client, app, headers):
        r = await client.post(
            "/agent/stream",
            headers=headers,
            json={"query": "test", "kind": "graph", "web": True},
        )
        assert r.status_code == 422
