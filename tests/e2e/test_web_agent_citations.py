"""Cheap-tier coverage for the web-agent citation channel.

Stubs out ``BaseAgent.run`` with a canned event sequence so the test
runs without a Tavily key or LLM. The runner under test is the real
:func:`api.runners.agent_runner.stream_agent` — assertions cover its
contract for ``kind="web"``:

* the model's ``## Sources`` section is the authoritative sup → url
  legend; ``citations.items`` are sorted by sup;
* fall back to first-seen tool order when the model didn't write a
  Sources section;
* one ``citations`` event per run (always emitted, even when empty);
* citations precede final and final precedes done (``citations →
  final → done``);
* ``_full_result`` is stripped from every SSE frame and the
  agent-internal ``final`` is swallowed in ``kind="web"``;
* web_fetch redirect: request URL aliased to final_url so the model
  can cite either form;
* ``kind="base"`` / ``"graph"`` keep the chat contract
  (no citations event — frontend reverse-derives evidence chips).
"""
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from agentic.agent.base import BaseAgent
from api.runners.agent_runner import stream_agent
from config.config_store import ConfigStore


pytestmark = [pytest.mark.anyio]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _ok(observation_type: str, **fields) -> str:
    payload = {"observation_type": observation_type, "ok": True, **fields}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class _StubBaseAgent(BaseAgent):
    """Scripted BaseAgent for the SSE contract tests.

    Bypasses ``__init__`` because ``stream_agent`` only ever calls
    ``.run()`` and the isinstance check; pulling in the real LLM /
    tool registry would require live keys.
    """

    def __init__(self, scripted_events: List[Tuple[str, Dict[str, Any]]], answer: str = "stub answer [^1]") -> None:
        self._scripted_events = scripted_events
        self._answer = answer

    def run(
        self,
        query: str,
        tracer: Optional[Any] = None,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        *,
        max_loops: Optional[int] = None,
        max_token_budget: Optional[int] = None,
        system_prompt: Optional[str] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        for event_name, data in self._scripted_events:
            if on_event is not None:
                on_event(event_name, data)
        # BaseAgent.run normally emits a ``final`` frame right before
        # returning. Stub the same shape so the runner has something to
        # swallow (kind="web" path) or forward (other kinds).
        if on_event is not None:
            on_event(
                "final",
                {
                    "answer": self._answer,
                    "exit_reason": "natural",
                    "loops": 1,
                    "total_cost": 0.0,
                },
            )
        return {
            "answer": self._answer,
            "exit_reason": "natural",
            "loops": 1,
            "total_cost": 0.0,
            "input_tokens_total": 0,
            "cached_tokens_total": 0,
            "output_tokens_total": 0,
        }


async def _drain(gen) -> List[Tuple[str, Dict[str, Any]]]:
    raw = b""
    async for chunk in gen:
        raw += chunk
    events: List[Tuple[str, Dict[str, Any]]] = []
    event_name: Optional[str] = None
    data_lines: List[str] = []
    for line in raw.decode("utf-8").splitlines():
        if line == "":
            if event_name and data_lines:
                try:
                    events.append((event_name, json.loads("\n".join(data_lines))))
                except json.JSONDecodeError:
                    events.append((event_name, {"_raw": "\n".join(data_lines)}))
            event_name, data_lines = None, []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event: "):
            event_name = line[len("event: "):].strip()
        elif line.startswith("data: "):
            data_lines.append(line[len("data: "):])
    return events


# ----------------------------------------------------------------- tests


async def test_web_agent_legend_from_sources_section_drives_sup_order():
    """Model's ``## Sources`` section authoritative; tool order doesn't matter."""
    # web_search returns three URLs in (A, B, C) order …
    search_envelope = _ok(
        "WebSearchObservation",
        query="HK insurance",
        n_results=3,
        results=[
            {
                "title": "IA HK overview",
                "url": "https://www.ia.org.hk/en/index.html",
                "snippet": "Insurance Authority overview.",
                "score": 0.91,
                "published_date": "2024-09-01",
            },
            {
                "title": "HKMA pension page",
                "url": "https://www.hkma.gov.hk/eng/pension",
                "snippet": "MPF + ORSO scheme summary.",
                "score": 0.82,
            },
            {
                "title": "CSRC notice",
                "url": "https://www.csrc.gov.cn/notice",
                "snippet": "证监会公告全文",
                "score": 0.71,
            },
        ],
    )
    # … but the model cites them in (B, C) order, skipping A entirely.
    answer = (
        "MPF 与 HKMA 监管 [^1]. 证监会监管内地业务 [^2].\n\n"
        "## Sources\n"
        "[^1] HKMA pension — https://www.hkma.gov.hk/eng/pension\n"
        "[^2] CSRC notice — https://www.csrc.gov.cn/notice\n"
    )
    scripted = [
        ("tool_call", {"loop": 1, "name": "web_search", "args": {"query": "HK insurance"}}),
        (
            "tool_result",
            {
                "loop": 1,
                "name": "web_search",
                "preview": search_envelope[:300],
                "retrieved_tokens": 200,
                "error": None,
                "_full_result": search_envelope,
            },
        ),
    ]
    agent = _StubBaseAgent(scripted, answer=answer)
    config = ConfigStore.defaults_only()

    events = await _drain(
        stream_agent(query="HK insurance", kind="web", agent=agent, config=config)
    )

    citation_frames = [d for n, d in events if n == "citations"]
    assert len(citation_frames) == 1
    items = citation_frames[0]["items"]
    # Sup numbering follows the model's section, NOT tool first-seen.
    assert [it["sup"] for it in items] == [1, 2]
    assert items[0]["url"] == "https://www.hkma.gov.hk/eng/pension"
    assert items[1]["url"] == "https://www.csrc.gov.cn/notice"
    # Title / snippet / score get hydrated from the URL pool, not the
    # legend's possibly-truncated label.
    assert items[0]["title"] == "HKMA pension page"
    assert items[0]["score"] == 0.82
    assert items[1]["snippet"] and "证监会" in items[1]["snippet"]


async def test_web_agent_falls_back_to_first_seen_when_no_sources_section():
    search_envelope = _ok(
        "WebSearchObservation",
        results=[
            {"title": "T1", "url": "https://example.com/a", "snippet": "A", "score": 0.5},
            {"title": "T2", "url": "https://example.com/b", "snippet": "B", "score": 0.4},
        ],
    )
    scripted = [
        ("tool_call", {"loop": 1, "name": "web_search", "args": {"query": "x"}}),
        (
            "tool_result",
            {
                "loop": 1,
                "name": "web_search",
                "preview": search_envelope[:300],
                "retrieved_tokens": 50,
                "error": None,
                "_full_result": search_envelope,
            },
        ),
    ]
    # Answer omits Sources section; runner must fall back to pool order.
    agent = _StubBaseAgent(scripted, answer="Just an answer with [^1] [^2].")
    config = ConfigStore.defaults_only()
    events = await _drain(
        stream_agent(query="x", kind="web", agent=agent, config=config)
    )
    items = next(d for n, d in events if n == "citations")["items"]
    assert [it["sup"] for it in items] == [1, 2]
    assert [it["url"] for it in items] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


async def test_web_agent_aliases_redirect_url_against_final_url():
    """``web_fetch`` redirect: request URL maps onto the same pool entry."""
    search_envelope = _ok(
        "WebSearchObservation",
        results=[
            {"title": "Legacy", "url": "https://example.com/legacy", "snippet": "old", "score": 0.5},
        ],
    )
    # web_fetch was asked for /legacy but the server 301'd to /new.
    fetch_envelope = _ok(
        "WebFetchObservation",
        url="https://example.com/new",
        title="New URL",
        text="canonical body",
        chars=14,
        truncated=False,
    )
    answer = (
        "See discussion [^1] and detail [^2].\n\n"
        "## Sources\n"
        # Cite the requested URL (pre-redirect) — must still resolve.
        "[^1] Legacy — https://example.com/legacy\n"
        # Cite the final URL — same logical entry but a different key.
        "[^2] New URL — https://example.com/new\n"
    )
    scripted = [
        ("tool_call", {"loop": 1, "name": "web_search", "args": {"query": "x"}}),
        (
            "tool_result",
            {
                "loop": 1,
                "name": "web_search",
                "preview": search_envelope[:300],
                "retrieved_tokens": 10,
                "error": None,
                "_full_result": search_envelope,
            },
        ),
        ("tool_call", {"loop": 1, "name": "web_fetch", "args": {"url": "https://example.com/legacy"}}),
        (
            "tool_result",
            {
                "loop": 1,
                "name": "web_fetch",
                "preview": fetch_envelope[:300],
                "retrieved_tokens": 14,
                "error": None,
                "_full_result": fetch_envelope,
            },
        ),
    ]
    agent = _StubBaseAgent(scripted, answer=answer)
    config = ConfigStore.defaults_only()
    events = await _drain(
        stream_agent(query="x", kind="web", agent=agent, config=config)
    )
    items = next(d for n, d in events if n == "citations")["items"]
    assert len(items) == 2
    # Alias merge: both sup labels resolve to the same canonical
    # entry, so chip URLs are the final post-redirect form. The
    # frontend Drawer thus opens a working URL even when the model
    # cited the pre-redirect legacy form.
    assert all(it["url"] == "https://example.com/new" for it in items)
    # Title comes from the web_fetch envelope (final URL won the merge);
    # score/published_date carried forward from web_search would be
    # preserved in entry — assert at least the title hydrated.
    assert items[0]["title"] == "New URL"
    assert items[1]["title"] == "New URL"


async def test_web_agent_emits_empty_citations_when_no_web_tool_called():
    scripted = [("status", {"phase": "thinking", "loop": 1})]
    agent = _StubBaseAgent(scripted, answer="no sources")
    config = ConfigStore.defaults_only()

    events = await _drain(
        stream_agent(query="trivial", kind="web", agent=agent, config=config)
    )
    citation_frames = [d for n, d in events if n == "citations"]
    assert len(citation_frames) == 1
    assert citation_frames[0]["items"] == []


async def test_web_agent_skips_malformed_envelope_without_crashing():
    scripted = [
        ("tool_call", {"loop": 1, "name": "web_search", "args": {"query": "x"}}),
        (
            "tool_result",
            {
                "loop": 1,
                "name": "web_search",
                "preview": "not json",
                "retrieved_tokens": 0,
                "error": "fetch_error",
                "_full_result": "{ malformed",
            },
        ),
    ]
    agent = _StubBaseAgent(scripted, answer="failure path")
    config = ConfigStore.defaults_only()
    events = await _drain(
        stream_agent(query="x", kind="web", agent=agent, config=config)
    )
    citation_frames = [d for n, d in events if n == "citations"]
    assert len(citation_frames) == 1
    assert citation_frames[0]["items"] == []


async def test_web_agent_ordering_is_citations_then_final_then_done():
    """citations precedes the runner-emitted final; agent-internal final is swallowed."""
    search_envelope = _ok(
        "WebSearchObservation",
        results=[{"title": "T", "url": "https://example.com/a", "snippet": "A", "score": 0.5}],
    )
    scripted = [
        ("tool_call", {"loop": 1, "name": "web_search", "args": {"query": "x"}}),
        (
            "tool_result",
            {
                "loop": 1,
                "name": "web_search",
                "preview": search_envelope[:300],
                "retrieved_tokens": 10,
                "error": None,
                "_full_result": search_envelope,
            },
        ),
    ]
    agent = _StubBaseAgent(scripted, answer="answer with [^1].")
    config = ConfigStore.defaults_only()
    events = await _drain(
        stream_agent(query="x", kind="web", agent=agent, config=config)
    )
    names = [n for n, _ in events]
    # exactly one final frame (runner-emitted) — the stub's own final
    # must be swallowed by wrapped_on_event for kind="web".
    assert names.count("final") == 1
    citations_idx = names.index("citations")
    final_idx = names.index("final")
    done_idx = names.index("done")
    assert citations_idx < final_idx < done_idx
    # _full_result never leaks out
    for name, data in events:
        assert "_full_result" not in data, f"{name} leaked _full_result"


async def test_chat_kind_base_emits_read_citations_and_forwards_final():
    """Plain chat agent: read units flush one citations frame before final."""
    read_envelope = _ok(
        "PageReadObservation",
        unit_type="page",
        units=[{"unit_id": "f1::p001", "file_id": "f1", "page_id": "p001", "page_number": 1, "text": "stub"}],
    )
    scripted = [
        ("tool_call", {"loop": 1, "name": "read", "args": {"unit_ids": ["f1::p001"]}}),
        (
            "tool_result",
            {
                "loop": 1,
                "name": "read",
                "preview": read_envelope[:300],
                "retrieved_tokens": 1,
                "error": None,
                "_full_result": read_envelope,
            },
        ),
    ]
    agent = _StubBaseAgent(scripted, answer="answer body")
    config = ConfigStore.defaults_only()
    events = await _drain(
        stream_agent(query="x", kind="base", agent=agent, config=config)
    )
    names = [n for n, _ in events]
    # Local kinds (base/graph) flush read-unit citations once, before
    # the single forwarded final.
    assert names.index("citations") < names.index("final")
    assert names.count("final") == 1
    for name, data in events:
        assert "_full_result" not in data, f"{name} leaked _full_result"


async def test_web_agent_pushes_citations_even_when_agent_raises():
    """Error path still flushes accumulated URLs before close(error)."""
    search_envelope = _ok(
        "WebSearchObservation",
        results=[{"title": "T", "url": "https://example.com/a", "snippet": "A", "score": 0.5}],
    )

    class _RaisingAgent(_StubBaseAgent):
        def run(self, *args, on_event=None, **kwargs):
            if on_event is not None:
                on_event("tool_call", {"loop": 1, "name": "web_search", "args": {"query": "x"}})
                on_event(
                    "tool_result",
                    {
                        "loop": 1,
                        "name": "web_search",
                        "preview": search_envelope[:300],
                        "retrieved_tokens": 10,
                        "error": None,
                        "_full_result": search_envelope,
                    },
                )
            raise RuntimeError("boom mid-run")

    agent = _RaisingAgent([], answer="never returned")
    config = ConfigStore.defaults_only()
    events = await _drain(
        stream_agent(query="x", kind="web", agent=agent, config=config)
    )

    citation_frames = [d for n, d in events if n == "citations"]
    assert len(citation_frames) == 1
    items = citation_frames[0]["items"]
    # No Sources section to parse on the error path → fall back to pool order.
    assert len(items) == 1
    assert items[0]["url"] == "https://example.com/a"

    names = [n for n, _ in events]
    assert "citations" in names and "done" in names
    assert names.index("citations") < names.index("done")
    if "error" in names:
        assert names.index("citations") < names.index("error")
