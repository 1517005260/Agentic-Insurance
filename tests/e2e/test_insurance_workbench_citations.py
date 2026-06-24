"""Cheap-tier coverage for the workbench citation channel.

Stubs out the BaseAgent / ProofAgent ``run`` entry point with a
canned event sequence so the test can run without an LLM key. The
runner under test is the real ``stream_workbench_agent`` helper —
the assertions cover its contract:

* one ``citations`` event per run, with one item per distinct
  ``(file_id, page_id)`` returned by ``read``;
* sup labels start at 1 and increment in first-seen order;
* the ``citations`` frame precedes ``final`` and follows every
  ``tool_result``;
* the agent-internal ``final`` frame (BaseAgent / ProofAgent both
  emit one before returning) is suppressed; only the runner's
  flavor-tagged ``final`` reaches the wire;
* ``_full_result`` (the internal hand-off field) never reaches the
  SSE wire;
* the exception path still pushes a ``citations`` frame so the
  frontend can clear stale state.
"""
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from agentic.agent.base import BaseAgent
from agentic.agent.proof_agent import ProofAgent, ProofRunResult
from api.runners._workbench import stream_workbench_agent
from config.config_store import ConfigStore


pytestmark = [pytest.mark.anyio]


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _ok(observation_type: str, **fields) -> str:
    payload = {"observation_type": observation_type, "ok": True, **fields}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class _StubBaseAgent(BaseAgent):
    """BaseAgent shape with a scripted run().

    Subclasses ``BaseAgent`` so the runner's isinstance guard accepts
    it; bypasses the real __init__ entirely (no LLM client / tool
    registry / tokenizer) since ``stream_workbench_agent`` only ever
    calls .run(). The default scripted sequence ends with a ``final``
    event so tests cover the agent-final-swallow path the real loop
    triggers via :func:`agentic.agent.base.BaseAgent.run`.
    """

    def __init__(
        self,
        scripted_events: List[Tuple[str, Dict[str, Any]]],
        answer: str = "stub answer [^1] [^2]",
        raise_after: bool = False,
    ) -> None:
        # Intentionally do not call super().__init__: the runner only
        # uses .run() and the isinstance check.
        self._scripted_events = scripted_events
        self._answer = answer
        self._raise_after = raise_after

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
        # Mirror the real BaseAgent: emit a ``final`` event right
        # before returning. The runner must swallow this in favor of
        # its own flavor-tagged ``final``.
        if on_event is not None:
            on_event("final", {"answer": self._answer, "exit_reason": "natural", "loops": 1, "total_cost": 0.0})
        if self._raise_after:
            raise RuntimeError("scripted-failure")
        return {
            "answer": self._answer,
            "exit_reason": "natural",
            "loops": 1,
            "total_cost": 0.0,
            "input_tokens_total": 0,
            "cached_tokens_total": 0,
            "output_tokens_total": 0,
        }


class _StubProofAgent(ProofAgent):
    """ProofAgent shape with a scripted run() returning a ProofRunResult."""

    def __init__(self, scripted_events: List[Tuple[str, Dict[str, Any]]], answer: str = "stub proof answer [^1]") -> None:
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
    ) -> ProofRunResult:
        for event_name, data in self._scripted_events:
            if on_event is not None:
                on_event(event_name, data)
        if on_event is not None:
            on_event(
                "final",
                {
                    "answer": self._answer,
                    "decision": "CERTIFIED",
                    "exit_reason": "finalized",
                    "loops": 2,
                    "total_cost": 0.0,
                },
            )
        return ProofRunResult(
            decision="CERTIFIED",
            answer=self._answer,
            obligations=[],
            claims=[],
            candidate_gaps=[],
            trajectory=[],
            loops=2,
            total_cost=0.0,
            exit_reason="finalized",
        )


async def _drain(gen) -> List[Tuple[str, Dict[str, Any]]]:
    """Decode the SSE byte stream to a list of (event, data) pairs."""
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


# ---------------------------------------------------------------- tests


async def test_workbench_emits_citations_event_with_dedup_and_ordering():
    """read tool with two distinct pages yields two CitationItems, sup=1,2."""
    page_a_envelope = _ok(
        "PageReadObservation",
        unit_type="page",
        units=[
            {
                "unit_id": "f1::p001",
                "file_id": "f1",
                "page_id": "p001",
                "page_number": 1,
                "text": "Page A text. Limit: HKD 5000.",
            }
        ],
        summary={"requested": 1, "new": 1},
    )
    # Second call repeats page A (must dedup) and adds page B.
    page_ab_envelope = _ok(
        "PageReadObservation",
        unit_type="page",
        units=[
            {
                "unit_id": "f1::p001",
                "file_id": "f1",
                "page_id": "p001",
                "page_number": 1,
                "text": "Page A text. Limit: HKD 5000.",
            },
            {
                "unit_id": "f1::p002",
                "file_id": "f1",
                "page_id": "p002",
                "page_number": 2,
                "text": "Page B excludes pre-existing conditions.",
            },
        ],
        summary={"requested": 2, "new": 1},
    )
    scripted = [
        ("status", {"phase": "thinking", "loop": 1}),
        ("tool_call", {"loop": 1, "name": "read", "args": {"unit_ids": ["f1::p001"]}}),
        (
            "tool_result",
            {
                "loop": 1,
                "name": "read",
                "preview": page_a_envelope[:300],
                "retrieved_tokens": 12,
                "error": None,
                "_full_result": page_a_envelope,
            },
        ),
        ("tool_call", {"loop": 1, "name": "read", "args": {"unit_ids": ["f1::p001", "f1::p002"]}}),
        (
            "tool_result",
            {
                "loop": 1,
                "name": "read",
                "preview": page_ab_envelope[:300],
                "retrieved_tokens": 30,
                "error": None,
                "_full_result": page_ab_envelope,
            },
        ),
    ]
    agent = _StubBaseAgent(scripted)
    config = ConfigStore.defaults_only()

    events = await _drain(
        stream_workbench_agent(
            user_prompt="dummy",
            agent=agent,
            kind="base",
            config=config,
            prompt_key="prompt.compare",
            flavor="compare",
        )
    )

    # ---- citation event shape ----
    citation_frames = [d for n, d in events if n == "citations"]
    assert len(citation_frames) == 1, f"expected exactly one citations frame, got {len(citation_frames)}"
    items = citation_frames[0]["items"]
    assert len(items) == 2, f"expected 2 deduped items, got {len(items)}"
    assert [it["sup"] for it in items] == [1, 2]
    assert items[0]["file_id"] == "f1" and items[0]["page_id"] == "p001"
    assert items[0]["page_number"] == 1
    assert items[0]["page_preview"] and "Page A" in items[0]["page_preview"]
    assert items[1]["file_id"] == "f1" and items[1]["page_id"] == "p002"
    assert items[1]["page_number"] == 2

    # ---- ordering: tool_result… → citations → final → done ----
    names = [n for n, _ in events]
    assert "citations" in names and "final" in names and "done" in names
    assert names.index("citations") < names.index("final") < names.index("done")
    last_tool_result = max(i for i, n in enumerate(names) if n == "tool_result")
    assert last_tool_result < names.index("citations")

    # ---- exactly one final on the wire (agent's internal final swallowed) ----
    final_count = sum(1 for n in names if n == "final")
    assert final_count == 1, f"expected exactly one final frame, got {final_count}: {names}"

    # ---- _full_result must NOT leak into any frame ----
    for name, data in events:
        assert "_full_result" not in data, f"{name} frame leaked _full_result"

    # ---- tool_result frames carry is_evidence=True for read ----
    tr_frames = [d for n, d in events if n == "tool_result"]
    assert all(d.get("is_evidence") is True for d in tr_frames)

    # ---- final summary surfaces citations_count ----
    final_frame = next(d for n, d in events if n == "final")
    assert final_frame.get("citations_count") == 2
    assert final_frame.get("flavor") == "compare"


async def test_workbench_emits_empty_citations_when_no_read_called():
    """No read tool → empty citations list, frame still emitted."""
    scripted = [
        ("status", {"phase": "thinking", "loop": 1}),
        ("tool_call", {"loop": 1, "name": "list_files", "args": {}}),
        (
            "tool_result",
            {
                "loop": 1,
                "name": "list_files",
                "preview": "[]",
                "retrieved_tokens": 0,
                "error": None,
                "_full_result": "[]",
            },
        ),
    ]
    agent = _StubBaseAgent(scripted, answer="no evidence available")
    config = ConfigStore.defaults_only()

    events = await _drain(
        stream_workbench_agent(
            user_prompt="dummy",
            agent=agent,
            kind="base",
            config=config,
            prompt_key="prompt.recommend",
            flavor="recommend",
        )
    )

    citation_frames = [d for n, d in events if n == "citations"]
    assert len(citation_frames) == 1
    assert citation_frames[0]["items"] == []

    # list_files is NOT an evidence tool — frame must be tagged False.
    tr_frames = [d for n, d in events if n == "tool_result"]
    assert tr_frames and all(d.get("is_evidence") is False for d in tr_frames)


async def test_workbench_skips_malformed_read_envelope_without_crashing():
    """Junk tool result is logged-and-skipped; citation list stays empty."""
    scripted = [
        ("tool_call", {"loop": 1, "name": "read", "args": {}}),
        (
            "tool_result",
            {
                "loop": 1,
                "name": "read",
                "preview": "not json",
                "retrieved_tokens": 0,
                "error": "tool_dispatch_exception",
                "_full_result": "this is definitely not json",
            },
        ),
    ]
    agent = _StubBaseAgent(scripted, answer="failure")
    config = ConfigStore.defaults_only()

    events = await _drain(
        stream_workbench_agent(
            user_prompt="dummy",
            agent=agent,
            kind="base",
            config=config,
            prompt_key="prompt.claim_check",
            flavor="claim",
        )
    )
    citation_frames = [d for n, d in events if n == "citations"]
    assert len(citation_frames) == 1
    assert citation_frames[0]["items"] == []


async def test_workbench_proof_kind_emits_citations_and_swallows_agent_final():
    """ProofAgent path: same citation contract, decision in final."""
    page_envelope = _ok(
        "PageReadObservation",
        unit_type="page",
        units=[
            {
                "unit_id": "f9::p010",
                "file_id": "f9",
                "page_id": "p010",
                "page_number": 10,
                "text": "Exclusion clause: war and nuclear events.",
            }
        ],
        summary={"requested": 1, "new": 1},
    )
    scripted = [
        ("status", {"phase": "thinking", "loop": 1}),
        ("tool_call", {"loop": 1, "name": "read", "args": {"unit_ids": ["f9::p010"]}}),
        (
            "tool_result",
            {
                "loop": 1,
                "name": "read",
                "preview": page_envelope[:300],
                "observation_id": "obs_0001",
                "must_finalize_next": False,
                "_full_result": page_envelope,
            },
        ),
    ]
    agent = _StubProofAgent(scripted)
    config = ConfigStore.defaults_only()

    events = await _drain(
        stream_workbench_agent(
            user_prompt="dummy",
            agent=agent,
            kind="proof",
            config=config,
            prompt_key="prompt.exclusion_audit",
            flavor="exclusion",
        )
    )

    citation_frames = [d for n, d in events if n == "citations"]
    assert len(citation_frames) == 1
    items = citation_frames[0]["items"]
    assert len(items) == 1 and items[0]["sup"] == 1
    assert items[0]["file_id"] == "f9" and items[0]["page_number"] == 10

    final_frames = [d for n, d in events if n == "final"]
    assert len(final_frames) == 1, "agent-internal final must be swallowed"
    assert final_frames[0]["flavor"] == "exclusion"
    assert final_frames[0]["decision"] == "CERTIFIED"
    assert final_frames[0]["citations_count"] == 1


async def test_workbench_pushes_citations_even_when_agent_raises():
    """Exception path: citations frame still emitted before error/done."""
    page_envelope = _ok(
        "PageReadObservation",
        unit_type="page",
        units=[
            {
                "unit_id": "f3::p001",
                "file_id": "f3",
                "page_id": "p001",
                "page_number": 1,
                "text": "Partial evidence captured before crash.",
            }
        ],
        summary={"requested": 1, "new": 1},
    )
    scripted = [
        ("tool_call", {"loop": 1, "name": "read", "args": {}}),
        (
            "tool_result",
            {
                "loop": 1,
                "name": "read",
                "preview": page_envelope[:300],
                "retrieved_tokens": 6,
                "error": None,
                "_full_result": page_envelope,
            },
        ),
    ]
    agent = _StubBaseAgent(scripted, raise_after=True)
    config = ConfigStore.defaults_only()

    events = await _drain(
        stream_workbench_agent(
            user_prompt="dummy",
            agent=agent,
            kind="base",
            config=config,
            prompt_key="prompt.policy_calc",
            flavor="policy_calc",
        )
    )
    names = [n for n, _ in events]
    assert "citations" in names and "error" in names and "done" in names
    # Frame ordering: citations BEFORE error so the frontend can hydrate
    # the drawer with whatever evidence the agent did manage to read.
    assert names.index("citations") < names.index("error") < names.index("done")
    items = next(d for n, d in events if n == "citations")["items"]
    assert len(items) == 1 and items[0]["file_id"] == "f3"


async def test_workbench_handles_passage_units():
    """Passage / table_row read envelopes follow the same units shape."""
    passage_envelope = _ok(
        "PassageReadObservation",
        unit_type="passage",
        units=[
            {
                "unit_id": "psg_42",
                "file_id": "f2",
                "page_id": "p007",
                "page_number": 7,
                "parent_section_id": "sec_a",
                "block_label": "para",
                "text": "Coverage is limited to 30 days post-discharge.",
            }
        ],
        summary={"requested": 1, "not_found": 0},
    )
    scripted = [
        ("tool_call", {"loop": 1, "name": "read", "args": {"unit_type": "passage"}}),
        (
            "tool_result",
            {
                "loop": 1,
                "name": "read",
                "preview": passage_envelope[:300],
                "retrieved_tokens": 8,
                "error": None,
                "_full_result": passage_envelope,
            },
        ),
    ]
    agent = _StubBaseAgent(scripted, answer="passage answer [^1]")
    config = ConfigStore.defaults_only()

    events = await _drain(
        stream_workbench_agent(
            user_prompt="dummy",
            agent=agent,
            kind="base",
            config=config,
            prompt_key="prompt.exclusion_audit",
            flavor="exclusion",
        )
    )
    items = next(d for n, d in events if n == "citations")["items"]
    assert len(items) == 1
    assert items[0]["sup"] == 1
    assert items[0]["file_id"] == "f2"
    assert items[0]["page_id"] == "p007"
    assert items[0]["page_number"] == 7
