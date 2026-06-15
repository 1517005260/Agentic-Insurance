"""The workbench serial gate must actually serialize overlapping agent runs.

The insurance workbench endpoints (compare / recommend / claim-check / ...)
share one flaky upstream relay; overlapping runs hammer it with parallel
connections it drops under load. ``stream_workbench_agent`` gates runs
through ``WORKBENCH_AGENT_MAX_CONCURRENCY`` (default 1 = serial). These
tests drive the real runner entry with a fake agent that records peak
concurrency, asserting the gate holds (and that it lifts when disabled).
"""
import asyncio
import threading
import time

import api.runners._workbench as wb
from agentic.agent.base import BaseAgent


class _Tracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def __enter__(self) -> None:
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def __exit__(self, *exc) -> None:
        with self._lock:
            self.current -= 1


class _ConcurrencyAgent(BaseAgent):
    """A BaseAgent whose run() records how many runs overlap in real time."""

    def __init__(self, tracker: _Tracker) -> None:  # noqa: D401 — no super().__init__
        self._tracker = tracker

    def run(self, user_prompt, *, tracer=None, on_event=None, cancel_check=None, **kwargs):
        with self._tracker:
            time.sleep(0.15)  # long enough that ungated runs would overlap
        return {"answer": "ok", "exit_reason": "natural", "loops": 1, "total_cost": 0.0}


class _FakeConfig:
    def get(self, key):
        return "system prompt"

    def materialize_agent_kwargs(self, kind):
        return {}

    def citation_preview_chars(self):
        return 200


async def _drain(agent, config) -> None:
    async for _ in wb.stream_workbench_agent(
        user_prompt="q",
        agent=agent,
        kind="base",
        config=config,
        prompt_key="prompt.compare",
        flavor="compare",
    ):
        pass


def _run_two_concurrently(max_concurrency: int, monkeypatch) -> int:
    monkeypatch.setattr(wb, "WORKBENCH_AGENT_MAX_CONCURRENCY", max_concurrency)
    wb._agent_gate = None  # rebuild on the loop this scenario creates
    tracker = _Tracker()
    cfg = _FakeConfig()
    agents = [_ConcurrencyAgent(tracker), _ConcurrencyAgent(tracker)]

    async def scenario():
        await asyncio.gather(*(_drain(a, cfg) for a in agents))

    try:
        asyncio.run(scenario())
    finally:
        wb._agent_gate = None
    return tracker.peak


def test_serial_gate_prevents_overlap(monkeypatch):
    assert _run_two_concurrently(1, monkeypatch) == 1


def test_gate_disabled_allows_overlap(monkeypatch):
    # max_concurrency < 1 disables the gate, so the two runs overlap.
    assert _run_two_concurrently(0, monkeypatch) == 2
