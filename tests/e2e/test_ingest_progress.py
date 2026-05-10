"""Cheap-tier coverage for the ingestion-progress on_event contract.

Goals:

* ``_ingest_one(on_event=...)`` emits one ``stage`` event per phase
  boundary across page_assets + every supplied builder, in the right
  order (serial mode is deterministic; parallel mode is loose but
  every (stage, phase) pair must appear once).
* Builder failure converts to ``IndexBuildResult(failed=True)`` AND
  emits the ``done`` frame with an ``error`` field — never bubbles.
* ``on_event=None`` keeps the legacy silent path (no events recorded
  by a downstream observer that we wedge in via a counter).
* Callback exceptions are swallowed (the algorithm pipeline must
  not depend on a clean consumer).

We don't run real builders: a stub IndexBuilder records what was asked
and returns a synthetic result. The point of this test is the wire
contract, not the index data.
"""
import asyncio
import importlib
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))




@pytest.fixture
def parse_module():
    return importlib.import_module("pipeline.parse_and_index")


@pytest.fixture
def builders_module():
    return importlib.import_module("ingestion.index.base")


def _make_stub_builder(builders_module, name: str, *, raise_exc: bool = False):
    """Subclass IndexBuilder inline so we don't depend on real index code.

    ABC requires the abstract method to exist on the class at creation
    time, so we materialize ``_build`` in the class body via a fresh
    type per call rather than assigning post-hoc.
    """
    builder_name = name
    should_raise = raise_exc

    def _build(self, file_id, pages):
        if should_raise:
            raise RuntimeError(f"stub failure ({builder_name})")
        return builders_module.IndexBuildResult(
            index_name=builder_name,
            file_id=file_id,
            output_dir="",
            item_count=len(pages),
            skipped_reason=None,
        )

    cls = type(
        f"StubBuilder_{builder_name}",
        (builders_module.IndexBuilder,),
        {"name": builder_name, "_build": _build},
    )
    return cls()


def _fake_parse(monkeypatch, parse_module, n_pages: int = 3):
    """Bypass paddle: feed a synthetic ParseResult-ish object and pre-canned pages."""
    fake_pages = [object()] * n_pages

    def _fake_build_page_assets(parse, persist=True):  # noqa: ARG001
        return list(fake_pages)

    monkeypatch.setattr(parse_module, "build_page_assets", _fake_build_page_assets)

    class _Parse:
        file_id = "stub_file"
        total_pages = n_pages

    return _Parse(), fake_pages


# ---------------------------------------------------------------- tests ----


def test_ingest_one_emits_serial_stage_pairs(parse_module, builders_module, monkeypatch):
    """Serial mode: every (stage, phase) pair appears exactly once, in input order."""
    parse, _ = _fake_parse(monkeypatch, parse_module, n_pages=2)
    b1 = _make_stub_builder(builders_module, "text_dense")
    b2 = _make_stub_builder(builders_module, "bm25")
    b3 = _make_stub_builder(builders_module, "graph")

    events: List[tuple[str, Dict[str, Any]]] = []

    def on_event(event, data):
        events.append((event, dict(data)))

    pages, results = parse_module._ingest_one(
        parse, [b1, b2, b3], parallel_builders=False, on_event=on_event
    )

    assert len(pages) == 2
    assert [r.index_name for r in results] == ["text_dense", "bm25", "graph"]
    assert all(not r.failed for r in results)

    # Expected wire trace.
    expected_pairs = [
        ("page_assets", "start"),
        ("page_assets", "done"),
        ("text_dense", "start"),
        ("text_dense", "done"),
        ("bm25", "start"),
        ("bm25", "done"),
        ("graph", "start"),
        ("graph", "done"),
    ]
    actual_pairs = [(d["stage"], d["phase"]) for ev, d in events if ev == "stage"]
    assert actual_pairs == expected_pairs

    # done frames carry elapsed_ms; start frames don't.
    for ev, d in events:
        if ev != "stage":
            continue
        if d["phase"] == "done":
            assert "elapsed_ms" in d
        else:
            assert "elapsed_ms" not in d


def test_ingest_one_parallel_emits_all_pairs(parse_module, builders_module, monkeypatch):
    """Parallel mode: order across builders is undefined, but every (stage, phase) shows once."""
    parse, _ = _fake_parse(monkeypatch, parse_module, n_pages=2)
    bs = [
        _make_stub_builder(builders_module, "text_dense"),
        _make_stub_builder(builders_module, "vision_dense"),
        _make_stub_builder(builders_module, "bm25"),
        _make_stub_builder(builders_module, "graph"),
    ]

    events: List[tuple[str, Dict[str, Any]]] = []
    lock = threading.Lock()

    def on_event(event, data):
        with lock:
            events.append((event, dict(data)))

    parse_module._ingest_one(parse, bs, parallel_builders=True, on_event=on_event)

    # page_assets always first (runs on the calling thread before fan-out).
    head = [(d["stage"], d["phase"]) for ev, d in events[:2] if ev == "stage"]
    assert head == [("page_assets", "start"), ("page_assets", "done")]

    pairs = {(d["stage"], d["phase"]) for ev, d in events if ev == "stage"}
    for name in ("text_dense", "vision_dense", "bm25", "graph"):
        assert (name, "start") in pairs, f"missing start for {name}"
        assert (name, "done") in pairs, f"missing done for {name}"


def test_ingest_one_builder_failure_emits_error_in_done(parse_module, builders_module, monkeypatch):
    """A raising builder still emits the done frame, with ``error`` populated."""
    parse, _ = _fake_parse(monkeypatch, parse_module, n_pages=1)
    bs = [
        _make_stub_builder(builders_module, "text_dense"),
        _make_stub_builder(builders_module, "graph", raise_exc=True),
    ]

    events: List[tuple[str, Dict[str, Any]]] = []

    def on_event(event, data):
        events.append((event, dict(data)))

    pages, results = parse_module._ingest_one(
        parse, bs, parallel_builders=False, on_event=on_event
    )

    assert len(pages) == 1
    by_name = {r.index_name: r for r in results}
    assert not by_name["text_dense"].failed
    assert by_name["graph"].failed
    assert "stub failure (graph)" in (by_name["graph"].skipped_reason or "")

    graph_done = next(
        d for ev, d in events
        if ev == "stage" and d.get("stage") == "graph" and d.get("phase") == "done"
    )
    assert "error" in graph_done
    assert "stub failure" in graph_done["error"]


def test_ingest_one_silent_when_on_event_none(parse_module, builders_module, monkeypatch):
    """on_event=None keeps the legacy path — no recorded events, no errors."""
    parse, _ = _fake_parse(monkeypatch, parse_module, n_pages=1)
    b = _make_stub_builder(builders_module, "bm25")
    pages, results = parse_module._ingest_one(parse, [b], parallel_builders=False)
    assert len(pages) == 1
    assert results[0].index_name == "bm25"


def test_ingest_one_page_assets_failure_emits_error_done(parse_module, builders_module, monkeypatch):
    """build_page_assets raising must still emit page_assets:done with error."""
    fake_parse = type("P", (), {"file_id": "x", "total_pages": 2})()

    def _boom(parse, persist=True):  # noqa: ARG001
        raise RuntimeError("page assets exploded")

    monkeypatch.setattr(parse_module, "build_page_assets", _boom)

    events = []

    def on_event(event, data):
        events.append((event, dict(data)))

    with pytest.raises(RuntimeError, match="page assets exploded"):
        parse_module._ingest_one(
            fake_parse, [], parallel_builders=False, on_event=on_event
        )

    pa_done = next(
        d for ev, d in events
        if ev == "stage" and d.get("stage") == "page_assets" and d.get("phase") == "done"
    )
    assert "error" in pa_done
    assert "page assets exploded" in pa_done["error"]


def test_parse_and_index_emits_parse_stage(parse_module, builders_module, monkeypatch, tmp_path):
    """parse_and_index emits parse:start/done from inside the pipeline."""
    fake_parse = type("P", (), {"file_id": "x", "total_pages": 3, "batches": []})()

    class _StubParser:
        def parse(self, source, file_id=None, overwrite=False):  # noqa: ARG002
            return fake_parse

    monkeypatch.setattr(parse_module, "build_page_assets", lambda parse, persist=True: [object()] * 3)

    events = []

    def on_event(event, data):
        events.append((event, dict(data)))

    src = tmp_path / "stub.pdf"
    src.write_bytes(b"%PDF stub")

    parse_module.parse_and_index(
        src,
        parser=_StubParser(),
        builders=[],
        parallel_builders=False,
        on_event=on_event,
    )

    pairs = [(d["stage"], d["phase"]) for ev, d in events if ev == "stage"]
    assert pairs[:2] == [("parse", "start"), ("parse", "done")]
    assert pairs[2:4] == [("page_assets", "start"), ("page_assets", "done")]


def test_parse_and_index_parse_failure_emits_error(parse_module, monkeypatch, tmp_path):
    """Parser raising emits parse:done with error before propagating."""

    class _BoomParser:
        def parse(self, source, file_id=None, overwrite=False):  # noqa: ARG002
            raise ValueError("paddle exploded")

    events = []

    def on_event(event, data):
        events.append((event, dict(data)))

    src = tmp_path / "stub.pdf"
    src.write_bytes(b"%PDF stub")

    with pytest.raises(ValueError, match="paddle exploded"):
        parse_module.parse_and_index(
            src,
            parser=_BoomParser(),
            builders=[],
            parallel_builders=False,
            on_event=on_event,
        )

    parse_done = next(
        d for ev, d in events
        if ev == "stage" and d.get("stage") == "parse" and d.get("phase") == "done"
    )
    assert "error" in parse_done
    assert "paddle exploded" in parse_done["error"]


def test_ingest_one_swallows_callback_exceptions(parse_module, builders_module, monkeypatch):
    """A buggy on_event must not break the build chain."""
    parse, _ = _fake_parse(monkeypatch, parse_module, n_pages=1)
    b = _make_stub_builder(builders_module, "bm25")

    calls = {"n": 0}

    def boom(event, data):  # noqa: ARG001
        calls["n"] += 1
        raise RuntimeError("intentional")

    pages, results = parse_module._ingest_one(
        parse, [b], parallel_builders=False, on_event=boom
    )
    assert len(pages) == 1
    assert results[0].index_name == "bm25"
    # 4 events tried: page_assets start/done + bm25 start/done.
    assert calls["n"] == 4


# ------------------------------ ingestion_runner registry --------------------


@pytest.mark.asyncio
async def test_ingestion_runner_registry_lifecycle():
    """register / get / unregister round trip.

    The bus registry is a thin Dict facade; the multi-consumer fan-out
    happens inside :class:`EventBus` (replay_buffered=True). The legacy
    ``claim_stream`` single-shot guard is gone — a second subscriber to
    an active job is now legal so the FilesPage minimized-ingest chip
    can reopen the progress drawer mid-flight.
    """
    runner = importlib.import_module("api.runners.ingestion_runner")
    events_mod = importlib.import_module("api.runners.events")

    bus = events_mod.EventBus(loop=asyncio.get_running_loop(), replay_buffered=True)
    runner.register_bus(99999, bus)
    assert runner.get_bus(99999) is bus

    runner.unregister_bus(99999)
    assert runner.get_bus(99999) is None


@pytest.mark.asyncio
async def test_ingestion_runner_wait_for_bus_finds_late_register():
    """wait_for_bus: bus appears mid-poll → returns it."""
    runner = importlib.import_module("api.runners.ingestion_runner")
    events_mod = importlib.import_module("api.runners.events")

    loop = asyncio.get_running_loop()
    bus = events_mod.EventBus(loop=loop)

    async def _delayed_register():
        await asyncio.sleep(0.15)
        runner.register_bus(88888, bus)

    asyncio.create_task(_delayed_register())
    found = await runner.wait_for_bus(88888, timeout=1.0, poll_ms=50)
    assert found is bus
    runner.unregister_bus(88888)


@pytest.mark.asyncio
async def test_ingestion_runner_wait_for_bus_times_out():
    """wait_for_bus: nobody registers → returns None inside the timeout."""
    runner = importlib.import_module("api.runners.ingestion_runner")
    found = await runner.wait_for_bus(77777, timeout=0.3, poll_ms=50)
    assert found is None
