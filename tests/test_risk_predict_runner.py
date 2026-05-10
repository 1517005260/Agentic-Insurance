"""Pure-Python unit coverage for ``risk_predict_runner``.

Live e2e (``tests/e2e/test_insurance_e2e.py::test_risk_predict_emits_risk_subgraph``)
needs a real LLM key + an indexed corpus, so it's marked ``live`` and
skipped by default. The byte-level SSE interceptor is the highest-risk
component and deserves coverage that runs in CI without external deps —
that's what this file does.

Three concerns:

1. **risk_subgraph builder shape** (``_build_risk_subgraph``) — column
   counts, edge model honesty (no fabricated bipartite), correct
   filtering of noise entities.
2. **frame interceptor** (``_maybe_augment_final``) — pass-through for
   heartbeat / token frames, augmentation only on ``final``,
   graceful fallback on malformed JSON.
3. **end-to-end stream wrapper** (``stream_risk_predict``) under a
   ``stream_agent`` mock — covers cross-chunk frame buffering,
   error-path fallback, and the contract that ``risk_subgraph`` lands
   on the augmented final.
"""
import asyncio
import json
from typing import AsyncIterator, List

import pytest

from api.runners.risk_predict_runner import (
    _build_risk_subgraph,
    _maybe_augment_final,
    _split_event_frame,
    stream_risk_predict,
)
from api.schemas.insurance import CustomerProfile
from api.services.graph_service import GraphServiceUnavailable


def _profile() -> CustomerProfile:
    return CustomerProfile(
        age=35,
        gender="M",
        occupation="软件工程师",
        health_history=["高血压"],
    )


# ============================================================ build risk subgraph


class TestBuildRiskSubgraph:
    def test_no_seeds_returns_only_customer_fields(self):
        rs = _build_risk_subgraph(_profile(), {"mode": "no_seeds"})
        assert rs["mode"] == "no_seeds"
        assert rs["risk_factors"] == []
        assert rs["triggered_clauses"] == []
        assert rs["edges"] == []
        assert len(rs["customer_fields"]) >= 4

    def test_ppr_drops_noise_entities_with_no_passage_edge(self):
        """Actived entities lacking real PPR adjacency to any in-scope
        passage must NOT appear as risk_factors (codex P1-3)."""
        ppr = {
            "mode": "ppr",
            "seeds": [],
            "actived_entities": [
                {"id": "e-x", "surface": "高血压", "score": 0.6, "iteration_tier": 1},
                {"id": "e-noise", "surface": "无关实体", "score": 0.3, "iteration_tier": 3},
            ],
            "passages": [
                {"hash_id": "p-1", "file_id": "f1", "page_id": "p_001", "score": 0.8},
            ],
            "edges": [
                {"source": "e-x", "target": "p-1", "weight": 1.0},
                # e-noise has no edge to any passage
            ],
        }
        rs = _build_risk_subgraph(_profile(), ppr)
        ids = {rf["id"] for rf in rs["risk_factors"]}
        assert ids == {"rf_e-x"}

    def test_triggered_clauses_carry_no_sup(self):
        """Codex P1-2: ``sup`` belongs to agent's read-order namespace
        (citations[].sup); reusing it for PPR rank would cross-link
        Sankey clicks to wrong passages."""
        ppr = {
            "mode": "ppr",
            "actived_entities": [
                {"id": "e-x", "surface": "高血压", "score": 0.6, "iteration_tier": 1},
            ],
            "passages": [
                {"hash_id": "p-1", "file_id": "f1", "page_id": "p_001", "score": 0.8},
            ],
            "edges": [{"source": "e-x", "target": "p-1", "weight": 1.0}],
        }
        rs = _build_risk_subgraph(_profile(), ppr)
        assert rs["triggered_clauses"]
        for clause in rs["triggered_clauses"]:
            assert "sup" not in clause
            assert {"id", "file_id", "page_id"} <= set(clause.keys())

    def test_l23_edges_only_from_real_ppr_adjacency(self):
        """L2→L3 must come from ppr_result.edges, not a fabricated
        complete bipartite. Two entities, two passages, but only one
        real edge → exactly one L2→L3 edge in the output."""
        ppr = {
            "mode": "ppr",
            "actived_entities": [
                {"id": "e-x", "surface": "高血压", "score": 0.6, "iteration_tier": 1},
                {"id": "e-y", "surface": "心血管", "score": 0.5, "iteration_tier": 1},
            ],
            "passages": [
                {"hash_id": "p-1", "file_id": "f1", "page_id": "p_001", "score": 0.8},
                {"hash_id": "p-2", "file_id": "f1", "page_id": "p_002", "score": 0.5},
            ],
            "edges": [
                {"source": "e-x", "target": "p-1", "weight": 1.0},
                # e-y has no edge → must be filtered as noise
            ],
        }
        rs = _build_risk_subgraph(_profile(), ppr)
        rf_ids = {rf["id"] for rf in rs["risk_factors"]}
        assert rf_ids == {"rf_e-x"}
        l23 = [e for e in rs["edges"] if e["source"].startswith("rf_")]
        assert len(l23) == 1
        assert l23[0]["source"] == "rf_e-x"

    def test_l12_edges_are_uniform_priors(self):
        ppr = {
            "mode": "ppr",
            "actived_entities": [
                {"id": "e-x", "surface": "高血压", "score": 0.6, "iteration_tier": 1},
            ],
            "passages": [
                {"hash_id": "p-1", "file_id": "f1", "page_id": "p_001", "score": 0.8},
            ],
            "edges": [{"source": "e-x", "target": "p-1", "weight": 1.0}],
        }
        rs = _build_risk_subgraph(_profile(), ppr)
        l12 = [e for e in rs["edges"] if not e["source"].startswith("rf_")]
        weights = {round(e["weight"], 4) for e in l12}
        # All L1→L2 edges have the same weight (1 / #fields)
        assert len(weights) == 1


# ============================================================ frame interceptor


class TestFrameInterceptor:
    RS = {"customer_fields": [], "risk_factors": [], "triggered_clauses": [], "edges": [], "mode": "no_seeds"}

    def test_heartbeat_passthrough(self):
        frame = b": keepalive\n\n"
        out, was_final = _maybe_augment_final(frame, self.RS)
        assert out == frame and not was_final

    def test_token_frame_passthrough(self):
        frame = b'event: token\ndata: {"delta":"a"}\n\n'
        out, was_final = _maybe_augment_final(frame, self.RS)
        assert out == frame and not was_final

    def test_final_is_augmented(self):
        frame = b'event: final\ndata: {"answer":"x","loops":3}\n\n'
        out, was_final = _maybe_augment_final(frame, self.RS)
        assert was_final
        # Re-parse the re-emitted frame and assert the augmented fields.
        parts = out.decode().split("\n", 2)
        data_line = parts[1]
        assert data_line.startswith("data: ")
        data = json.loads(data_line[len("data: "):])
        assert data["flavor"] == "risk_predict"
        assert data["risk_subgraph"] == self.RS
        assert data["answer"] == "x"  # original fields preserved

    def test_malformed_final_data_passes_through(self):
        """A malformed JSON ``final`` payload must not break the stream;
        we pass the original bytes through and the frontend can decide
        what to do (typically: render as agent error)."""
        frame = b'event: final\ndata: not-json\n\n'
        out, was_final = _maybe_augment_final(frame, self.RS)
        assert out == frame and not was_final

    def test_split_event_frame_returns_none_for_unknown(self):
        # Custom event without "data:" line
        assert _split_event_frame(b"event: weird\n\n") is None
        # No event line at all
        assert _split_event_frame(b'data: {"a":1}\n\n') is None


# ============================================================ end-to-end stream


class _FakeGraphService:
    """Stand-in for ``GraphService`` exposing only ``ppr_subgraph``.

    Lets us drive the wrapper with a deterministic PPR result without
    booting the real igraph/faiss stack. The shape mirrors what
    ``GraphService.ppr_subgraph`` returns.
    """

    def __init__(self, *, mode: str = "no_seeds", raise_unavailable: bool = False) -> None:
        self.mode = mode
        self.raise_unavailable = raise_unavailable

    def ppr_subgraph(self, query: str, *, file_ids=None):  # noqa: D401
        if self.raise_unavailable:
            raise GraphServiceUnavailable("graph not built yet")
        return {
            "mode": self.mode,
            "seeds": [],
            "actived_entities": [],
            "passages": [],
            "edges": [],
        }


async def _drain(gen: AsyncIterator[bytes]) -> List[bytes]:
    out: List[bytes] = []
    async for chunk in gen:
        out.append(chunk)
    return out


def _join_frames(chunks: List[bytes]) -> List[bytes]:
    """Re-split the wire bytes into individual SSE frames so the test
    can assert against complete records regardless of how many TCP
    chunks the wrapper coalesced into."""
    blob = b"".join(chunks)
    return [f + b"\n\n" for f in blob.split(b"\n\n") if f]


@pytest.mark.asyncio
async def test_stream_emits_error_done_when_graph_service_unavailable(monkeypatch):
    """PPR pre-pass failure must yield error → done as a complete SSE
    sequence; the stream MUST NOT then call into the agent."""

    async def _agent_stream(**kwargs):  # pragma: no cover - must not run
        yield b"event: token\ndata: {\"delta\":\"x\"}\n\n"

    monkeypatch.setattr(
        "api.runners.risk_predict_runner.stream_agent", _agent_stream
    )
    chunks = await _drain(
        stream_risk_predict(
            file_id="f",
            customer=_profile(),
            scenario=None,
            agent=None,  # never used because PPR pre-pass fails first
            graph_service=_FakeGraphService(raise_unavailable=True),
            config=_FakeConfig(),
        )
    )
    frames = _join_frames(chunks)
    names = [_frame_event_name(f) for f in frames]
    assert names == ["error", "done"]


@pytest.mark.asyncio
async def test_stream_passes_through_and_augments_final(monkeypatch):
    """End-to-end happy path: token frames pass through verbatim, the
    final frame gets augmented with risk_subgraph + flavor, and the
    chunking choice of the inner stream does not change the visible
    frame boundaries."""

    # Split the canonical SSE sequence across awkward byte boundaries —
    # mid-frame, mid-data — to exercise the wrapper's buffering.
    payload = (
        b'event: token\ndata: {"delta":"a"}\n\n'
        b'event: token\ndata: {"delta":"b"}\n\n'
        b': keepalive\n\n'
        b'event: final\ndata: {"answer":"hi","loops":2}\n\n'
    )

    async def _agent_stream(**kwargs):
        # Emit one byte at a time. Worst-case fragmentation; if the
        # wrapper handles this, it handles every realistic chunking.
        for i in range(len(payload)):
            yield payload[i : i + 1]

    monkeypatch.setattr(
        "api.runners.risk_predict_runner.stream_agent", _agent_stream
    )
    chunks = await _drain(
        stream_risk_predict(
            file_id="f",
            customer=_profile(),
            scenario=None,
            agent=None,
            graph_service=_FakeGraphService(mode="no_seeds"),
            config=_FakeConfig(),
        )
    )
    frames = _join_frames(chunks)
    names = [_frame_event_name(f) for f in frames]
    assert names == ["token", "token", None, "final"]  # None = heartbeat

    final_frame = frames[-1]
    data_line = final_frame.split(b"\n", 2)[1]
    data = json.loads(data_line[len(b"data: "):])
    assert data["flavor"] == "risk_predict"
    assert data["answer"] == "hi"
    assert data["risk_subgraph"]["mode"] == "no_seeds"
    # Heartbeat retains its `: ` prefix, not converted into an event
    assert frames[2].startswith(b":")


# ============================================================ helpers


class _FakeConfig:
    """Tiny stand-in for ConfigStore.

    Exposes only ``get`` because the runner asks for ``prompt.risk_predict``
    and never needs the per-kind agent kwargs (which would land in
    stream_agent, mocked above)."""

    def get(self, key: str) -> str:
        if key == "prompt.risk_predict":
            return "<unused — stream_agent is mocked>"
        raise KeyError(key)


def _frame_event_name(frame: bytes) -> str | None:
    if frame.startswith(b":"):
        return None
    parsed = _split_event_frame(frame)
    if parsed is None:
        return None
    return parsed[0]
