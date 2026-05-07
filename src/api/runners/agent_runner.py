"""Agent runner — base / proof / graph behind one streaming entry point.

Same EventBus pattern as :mod:`api.runners.rag_runner`. Adds:

* per-tool ``is_evidence`` tag on ``tool_result`` events (so the
  frontend can render read_page / proof_scan / graph_explore-neighbors
  as inline citation cards instead of generic explore steps);
* tracer attachment so the assistant message can persist a relative
  ``trace_path`` for later detail lookup;
* result accumulation surfaced via an ``asyncio.Future`` so the route
  handler can ``await`` the agent's return value once the bus drains
  and write the assistant message inside the request's async session.
"""
import asyncio
import logging
from typing import Any, AsyncIterator, Dict, Optional

from agentic.agent.base import BaseAgent
from agentic.agent.proof_agent import ProofAgent, ProofRunResult
from api.runners._tracing import CapturingTracer
from api.runners.events import EventBus, EventType
from config.config_store import ConfigStore
from config.settings import relativize_trace_dir
from tracer import Tracer


logger = logging.getLogger(__name__)


# ----------------------------------------------------------- evidence ----

# Tools whose ``tool_result`` payload should carry ``is_evidence=True``.
# Updated after reviewing the actual tool catalog: only the two tools
# that return verbatim page / atom content count. Everything else
# (semantic / bm25 / pattern / graph_explore in any mode) is discovery
# — the model still needs to call ``read`` to commit to a page as
# evidence. ``read`` is the single name registered by ReadTool
# (`agentic/tools/acquisition/read.py:74`); ``proof_scan`` is the
# proof-side atom reader.
_EVIDENCE_TOOL_NAMES: frozenset[str] = frozenset({
    "read",
    "proof_scan",
})


def _is_evidence(name: Optional[str], args: Optional[Dict[str, Any]]) -> bool:
    """Decide whether this tool call counts as an inline citation.

    Flat name check: every other tool (search / explore / list) only
    returns candidates the model still has to read separately.
    """
    return name in _EVIDENCE_TOOL_NAMES


# --------------------------------------------------------- main entry ----


def _make_proof_final_payload(result: ProofRunResult) -> Dict[str, Any]:
    return {
        "answer": result.answer,
        "decision": result.decision,
        "exit_reason": result.exit_reason,
        "loops": result.loops,
        "total_cost": result.total_cost,
    }


def _make_base_final_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "answer": result.get("answer", ""),
        "exit_reason": result.get("exit_reason"),
        "loops": result.get("loops"),
        "total_cost": result.get("total_cost"),
        "input_tokens_total": result.get("input_tokens_total"),
        "cached_tokens_total": result.get("cached_tokens_total"),
        "output_tokens_total": result.get("output_tokens_total"),
    }


async def stream_agent(
    *,
    query: str,
    kind: str,                                    # "base" | "proof" | "graph"
    agent: Any,                                   # BaseAgent or ProofAgent singleton
    config: Optional[ConfigStore] = None,
    tracer: Optional[Tracer] = None,
    result_future: Optional["asyncio.Future[Dict[str, Any]]"] = None,
) -> AsyncIterator[bytes]:
    """Stream one agent run as SSE bytes.

    ``agent`` is the lifespan-built singleton matching ``kind``.
    ``tracer`` (when provided) makes the run write to
    ``STORAGE_PATH/<flavor>/<date>/<run_id>``; the runner resolves that
    folder relative to ``STORAGE_PATH`` and stuffs it into the result
    payload so the route can persist it.

    ``result_future`` (when provided) is set with a dict carrying
    ``answer`` / ``exit_reason`` / ``loops`` / ``decision``? /
    ``total_cost`` / ``trace_path``? once the agent returns. The route
    handler awaits it after the bus drains to write the assistant
    message inside the request's async DB session.
    """
    if kind not in ("base", "proof", "graph"):
        raise ValueError(f"unsupported agent kind: {kind!r}")

    loop = asyncio.get_running_loop()
    bus = EventBus(loop=loop)
    capturing = CapturingTracer(tracer) if tracer is not None else None

    # Snapshot the config before the worker starts so a concurrent
    # PATCH cannot half-apply mid-run.
    effective_config = config if config is not None else ConfigStore.defaults_only()
    agent_overrides = effective_config.materialize_agent_kwargs(kind)

    def wrapped_on_event(event_name: str, data: Dict[str, Any]) -> None:
        # Enrich tool_result frames with is_evidence so the frontend
        # can split inline citation cards from generic explore steps.
        if event_name == EventType.TOOL_RESULT:
            data = {**data, "is_evidence": _is_evidence(data.get("name"), None)}
        bus.push(event_name, data)

    def run_in_thread() -> None:
        result_payload: Dict[str, Any] = {}
        try:
            if kind == "proof":
                if not isinstance(agent, ProofAgent):
                    raise TypeError("proof kind requires ProofAgent instance")
                proof_result = agent.run(
                    query,
                    tracer=capturing,
                    on_event=wrapped_on_event,
                    **agent_overrides,
                )
                result_payload = _make_proof_final_payload(proof_result)
            else:
                if not isinstance(agent, BaseAgent):
                    raise TypeError(f"{kind!r} kind requires BaseAgent instance")
                base_result = agent.run(
                    query,
                    tracer=capturing,
                    on_event=wrapped_on_event,
                    **agent_overrides,
                )
                result_payload = _make_base_final_payload(base_result)

            if capturing is not None and capturing.last_run_dir is not None:
                try:
                    result_payload["trace_path"] = relativize_trace_dir(
                        capturing.last_run_dir
                    )
                except ValueError:
                    # tracer.root pointed outside STORAGE_PATH (test
                    # override). Skip rather than store an absolute
                    # path that won't survive a STORAGE_PATH change.
                    logger.warning(
                        "trace dir %s is outside STORAGE_PATH; trace_path omitted",
                        capturing.last_run_dir,
                    )
        except Exception as exc:
            logger.exception("agent runner failed (kind=%s)", kind)
            _schedule_future_exception(loop, result_future, exc)
            bus.close(error=f"{type(exc).__name__}: {exc}", error_type=type(exc).__name__)
            return

        _schedule_future_result(loop, result_future, result_payload)
        bus.close()

    loop.run_in_executor(None, run_in_thread)
    async for chunk in bus.stream():
        yield chunk


# --------------------------------------------------- future scheduling ----

# Helpers to set a future from a worker thread without racing the
# loop-side timeout / cancellation. We schedule the ``done()`` check
# AND the set onto the loop in one callback so they run atomically
# from the future's POV — checking ``done()`` from the worker thread
# is meaningless because the answer can change before our scheduled
# callback fires.

def _schedule_future_result(
    loop: asyncio.AbstractEventLoop,
    future: Optional["asyncio.Future"],
    value: Any,
) -> None:
    if future is None:
        return
    def _set() -> None:
        if not future.done():
            future.set_result(value)
    try:
        loop.call_soon_threadsafe(_set)
    except RuntimeError:
        logger.debug("event loop closed; dropping future result")


def _schedule_future_exception(
    loop: asyncio.AbstractEventLoop,
    future: Optional["asyncio.Future"],
    exc: BaseException,
) -> None:
    if future is None:
        return
    def _set() -> None:
        if not future.done():
            future.set_exception(exc)
    try:
        loop.call_soon_threadsafe(_set)
    except RuntimeError:
        logger.debug("event loop closed; dropping future exception")
