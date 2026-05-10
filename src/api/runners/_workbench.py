"""Shared scaffolding for the insurance workbench runners.

The five workbenches differ only in:
  * which prompt key they pull from admin config,
  * which agent singleton they call (base / proof),
  * how they format the user prompt from structured input,
  * what extra fields they fold into the ``final`` SSE payload.

This module exposes :func:`stream_workbench_agent` which handles the
shared bus / threadpool / EventBus / future-resolution machinery so
each runner is just "build prompt → call helper".

Each workbench prompt forces the LLM to anchor every claim to a
``[^k]`` superscript. The runner mirrors RAG: it observes ``read``
tool envelopes as they stream past, mints a :class:`CitationItem`
for every distinct ``(file_id, page_id)`` pair in first-seen order,
and pushes one ``citations`` SSE event before the runner's ``final``
frame so the frontend's CitationDrawer can resolve every sup the
answer cites. ``proof_scan`` envelopes are skipped — they carry
unit_ids only, and the agent must call ``read`` to ground any
ScanClaim into verbatim text anyway.
"""
import asyncio
import json
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from agentic.agent.base import BaseAgent
from agentic.agent.proof_agent import ProofAgent, ProofRunResult
from api.runners._tracing import CapturingTracer
from api.runners.events import EventBus, EventType
from api.services.citation import CitationItem
from config.config_store import ConfigStore
from config.settings import relativize_trace_dir
from tracer import Tracer


logger = logging.getLogger(__name__)


_EVIDENCE_TOOL_NAMES: frozenset[str] = frozenset({"read", "proof_scan"})


def _is_evidence(name: Optional[str]) -> bool:
    return name in _EVIDENCE_TOOL_NAMES


async def stream_workbench_agent(
    *,
    user_prompt: str,
    agent: Any,
    kind: str,                 # "base" | "proof"
    config: ConfigStore,
    prompt_key: str,           # admin config key for the SYSTEM prompt
    flavor: str,               # tracer subdir: "compare", "exclusion", ...
    final_extras: Optional[Dict[str, Any]] = None,
    tracer: Optional[Tracer] = None,
    result_future: Optional["asyncio.Future[Dict[str, Any]]"] = None,
    is_evidence: Callable[[Optional[str]], bool] = _is_evidence,
) -> AsyncIterator[bytes]:
    """Run a workbench agent with a custom system prompt + user prompt.

    Why not reuse :func:`api.runners.agent_runner.stream_agent`? The
    workbench needs:
      * a *workbench-specific* system prompt pulled from a separate
        admin key (not ``prompt.base_agent`` / ``prompt.proof_agent``);
      * a structured user prompt assembled from the request body, not
        the raw ``query`` field;
      * extra final-payload fields the chat surface doesn't carry.
    The shared parts (EventBus, threadpool, future resolution,
    tool_result evidence enrichment, trace_path) are factored here so
    every workbench gets them for free.
    """
    if kind not in ("base", "proof"):
        raise ValueError(f"unsupported workbench kind: {kind!r}")
    if not isinstance(agent, (BaseAgent, ProofAgent)):
        raise TypeError("agent must be BaseAgent or ProofAgent")

    loop = asyncio.get_running_loop()
    bus = EventBus(loop=loop)
    capturing = CapturingTracer(tracer) if tracer is not None else None

    # Snapshot config once — concurrent admin PATCH cannot half-apply mid-run.
    system_prompt = str(config.get(prompt_key))
    # Workbenches inherit the per-kind ``max_loops`` / ``max_token_budget``
    # from the corresponding agent kind so admin tuning of the base /
    # proof agent flows through. The system prompt comes from the
    # workbench-specific key above.
    agent_overrides = config.materialize_agent_kwargs(kind)
    agent_overrides["system_prompt"] = system_prompt
    preview_chars = config.citation_preview_chars()

    # Citation accumulator. Populated as ``read`` tool results flow
    # past; one CitationItem per distinct ``(file_id, page_id)`` pair
    # in first-seen order. ``proof_scan`` is ignored — its envelope
    # carries unit_ids only, which would require an inventory lookup
    # to resolve to (file_id, page_number); the ``read`` calls the
    # agent makes to ground a ScanClaim cover the same evidence.
    cited_units: List[CitationItem] = []
    seen_keys: Set[Tuple[str, str]] = set()

    def wrapped_on_event(event_name: str, data: Dict[str, Any]) -> None:
        if event_name == EventType.TOOL_RESULT:
            full_result = data.get("_full_result")
            tool_name = data.get("name")
            data = {k: v for k, v in data.items() if k != "_full_result"}
            data["is_evidence"] = is_evidence(tool_name)
            if data["is_evidence"] and tool_name == "read" and isinstance(full_result, str):
                _accumulate_read_citations(
                    full_result, cited_units, seen_keys, preview_chars
                )
        elif event_name == EventType.FINAL:
            # Swallow the agent-internal ``final`` frame. The runner
            # emits its own canonical ``final`` after ``citations``;
            # forwarding the agent's would put it BEFORE citations
            # and break the ``citations → final → done`` ordering the
            # frontend latches onto.
            return
        bus.push(event_name, data)

    def run_in_thread() -> None:
        result_payload: Dict[str, Any] = {}
        try:
            if kind == "proof":
                proof_result: ProofRunResult = agent.run(
                    user_prompt,
                    tracer=capturing,
                    on_event=wrapped_on_event,
                    cancel_check=lambda: bus.is_closed,
                    **agent_overrides,
                )
                result_payload = {
                    "answer": proof_result.answer,
                    "decision": proof_result.decision,
                    "exit_reason": proof_result.exit_reason,
                    "loops": proof_result.loops,
                    "total_cost": proof_result.total_cost,
                }
            else:
                base_result: Dict[str, Any] = agent.run(
                    user_prompt,
                    tracer=capturing,
                    on_event=wrapped_on_event,
                    cancel_check=lambda: bus.is_closed,
                    **agent_overrides,
                )
                result_payload = {
                    "answer": base_result.get("answer", ""),
                    "exit_reason": base_result.get("exit_reason"),
                    "loops": base_result.get("loops"),
                    "total_cost": base_result.get("total_cost"),
                    "input_tokens_total": base_result.get("input_tokens_total"),
                    "cached_tokens_total": base_result.get("cached_tokens_total"),
                    "output_tokens_total": base_result.get("output_tokens_total"),
                }

            if final_extras:
                result_payload.update(final_extras)

            if capturing is not None and capturing.last_run_dir is not None:
                try:
                    result_payload["trace_path"] = relativize_trace_dir(
                        capturing.last_run_dir
                    )
                except ValueError:
                    logger.warning(
                        "workbench trace dir %s is outside STORAGE_PATH; trace_path omitted",
                        capturing.last_run_dir,
                    )

            # Citations precede final per the SSE contract the frontend
            # relies on (``citations → final → done``). We always push
            # the frame, even when empty, so the client can clear any
            # stale citation state from a previous run on the same
            # session without special-casing absence.
            citation_items = [c.to_dict() for c in cited_units]
            bus.push(EventType.CITATIONS, {"items": citation_items})
            result_payload["citations"] = citation_items

            # Mirror the agent_runner: emit a final SSE event that the
            # frontend can latch onto without reading result_future.
            # ``answer`` is included in the final SSE payload so the
            # frontend can render it without round-tripping through the
            # result_future (the workbench pages don't have a session
            # message persistence layer like ChatPage; their answer
            # lives only in the SSE stream).
            answer_text = str(result_payload.get("answer", ""))
            bus.push(
                EventType.FINAL,
                {
                    "answer": answer_text,
                    "answer_chars": len(answer_text),
                    "exit_reason": result_payload.get("exit_reason"),
                    "loops": result_payload.get("loops"),
                    "total_cost": result_payload.get("total_cost"),
                    "decision": result_payload.get("decision"),
                    "flavor": flavor,
                    "citations_count": len(citation_items),
                    **(final_extras or {}),
                },
            )

        except Exception as exc:
            logger.exception("workbench %s runner failed", flavor)
            # Flush whatever citations have accumulated so far before
            # the error frame. The frontend's CitationDrawer otherwise
            # holds stale state from a previous run on the same
            # session — better an empty list than no event.
            try:
                bus.push(
                    EventType.CITATIONS,
                    {"items": [c.to_dict() for c in cited_units]},
                )
            except Exception:
                logger.exception("workbench %s: failed to push citations during error path", flavor)
            _schedule_future_exception(loop, result_future, exc)
            bus.close(error=f"{type(exc).__name__}: {exc}", error_type=type(exc).__name__)
            return

        _schedule_future_result(loop, result_future, result_payload)
        bus.close()

    loop.run_in_executor(None, run_in_thread)
    async for chunk in bus.stream():
        yield chunk


# ---------- citation extraction (read tool envelope → CitationItem) ----------


def _accumulate_read_citations(
    full_result: str,
    cited_units: List[CitationItem],
    seen_keys: Set[Tuple[str, str]],
    preview_chars: int,
) -> None:
    """Parse a ``read`` tool envelope and append unseen units to ``cited_units``.

    The read tool returns three observation flavours
    (page / passage / table_row); all three put a list of dicts under
    ``units`` whose entries carry ``file_id`` / ``page_id`` /
    ``page_number`` / ``text``. Errored units (status="not_found")
    lack those keys and are skipped silently — the agent will retry.

    Sup numbering is global across the run and increments in
    first-seen order, mirroring how RAG numbers reranked pages: the
    LLM sees the same legend convention either way.

    Failures to parse the envelope are logged and swallowed. A bogus
    tool result must not abort the agent run; the worst case is one
    fewer citation in the drawer.
    """
    try:
        payload = json.loads(full_result)
    except (json.JSONDecodeError, TypeError):
        return
    if not isinstance(payload, dict) or not payload.get("ok"):
        return
    units = payload.get("units")
    if not isinstance(units, list):
        return
    for unit in units:
        if not isinstance(unit, dict):
            continue
        file_id = unit.get("file_id")
        page_id = unit.get("page_id")
        if not file_id or not page_id:
            continue
        key = (str(file_id), str(page_id))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        text = unit.get("text") or ""
        snippet = text.strip().replace("\n", " ")
        cited_units.append(
            CitationItem(
                sup=len(cited_units) + 1,
                file_id=str(file_id),
                page_id=str(page_id),
                page_number=unit.get("page_number"),
                page_preview=snippet[:preview_chars] if snippet else None,
                observation_id=payload.get("observation_id"),
            )
        )


# ---------- shared future-set helpers (mirror agent_runner) ----------


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
        logger.debug("event loop closed; dropping workbench future result")


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
        logger.debug("event loop closed; dropping workbench future exception")
