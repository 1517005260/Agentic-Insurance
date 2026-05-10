"""Web-RAG runner — SSE wrapper around :func:`api.services.web_rag.stream_chat`.

Mirrors :mod:`api.runners.rag_runner` but the retrieval stack is
Tavily and there is no inline-page citation building (web cites are
URL-based, the LLM emits ``[^k]`` markers tied to the numbered
sources block we built into the prompt).
"""
import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

from api.runners._tracing import CapturingTracer
from api.runners.events import EventBus, EventType
from api.services import web_rag as web_rag_svc
from config.config_store import ConfigStore
from config.settings import relativize_trace_dir
from model_client import LLMClient
from model_client.web_search import TavilyClient
from tracer import Tracer


logger = logging.getLogger(__name__)


_KNOWN_EVENTS = {"status", "rewrite", "retrieval", "token", "citations", "final"}


async def stream_web_rag(
    *,
    query: str,
    llm: LLMClient,
    tavily: TavilyClient,
    config: Optional[ConfigStore] = None,
    include_domains: Optional[Sequence[str]] = None,
    exclude_domains: Optional[Sequence[str]] = None,
    tracer: Optional[Tracer] = None,
    result_future: Optional["asyncio.Future[Dict[str, Any]]"] = None,
    history: Optional[List[Tuple[str, str]]] = None,
) -> AsyncIterator[bytes]:
    """Stream one web-RAG run as SSE bytes.

    ``include_domains`` / ``exclude_domains`` are exposed for callers
    that want to hard-pin a domain set; the chat path leaves them
    ``None`` so the LLM can search the open web.
    """
    loop = asyncio.get_running_loop()
    bus = EventBus(loop=loop)
    capturing = CapturingTracer(tracer) if tracer is not None else None

    effective_config = config if config is not None else ConfigStore.defaults_only()
    system_prompt = effective_config.get("prompt.web_rag")
    max_results = effective_config.get("tavily.max_results")
    search_depth = effective_config.get("tavily.search_depth")
    answer_max_tokens = effective_config.get("rag.answer_max_tokens")

    def _set_future(payload: Dict[str, Any]) -> None:
        if result_future is None:
            return
        def _set() -> None:
            if not result_future.done():
                result_future.set_result(payload)
        try:
            loop.call_soon_threadsafe(_set)
        except RuntimeError:
            logger.debug("event loop closed; dropping web_rag future result")

    def _set_future_exception(exc: BaseException) -> None:
        if result_future is None:
            return
        def _set() -> None:
            if not result_future.done():
                result_future.set_exception(exc)
        try:
            loop.call_soon_threadsafe(_set)
        except RuntimeError:
            logger.debug("event loop closed; dropping web_rag future exception")

    # The tracer (if any) just gets a single record — there is no
    # multi-stage trajectory worth replaying — so we open a session,
    # log the query, and let the run dir close cleanly when the
    # context exits.
    trace_session = capturing.session(query) if capturing is not None else None

    def run_in_thread() -> None:
        assembled: Dict[str, Any] = {}
        try:
            generator = web_rag_svc.stream_chat(
                llm=llm,
                tavily=tavily,
                query=query,
                max_results=int(max_results),
                search_depth=str(search_depth),
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                system_prompt=str(system_prompt) if system_prompt else None,
                max_tokens=int(answer_max_tokens),
                cancel_check=lambda: bus.is_closed,
                history=history,
            )
            for event_name, data in generator:
                if event_name == "__assembled__":
                    assembled = data
                    continue
                if event_name in _KNOWN_EVENTS:
                    bus.push(event_name, data)
                else:
                    logger.warning("web_rag emitted unknown event: %r", event_name)
        except Exception as exc:
            logger.exception("web_rag runner failed")
            _set_future_exception(exc)
            bus.close(error=f"{type(exc).__name__}: {exc}", error_type=type(exc).__name__)
            return

        if trace_session is not None:
            try:
                # finalize writes ``final.json`` with {query, answer, ...}.
                # The history loader (api/services/history.py) reads
                # ``final.json.answer`` to reconstruct multi-turn pairs;
                # without this call the assistant side of every web-RAG
                # turn would be silently skipped on the next request.
                trace_session.finalize(
                    answer=assembled.get("answer", ""),
                    summary={
                        "n_sources": len(assembled.get("sources", [])),
                        "n_cited": len(assembled.get("cited", [])),
                    },
                )
            except Exception:
                logger.debug("trace_session.finalize failed", exc_info=True)

        payload: Dict[str, Any] = {
            "answer": assembled.get("answer", ""),
            "exit_reason": "ok",
            "citations": assembled.get("cited", []),
            "n_results": len(assembled.get("sources", [])),
            # Forward stage timings + the rewritten search query +
            # original user query + any rewrite error so the
            # session-persist layer (chat.py) can stash them in
            # ``chat_messages.metadata_json`` for the audit / trace UI.
            # All four are best-effort — they exist when stream_chat
            # ran cleanly and are absent on fail-fast paths above.
            "timings_ms": assembled.get("timings_ms"),
            "search_query": assembled.get("search_query"),
            "original_query": assembled.get("original_query"),
            "rewrite_error": assembled.get("rewrite_error"),
        }
        if capturing is not None and capturing.last_run_dir is not None:
            try:
                payload["trace_path"] = relativize_trace_dir(capturing.last_run_dir)
            except ValueError:
                logger.warning(
                    "web_rag trace dir %s is outside STORAGE_PATH; trace_path omitted",
                    capturing.last_run_dir,
                )
        _set_future(payload)
        bus.close()

    loop.run_in_executor(None, run_in_thread)
    async for chunk in bus.stream():
        yield chunk
