"""RAG runner — owns the per-request EventBus and pipeline invocation.

Glue between :class:`rag.pipeline.RAGPipeline` (sync, blocking, runs in
a worker thread) and the FastAPI :class:`StreamingResponse` (async,
yielding bytes). The route just calls :func:`stream_rag` and pipes its
output straight to the client.
"""
import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

from api.runners._tracing import CapturingTracer
from api.runners.events import EventBus, EventType
from api.services.citation import CitationBuilder
from config.config_store import ConfigStore
from config.settings import relativize_trace_dir
from rag.pipeline import RAGPipeline
from tracer import Tracer


logger = logging.getLogger(__name__)


async def stream_rag(
    *,
    query: str,
    file_ids: Optional[List[str]],
    pipeline: RAGPipeline,
    config: Optional[ConfigStore] = None,
    tracer: Optional[Tracer] = None,
    result_future: Optional["asyncio.Future[Dict[str, Any]]"] = None,
    history: Optional[List[Tuple[str, str]]] = None,
) -> AsyncIterator[bytes]:
    """Yield SSE-encoded bytes for one RAG query.

    Invocation contract: the route handler awaits this generator and
    yields its output directly into ``StreamingResponse``. The pipeline
    runs in the default executor; events flow back via ``EventBus``.

    The citation builder is constructed lazily after rerank — that's
    the only point where we know the actual top-N pages — and parsed
    against the assembled answer once the stream finishes, so the
    final ``citations`` event names exactly the pages the model could
    have cited.

    ``tracer`` (when provided) makes the run write to
    ``STORAGE_PATH/rag/<date>/<run_id>``. ``result_future`` (when
    provided) is set with a dict carrying ``answer`` / ``citations`` /
    ``timings_ms`` / ``channels_hit_counts`` / ``reranked_count`` /
    ``trace_path``? once the pipeline returns; the session route
    awaits it after the bus drains and writes the assistant message.
    """
    loop = asyncio.get_running_loop()
    bus = EventBus(loop=loop)
    capturing = CapturingTracer(tracer) if tracer is not None else None

    # Materialize the per-request config snapshot once. Holding it for
    # the duration of the call means an admin PATCH that lands mid-run
    # does not mutate the values this run is using.
    effective_config = config if config is not None else ConfigStore.defaults_only()
    # Use the pipeline's own RAGConfig as the base so any constructor-
    # time tuning of non-admin fields (e.g. per-channel topks) survives
    # the override; only the four admin-managed knobs get swapped in.
    rag_config = effective_config.materialize_rag_config(base=pipeline.config)
    system_prompt = effective_config.get("prompt.rag_business")
    preview_chars = effective_config.citation_preview_chars()

    # Captured by the closure passed to RAGPipeline. Set on rerank;
    # read after the pipeline returns so we can parse [^k] markers.
    # The builder is shared between the legend and pages-block providers
    # so both renderings use the SAME ``sup`` numbering.
    builder_holder: dict = {"builder": None, "answer": ""}

    def _ensure_builder(reranked: Sequence) -> CitationBuilder:
        builder = builder_holder["builder"]
        if builder is None:
            builder = CitationBuilder.from_reranked_pages(
                reranked, preview_chars=preview_chars
            )
            builder_holder["builder"] = builder
        return builder

    def legend_provider(reranked: Sequence) -> str:
        return _ensure_builder(reranked).render_legend_for_prompt()

    def pages_block_provider(reranked: Sequence) -> str:
        return _ensure_builder(reranked).render_pages_block(reranked)

    def _set_future(payload: Dict[str, Any]) -> None:
        # Atomic done()-check + set on the loop thread — see notes in
        # agent_runner._schedule_future_result.
        if result_future is None:
            return
        def _set() -> None:
            if not result_future.done():
                result_future.set_result(payload)
        try:
            loop.call_soon_threadsafe(_set)
        except RuntimeError:
            logger.debug("event loop closed; dropping rag future result")

    def _set_future_exception(exc: BaseException) -> None:
        if result_future is None:
            return
        def _set() -> None:
            if not result_future.done():
                result_future.set_exception(exc)
        try:
            loop.call_soon_threadsafe(_set)
        except RuntimeError:
            logger.debug("event loop closed; dropping rag future exception")

    def run_in_thread() -> None:
        """Pipeline call body — runs off the event loop."""
        try:
            result = pipeline.run(
                query=query,
                file_ids=file_ids,
                tracer=capturing,
                on_event=bus.push,
                stream=True,
                system_prompt=system_prompt,
                config_override=rag_config,
                citation_legend_provider=legend_provider,
                pages_block_provider=pages_block_provider,
                cancel_check=lambda: bus.is_closed,
                history=history,
            )
            builder_holder["answer"] = result.answer
            # Parse the assembled answer against the legend and emit
            # the citations summary BEFORE close() (close puts the
            # terminal done frame; citations / final must precede it).
            builder = builder_holder["builder"]
            cited = []
            if builder is not None:
                _, cited = builder.parse_response(result.answer)

            citations_payload = [c.to_dict() for c in cited]
            channels_hit_counts = {
                n: len(h) for n, h in result.channels.items()
            }
            timings_ms = {k: int(v * 1000) for k, v in result.timings.items()}

            bus.push(EventType.CITATIONS, {"items": citations_payload})
            bus.push(
                EventType.FINAL,
                {
                    "answer_chars": len(result.answer),
                    "reranked_count": len(result.pages),
                    "channels_hit_counts": channels_hit_counts,
                    "timings_ms": timings_ms,
                },
            )

            payload: Dict[str, Any] = {
                "answer": result.answer,
                "exit_reason": "ok",
                "citations": citations_payload,
                "channels_hit_counts": channels_hit_counts,
                "timings_ms": timings_ms,
                "reranked_count": len(result.pages),
            }
            if capturing is not None and capturing.last_run_dir is not None:
                try:
                    payload["trace_path"] = relativize_trace_dir(capturing.last_run_dir)
                except ValueError:
                    logger.warning(
                        "rag trace dir %s is outside STORAGE_PATH; trace_path omitted",
                        capturing.last_run_dir,
                    )
            _set_future(payload)
        except Exception as exc:
            logger.exception("RAG runner failed")
            _set_future_exception(exc)
            bus.close(error=f"{type(exc).__name__}: {exc}", error_type=type(exc).__name__)
            return
        bus.close()

    loop.run_in_executor(None, run_in_thread)
    async for chunk in bus.stream():
        yield chunk
