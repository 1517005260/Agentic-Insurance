"""Sync→async event bridge for streaming runners.

The algorithm layer (RAGPipeline, BaseAgent, ProofAgent) is synchronous
and runs inside ``loop.run_in_executor(...)``. SSE consumers (FastAPI
routes) are async generators yielding bytes. ``EventBus`` is the seam:

  * :meth:`push` — sync, called from the worker thread on every step.
  * :meth:`stream` — async, drives the SSE response. Yields encoded
    frames plus a periodic heartbeat so proxies don't time the
    connection out.
  * :meth:`close` — sync, ends the stream. Pushes a terminal ``done``
    frame (or ``error`` then ``done`` if a reason is given).

Cross-thread plumbing uses ``loop.call_soon_threadsafe`` to enqueue
items into an asyncio Queue owned by the consumer side. The bus binds
to the loop that ``stream()`` was called on, so construct it inside
the route handler (where the running loop is available).

A single bus is single-producer / single-consumer: one runner thread
pushes, one route iterator consumes. Re-use across requests is not
supported.
"""
import asyncio
import logging
from typing import Any, AsyncIterator, Mapping, Optional

from api.sse import HEARTBEAT_FRAME, format_event


logger = logging.getLogger(__name__)


# Event name vocabulary.
class EventType:
    # Generic
    STATUS = "status"             # phase transition (preprocess / retrieve / rerank / answering / tool)
    ERROR = "error"               # terminal-ish; runner may still send done after
    DONE = "done"                 # always the final frame
    FINAL = "final"               # summary right before done (decision, run_id, exit_reason)

    # RAG-specific
    PREPROCESS = "preprocess"     # one of {hyde, rewrite} sub-step start/done
    RETRIEVAL = "retrieval"       # one channel finished
    RERANKED = "reranked"         # rerank top-N ready
    TOKEN = "token"               # answer-stage delta
    CITATIONS = "citations"       # final list[CitationItem]

    # Agent-specific
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OBLIGATION = "obligation"     # proof
    CLAIM = "claim"               # proof
    GAP = "gap"                   # proof


# Default 15s — comfortably under nginx 60s default proxy_read_timeout.
DEFAULT_HEARTBEAT_INTERVAL = 15.0


class EventBus:
    """One-shot async-consumer / sync-producer pipe yielding SSE frames."""

    # Sentinel object used to stop the stream() loop. Put on the queue
    # by close() via the same call_soon_threadsafe path as normal items
    # so ordering is preserved (everything pushed before close lands first).
    _CLOSE = object()

    def __init__(
        self,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        *,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._heartbeat_interval = heartbeat_interval
        self._closed = False

    # ------------------------------------------------------------- producer

    @property
    def is_closed(self) -> bool:
        """True after either ``close()`` ran or ``stream()`` exited.

        Worker callers should poll this at expensive boundaries (e.g.
        per LLM token frame) and short-circuit when it's True — that's
        how a client disconnect actually stops the upstream LLM stream
        instead of merely silencing the queue.
        """
        return self._closed

    def push(self, event: str, data: Mapping[str, Any]) -> None:
        """Enqueue an event from any thread.

        Cheap (a single ``call_soon_threadsafe``); the algorithm-side
        callsite can spam it. Calls after ``close()`` are silently
        dropped — the algorithm thread may still be unwinding when the
        consumer disconnects, and we don't want those late events to
        raise across the thread boundary.
        """
        if self._closed:
            return
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, (event, data))
        except RuntimeError:
            # Loop already closed — consumer is gone. Drop.
            logger.debug("EventBus.push(%s): event loop closed, dropping", event)

    def close(self, *, error: Optional[str] = None, error_type: Optional[str] = None) -> None:
        """Signal end-of-stream. Idempotent.

        If ``error`` is given, an ``error`` frame is enqueued before the
        terminal ``done`` frame so the client can distinguish a clean
        end from an aborted one. Always followed by ``done`` so the
        client always sees one terminal frame either way.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if error is not None:
                self._loop.call_soon_threadsafe(
                    self._queue.put_nowait,
                    (EventType.ERROR, {"message": error, "type": error_type or "RuntimeError"}),
                )
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, (EventType.DONE, {})
            )
            self._loop.call_soon_threadsafe(self._queue.put_nowait, self._CLOSE)
        except RuntimeError:
            logger.debug("EventBus.close: event loop closed before close drained")

    # ------------------------------------------------------------- consumer

    async def stream(self) -> AsyncIterator[bytes]:
        """Yield SSE frames until close() drains the queue.

        Heartbeat: when the queue is idle for ``heartbeat_interval``
        seconds, yields a single ``: keepalive`` comment frame. The
        loop only exits after the close-sentinel is dequeued — so any
        events pushed before close are guaranteed to be flushed first.

        Client-disconnect handling: ``StreamingResponse`` cancels the
        async generator when the TCP connection drops, raising
        ``CancelledError`` (or just stopping iteration). The ``finally``
        flips ``_closed`` so any ``push()`` from the still-running
        worker thread becomes a no-op — we don't want to keep paying
        for LLM tokens nobody will read.
        """
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=self._heartbeat_interval
                    )
                except asyncio.TimeoutError:
                    yield HEARTBEAT_FRAME
                    continue

                if item is self._CLOSE:
                    return

                event, data = item  # type: ignore[misc]
                yield format_event(event, data)
        finally:
            self._closed = True
