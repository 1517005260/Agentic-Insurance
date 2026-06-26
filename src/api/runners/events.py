"""Sync→async event bridge for streaming runners.

The algorithm layer (RAGPipeline, BaseAgent) is synchronous
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

Two operating modes, selected by the ``replay_buffered`` constructor
flag (default False):

* **single-consumer / no replay** (chat runners) — one queue, one
  ``stream()``. ``is_closed`` flips when the consumer disconnects so
  the worker thread can short-circuit LLM token generation. This is
  what the chat path relies on.
* **multi-consumer / replay** (ingest) — push() also appends to a
  history buffer. Each ``stream()`` subscriber gets its own queue,
  is seeded with the full history on subscribe, then receives any
  subsequent live events. ``is_closed`` only flips on
  :meth:`close`, not on consumer disconnect — the bg task should
  keep running even after the user "minimizes" the progress dialog.
  A reconnecting client (e.g. via the FilesPage chip) replays the
  buffered stage timeline before going live, so they see the same
  picture as someone who watched from start.
"""
import asyncio
import logging
import threading
from typing import Any, AsyncIterator, List, Mapping, Optional, Tuple

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
    THOUGHT = "thought"           # intermediate LLM content (reasoning / plan), not the final answer
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # Graph-agent live replay — emitted by agent_runner.py when a graph
    # kind agent's graph tool returns. Carries enough metadata
    # for GraphPage to re-render the canvas around what the agent just
    # discovered (extracted from the tool envelope so the frontend
    # sees the agent's actual hits, not a separate /graph/expand call).
    GRAPH_SUBGRAPH = "graph_subgraph"


# Default 15s — comfortably under nginx 60s default proxy_read_timeout.
DEFAULT_HEARTBEAT_INTERVAL = 15.0


class EventBus:
    """Async-consumer / sync-producer pipe yielding SSE frames.

    Default mode is single-consumer / no-replay (chat runners rely on
    ``is_closed`` flipping on disconnect to cancel LLM streaming).

    With ``replay_buffered=True`` the bus instead keeps a history of
    every event and supports multiple ``stream()`` subscribers; each
    subscriber gets the full history replayed on subscribe and then
    receives any subsequent live events. ``is_closed`` no longer
    flips on consumer disconnect in this mode — only :meth:`close`
    sets it. This is the behavior the ingest path needs so a
    "minimized" upload dialog can be reopened later (or from another
    tab) and see the live stage timeline as if it had watched from
    the start.
    """

    # Sentinel object used to stop the stream() loop. Put on the queue
    # by close() via the same call_soon_threadsafe path as normal items
    # so ordering is preserved (everything pushed before close lands first).
    _CLOSE = object()

    def __init__(
        self,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        *,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        replay_buffered: bool = False,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._heartbeat_interval = heartbeat_interval
        self._replay_buffered = replay_buffered
        self._closed = False

        if replay_buffered:
            # Multi-consumer: each subscriber owns its own queue. The
            # producer fans events into all of them under _state_lock.
            # _history captures every push() so a late subscriber gets
            # the full timeline (including the terminal done/error frames
            # if close() already ran).
            self._subscribers: List[asyncio.Queue] = []
            self._history: List[Tuple[str, Mapping[str, Any]]] = []
            # threading.Lock — push()/close() run on the worker thread;
            # subscribe() runs on the event loop. asyncio.Lock won't do.
            self._state_lock = threading.Lock()
            self._queue = None  # single-queue field unused in this mode
        else:
            self._queue: asyncio.Queue = asyncio.Queue()
            self._subscribers = []
            self._history = []
            self._state_lock = threading.Lock()

    # ------------------------------------------------------------- producer

    @property
    def is_closed(self) -> bool:
        """Producer-closed flag.

        Single-consumer mode: also flips when ``stream()`` exits (client
        disconnect), which is what chat runners use to bail out of an
        in-flight LLM stream.

        Multi-consumer mode: only flips on :meth:`close`. Disconnects
        do not propagate back to the worker — the bg task keeps running
        and a reconnecting subscriber gets full history replay.
        """
        return self._closed

    def push(self, event: str, data: Mapping[str, Any]) -> None:
        """Enqueue an event from any thread.

        Cheap (a single ``call_soon_threadsafe`` per active queue).
        Calls after ``close()`` are silently dropped.
        """
        if self._replay_buffered:
            self._fanout(event, data)
            return
        if self._closed:
            return
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, (event, data))
        except RuntimeError:
            # Loop already closed — consumer is gone. Drop.
            logger.debug("EventBus.push(%s): event loop closed, dropping", event)

    def _fanout(self, event: str, data: Mapping[str, Any]) -> None:
        """Append to history + push into every subscriber queue.

        Held under ``_state_lock`` so a concurrent ``stream()`` snapshot
        can't see a partial history (history-append + queue.put_nowait
        are non-atomic vs. the new-subscriber replay loop). The
        ``_closed`` check is INSIDE the lock so a push that races a
        close cannot land after the terminal frames in history.
        """
        with self._state_lock:
            if self._closed:
                return
            item = (event, data)
            self._history.append(item)
            queues = list(self._subscribers)
        for q in queues:
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, item)
            except RuntimeError:
                logger.debug("EventBus.push(%s): event loop closed", event)

    def close(self, *, error: Optional[str] = None, error_type: Optional[str] = None) -> None:
        """Signal end-of-stream. Idempotent.

        If ``error`` is given, an ``error`` frame is enqueued before the
        terminal ``done`` frame so the client can distinguish a clean
        end from an aborted one. Always followed by ``done`` so the
        client always sees one terminal frame either way.
        """
        if self._replay_buffered:
            self._close_multi(error, error_type)
            return
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

    def _close_multi(self, error: Optional[str], error_type: Optional[str]) -> None:
        """Multi-consumer close.

        Append terminal frames to history AND flip ``_closed`` INSIDE
        the same ``_state_lock`` window. That ordering closes the
        subscriber-vs-close race: a ``_stream_multi`` caller that takes
        the lock either (a) observes ``_closed=False`` and registers
        itself on ``_subscribers`` so the upcoming terminal-frame push
        reaches it, or (b) observes ``_closed=True`` and a history that
        already includes the terminal frames so the replay self-finishes.
        With the flip outside the lock, a snapshot could land in the
        gap and queue a bare ``_CLOSE`` without first seeing the
        terminal frames — clients then see no ``done`` event.
        """
        terminal: List[Tuple[str, Mapping[str, Any]]] = []
        if error is not None:
            terminal.append(
                (EventType.ERROR, {"message": error, "type": error_type or "RuntimeError"})
            )
        terminal.append((EventType.DONE, {}))
        with self._state_lock:
            if self._closed:
                return
            self._history.extend(terminal)
            self._closed = True
            queues = list(self._subscribers)
        for q in queues:
            try:
                for item in terminal:
                    self._loop.call_soon_threadsafe(q.put_nowait, item)
                self._loop.call_soon_threadsafe(q.put_nowait, self._CLOSE)
            except RuntimeError:
                logger.debug("EventBus._close_multi: event loop closed")

    # ------------------------------------------------------------- consumer

    async def stream(self) -> AsyncIterator[bytes]:
        """Yield SSE frames until the bus closes (or the consumer exits).

        Heartbeat: when the queue is idle for ``heartbeat_interval``
        seconds, yields a single ``: keepalive`` comment frame.

        Single-consumer mode: ``finally`` flips ``_closed`` on consumer
        exit so the worker can stop generating.

        Multi-consumer mode: subscribe to a fresh per-call queue,
        seeded with the full history under the state lock so we
        atomically capture "history snapshot at subscribe time" + register
        for live events. If the bus is already closed, we drain history +
        terminal frames and return without affecting other subscribers.
        """
        if self._replay_buffered:
            async for frame in self._stream_multi():
                yield frame
            return
        # ------------------------------------------------ single-consumer
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

    async def _stream_multi(self) -> AsyncIterator[bytes]:
        q: asyncio.Queue = asyncio.Queue()
        with self._state_lock:
            # Atomic snapshot: replay everything pushed so far, then
            # register for new events. Without the lock, an event
            # could land between snapshot and register and be lost
            # (or duplicated if the worker raced ahead).
            for item in self._history:
                q.put_nowait(item)
            if self._closed:
                # History already contains terminal frames; emit a
                # close sentinel so this subscriber finishes after
                # replay rather than waiting for the heartbeat.
                q.put_nowait(self._CLOSE)
            else:
                self._subscribers.append(q)
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        q.get(), timeout=self._heartbeat_interval
                    )
                except asyncio.TimeoutError:
                    yield HEARTBEAT_FRAME
                    continue
                if item is self._CLOSE:
                    return
                event, data = item  # type: ignore[misc]
                yield format_event(event, data)
        finally:
            with self._state_lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)
