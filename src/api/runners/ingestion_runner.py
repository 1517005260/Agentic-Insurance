"""Ingestion progress streaming.

Owns the in-process registry of active per-job ``EventBus`` instances and
the small helpers the SSE route uses to subscribe.

Multi-consumer fan-out lives inside ``EventBus(replay_buffered=True)``
(see :mod:`api.runners.events`). This module is just the registry —
register on bg-task entry, look up from the SSE route, unregister on
exit. Two subscribers to the same job are legal: the second one gets a
fresh queue seeded with the full event history, so the FilesPage
"minimize → reopen" UX (and any extra browser tab) sees the same stage
timeline as someone watching from the start.

Lifecycle (lives in :mod:`api.services.files`)::

    bus = EventBus(loop=loop)
    register_bus(job_id, bus)
    try:
        ... parse + ingest ... pass on_event=bus.push down to pipeline ...
        bus.push("final", {...})
    except Exception as exc:
        bus.close(error=str(exc), error_type=type(exc).__name__)
    else:
        bus.close()
    finally:
        unregister_bus(job_id)

The route layer:

    bus = await wait_for_bus(job_id, timeout=2.0)
    if bus is None:
        # Job finished before the SSE caught up — replay the terminal
        # state from the DB row instead of looking dead.
        return synthesize_terminal(job_row)
    return StreamingResponse(bus.stream(), ...)
"""
import asyncio
import logging
from typing import Dict, Optional

from api.runners.events import EventBus


logger = logging.getLogger(__name__)


# Keyed by ``IngestJob.id``. Owned by the bg task: registered on entry,
# unregistered on exit (success OR failure). The route layer treats
# misses as "already finished or never started" and falls back to the
# DB row for a terminal-state replay.
_BUSES: Dict[int, EventBus] = {}

def register_bus(job_id: int, bus: EventBus) -> None:
    """Publish the bus so the SSE route can pick it up."""
    _BUSES[job_id] = bus


def unregister_bus(job_id: int) -> None:
    """Drop the bus. Idempotent.

    Safe to call after ``bus.close()`` — does not affect any active
    ``stream()`` iterator (close() already drained the close sentinel).
    """
    _BUSES.pop(job_id, None)


def get_bus(job_id: int) -> Optional[EventBus]:
    return _BUSES.get(job_id)


async def wait_for_bus(job_id: int, *, timeout: float = 2.0, poll_ms: int = 100) -> Optional[EventBus]:
    """Wait up to ``timeout`` seconds for the bg task to register.

    Starlette schedules ``BackgroundTasks`` after the 202 response
    flushes, so a quick client (XHR fired right after the upload
    POST returns) can reach this code path before ``run_parse_index``
    has had its first tick. A short poll bridges that window without
    requiring the upload route to pre-allocate the bus.
    """
    bus = get_bus(job_id)
    if bus is not None:
        return bus
    deadline = asyncio.get_event_loop().time() + timeout
    interval = poll_ms / 1000.0
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(interval)
        bus = get_bus(job_id)
        if bus is not None:
            return bus
    return None
