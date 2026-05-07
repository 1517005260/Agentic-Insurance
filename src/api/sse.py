"""SSE wire-format encoder.

Pure encoding — no FastAPI / asyncio. The HTTP layer just yields the
bytes this module returns.

Frame shape::

    event: <name>\\n
    data: <one-line JSON>\\n
    \\n

The data payload is always serialized as a single line (``ensure_ascii``
off so CJK is readable, separators tightened) so newlines inside the
payload can never split the SSE record. Comments (``: keepalive``) are
used for heartbeats — proxies that timeout idle connections (nginx 60s
default) need them.
"""
import json
from typing import Any, Mapping


HEARTBEAT_FRAME: bytes = b": keepalive\n\n"


def format_event(event: str, data: Mapping[str, Any]) -> bytes:
    """Encode one SSE event. Returns UTF-8 bytes ready to write to the wire.

    The event name is restricted to a small ASCII vocabulary defined in
    :mod:`api.runners.events` — we don't escape it because the producer
    side controls it. The data dict is JSON-encoded inline; embedded
    newlines collapse into ``\\n`` literals so the SSE record framing
    (``\\n\\n``) is never accidentally triggered.
    """
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")
