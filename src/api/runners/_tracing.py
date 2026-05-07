"""Tiny shared helper used by RAG and agent runners.

The :class:`tracer.Tracer` API hands ``TraceSession`` objects back to
the caller and forgets them; we need the resulting per-session
``run_dir`` to persist as ``trace_path`` on the assistant message.
``CapturingTracer`` is a duck-typed wrapper that captures the dir as
the tracer hands it out, without touching the algorithm-layer Tracer
class.
"""
from pathlib import Path
from typing import Optional

from tracer import Tracer


class CapturingTracer:
    """Wraps a Tracer and remembers the most recent session's ``run_dir``."""

    __slots__ = ("_inner", "base_dir", "last_run_dir")

    def __init__(self, inner: Tracer) -> None:
        self._inner = inner
        self.base_dir: Path = inner.base_dir
        self.last_run_dir: Optional[Path] = None

    def session(self, query: str, run_id: Optional[str] = None):
        sess = self._inner.session(query, run_id)
        self.last_run_dir = sess.run_dir
        return sess
