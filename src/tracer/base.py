"""Tracer + TraceSession.

The tracer surface is intentionally tiny:

* ``Tracer.session(query)``     — open one session per query
* ``session.snapshot(name, payload)`` — write/overwrite ``<name>.json``
* ``session.event(stream, record)``   — append one line to ``<stream>.jsonl``
* ``session.finalize(answer=, summary=)`` — write ``final.json`` with timing

Both ``name`` and ``stream`` may contain forward slashes to nest into
sub-directories (e.g. ``"channels/semantic"`` writes to
``<run_dir>/channels/semantic.json``). All file extensions are added by
the tracer, never by the caller.

That's it. Adding a new tool, a new pipeline stage, or a new diagnostic
event requires zero tracer changes — the caller just picks a name and
emits. Snapshot vs event is the only design distinction: snapshots are
one-shot artifacts that may be overwritten (e.g. ``preprocess``,
``rerank``); events are streamed records (e.g. ``trajectory``,
``llm_calls``). Pick whichever fits, or both — the tracer doesn't care.

JSON is pretty-printed (indent=2) on snapshots — these are read by
humans, not fed back to the model. JSONL events are compact one-line
records since each line is independently parseable.
"""

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import STORAGE_PATH


logger = logging.getLogger(__name__)


_VALID_FLAVORS = {"rag", "agentic"}


@dataclass
class Tracer:
    """Per-flavor sink. One per process is plenty.

    ``flavor`` selects ``local_storage/<flavor>/`` as the root; the
    session-id namespace is private to that flavor. ``root`` overrides
    ``STORAGE_PATH`` for tests.
    """

    flavor: str
    root: Optional[Path] = None
    _resolved_root: Path = field(init=False)

    def __post_init__(self):
        if self.flavor not in _VALID_FLAVORS:
            raise ValueError(f"flavor must be one of {sorted(_VALID_FLAVORS)}; got {self.flavor!r}")
        base = Path(self.root) if self.root else STORAGE_PATH
        self._resolved_root = base / self.flavor

    @property
    def base_dir(self) -> Path:
        return self._resolved_root

    def session(self, query: str, run_id: Optional[str] = None) -> "TraceSession":
        """Open a new session for one query. Folder is created eagerly."""
        now = datetime.now()
        date_dir = self._resolved_root / now.strftime("%Y-%m-%d")
        # Time prefix keeps `ls -1` chronological even within one day; the
        # uuid suffix avoids collisions when multiple queries fire in the
        # same second from a parallel runner.
        run_id = run_id or f"{now.strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = date_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return TraceSession(
            run_dir=run_dir,
            query=query,
            started_at=now,
            flavor=self.flavor,
        )


class TraceSession:
    """One open session = one query = one folder.

    The two recording primitives (`snapshot` and `event`) are all you
    need. They're deliberately untyped — payloads are arbitrary
    JSON-serializable dicts so producers can extend their schema
    without coordinating with this module.
    """

    _SNAPSHOT_EXT = ".json"
    _EVENT_EXT = ".jsonl"

    def __init__(
        self,
        *,
        run_dir: Path,
        query: str,
        started_at: datetime,
        flavor: str,
    ):
        self.run_dir = run_dir
        self.query = query
        self.flavor = flavor
        self.started_at = started_at
        self._lock = threading.Lock()
        self._finalized = False
        # The query goes down immediately so a crash leaves a useful
        # breadcrumb. Reserved name; callers are free to overwrite by
        # calling snapshot("query", ...) themselves.
        self.snapshot(
            "query",
            {
                "query": query,
                "flavor": flavor,
                "started_at": started_at.isoformat(timespec="seconds"),
                "run_dir": str(run_dir),
            },
        )

    # ------------------------------------------------------------- API

    def snapshot(self, name: str, payload: Dict[str, Any]) -> Path:
        """Write ``<run_dir>/<name>.json`` (overwrite on repeat).

        ``name`` may contain ``/`` to nest into a subdirectory.
        Trailing ``.json`` is stripped if present (caller convenience).
        """
        path = self._resolve_path(name, self._SNAPSHOT_EXT)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_atomic(path, payload)
        return path

    def daily(self, name: str, payload: Dict[str, Any]) -> str:
        """Write ``<day_dir>/<name>.json`` ONCE per day, deduped by content.

        Use for static run-prefix artifacts that are identical across
        every run on a given day — the canonical example is the
        agent's ``(system_prompt, tool_schemas, model)`` bundle, which
        can be 20+ KB and would otherwise be cloned per session for no
        new information.

        Behavior:

        * First call: write ``<day_dir>/<name>.json``.
        * Subsequent call with the same content: no-op.
        * Subsequent call with DIFFERENT content (e.g. someone tweaked
          the system prompt mid-day): write a sibling file
          ``<day_dir>/<name>_<hash8>.json`` so neither version is lost.

        The 12-char content hash is stamped into the session's
        ``query.json`` under ``<name>_hash`` along with the resolved
        path so a postmortem can find the matching artifact even if
        multiple variants exist.
        """
        serialized = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, default=str
        )
        content_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
        day_dir = self.run_dir.parent
        canonical = day_dir / f"{name}.json"
        target = canonical
        with self._lock:
            if canonical.exists():
                try:
                    existing = canonical.read_text(encoding="utf-8")
                    existing_hash = hashlib.sha256(existing.encode("utf-8")).hexdigest()[:12]
                except OSError as exc:
                    logger.warning("TraceSession: cannot read %s: %s", canonical, exc)
                    existing_hash = None
                if existing_hash != content_hash:
                    # Different content under same logical name — keep
                    # both, address by hash. Canonical stays whatever
                    # was written first today.
                    target = day_dir / f"{name}_{content_hash}.json"
            if not target.exists():
                target.write_text(serialized, encoding="utf-8")
        # Stamp into query.json so this session is self-describing.
        self._stamp_query({
            f"{name}_hash": content_hash,
            f"{name}_path": str(target.relative_to(day_dir.parent)),
        })
        return content_hash

    def event(self, stream: str, record: Dict[str, Any]) -> None:
        """Append one record to ``<run_dir>/<stream>.jsonl``.

        Use for streamed event logs (per-turn tool calls, per-LLM-call
        accounting, anything that grows during a run). Records are
        compact-JSON, one per line, so a tail-readable file emerges
        incrementally and a crash mid-run still leaves the prefix
        intact.
        """
        path = self._resolve_path(stream, self._EVENT_EXT)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def finalize(self, *, answer: str, summary: Dict[str, Any]) -> Path:
        """Write ``final.json`` with timing. Idempotent re-call overwrites."""
        ended_at = datetime.now()
        elapsed = (ended_at - self.started_at).total_seconds()
        payload = {
            "query": self.query,
            "answer": answer,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": ended_at.isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 3),
            "summary": summary,
        }
        path = self.snapshot("final", payload)
        self._finalized = True
        return path

    # ------------------------------------------------------------- internals

    def _resolve_path(self, name: str, ext: str) -> Path:
        cleaned = str(name).strip().lstrip("/")
        if not cleaned:
            raise ValueError("name must be non-empty")
        # Strip a redundant extension if the caller added one.
        for trailing in (ext, ".json", ".jsonl"):
            if cleaned.endswith(trailing):
                cleaned = cleaned[: -len(trailing)]
                break
        # Forbid escapes — name should describe a path RELATIVE to run_dir.
        if any(part == ".." for part in cleaned.split("/")):
            raise ValueError(f"name must not contain '..': {name!r}")
        return self.run_dir / (cleaned + ext)

    def _stamp_query(self, fields: Dict[str, Any]) -> None:
        """Merge new fields into ``query.json`` (created by __init__)."""
        path = self.run_dir / "query.json"
        with self._lock:
            try:
                if path.exists():
                    data = json.loads(path.read_text(encoding="utf-8"))
                else:
                    data = {}
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("TraceSession: cannot rehydrate %s: %s", path, exc)
                data = {}
            data.update(fields)
            tmp = path.with_suffix(path.suffix + ".tmp")
            try:
                tmp.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                tmp.replace(path)
            except OSError as exc:
                logger.warning("TraceSession: failed to stamp %s: %s", path, exc)

    def _write_atomic(self, path: Path, payload: Dict[str, Any]) -> None:
        """Temp-file + rename so a crash mid-write doesn't leave a torn JSON."""
        with self._lock:
            tmp = path.with_suffix(path.suffix + ".tmp")
            try:
                tmp.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                tmp.replace(path)
            except OSError as exc:
                logger.warning("TraceSession: failed to write %s: %s", path, exc)
                try:
                    tmp.unlink()
                except OSError:
                    pass
