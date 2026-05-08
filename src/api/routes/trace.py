"""Trace detail endpoint.

Reads the on-disk per-run trace artefacts that
:class:`api.runners._tracing.CapturingTracer` captured during a chat
session reply, so the frontend can render a "what did the agent
actually do" drawer (full trajectory + final result + per-turn LLM
calls).

Path resolution is delegated to :func:`config.settings.trace_run_path`
which is path-traversal safe — it raises :class:`ValueError` if the
relative path resolves outside ``STORAGE_PATH``.

RBAC: a chat message's trace is readable by the message's session
owner OR any admin. Any other caller gets 404 (NOT 403 — same
existence-leak prevention as the chat / files routes).
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_session
from api.models import ChatMessage, ChatSession, User
from config.settings import trace_run_path


logger = logging.getLogger(__name__)


router = APIRouter(tags=["chat"])


# Cap on bytes read from each artefact. trajectory.jsonl can grow
# to several MB on long agent runs (one JSON line per tool call +
# its observation); a 5 MB cap prevents the API response from
# accidentally pulling a multi-megabyte payload over the wire.
_MAX_FILE_BYTES = 5 * 1024 * 1024
# Trajectory record cap — even within the byte cap, > 500 rows is
# more than any legitimate UI scrolls through.
_MAX_TRAJECTORY_ROWS = 500
_MAX_LLM_CALL_ROWS = 200


@router.get("/chat/messages/{message_id}/trace", response_model=dict)
async def get_message_trace(
    message_id: int,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the assistant message's on-disk trace bundle.

    Loads the message → its ``metadata_json.trace_path`` → resolves
    under STORAGE_PATH → reads ``query.json`` / ``trajectory.jsonl``
    / ``final.json`` / ``llm_calls.jsonl`` if present. Any missing
    artefact slot is omitted from the response (not an error).
    """
    res = await db.execute(
        select(ChatMessage).where(ChatMessage.id == message_id)
    )
    message: Optional[ChatMessage] = res.scalar_one_or_none()
    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="message not found",
        )
    # Ownership: load owning session and check user_id.
    session_res = await db.execute(
        select(ChatSession).where(ChatSession.id == message.session_id)
    )
    session: Optional[ChatSession] = session_res.scalar_one_or_none()
    if session is None:
        # Orphan message — should not happen under FK cascade. Treat
        # as 404 to keep the existence-leak invariant.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="message not found",
        )
    is_admin = user.role == "admin"
    if session.user_id != user.id and not is_admin:
        # Don't leak existence to a probing client — same convention
        # as get_session_for_user in api.services.chat.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="message not found",
        )

    metadata = _parse_metadata(message.metadata_json)
    trace_path_rel = (metadata or {}).get("trace_path") if metadata else None
    if not trace_path_rel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="message has no trace_path (likely a user message or a run without tracer)",
        )

    try:
        run_dir = trace_run_path(str(trace_path_rel))
    except ValueError as exc:
        # The settings helper rejects path-traversal escapes. Surface
        # as 400 so the operator can spot a poisoned metadata row,
        # rather than 500 which would obscure the security signal.
        logger.warning("rejected trace_path=%r: %s", trace_path_rel, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"trace_path resolves outside STORAGE_PATH",
        ) from exc
    if not run_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"trace dir not found on disk: {trace_path_rel}",
        )

    return {
        "message_id": message.id,
        "session_id": message.session_id,
        "trace_path": str(trace_path_rel),
        "flavor": _infer_flavor(trace_path_rel),
        "query": _read_json(run_dir / "query.json"),
        "trajectory": _read_jsonl(
            run_dir / "trajectory.jsonl", max_rows=_MAX_TRAJECTORY_ROWS
        ),
        "final": _read_json(run_dir / "final.json"),
        "llm_calls": _read_jsonl(
            run_dir / "llm_calls.jsonl", max_rows=_MAX_LLM_CALL_ROWS
        ),
    }


# ---------- helpers ----------


def _parse_metadata(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _infer_flavor(rel_path: str) -> str:
    """First path segment is the flavor (rag / agentic / web_rag / web_agent)."""
    parts = Path(rel_path).parts
    return parts[0] if parts else ""


def _read_json(path: Path) -> Optional[Any]:
    """Read a JSON artefact. Missing → None. Oversize → truncated note."""
    try:
        if not path.is_file():
            return None
        size = path.stat().st_size
        if size > _MAX_FILE_BYTES:
            return {
                "_truncated": True,
                "_reason": f"file size {size} > cap {_MAX_FILE_BYTES}",
            }
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.info("trace _read_json failed path=%s: %r", path, exc)
        return {"_error": f"{type(exc).__name__}: {exc}"}


def _read_jsonl(path: Path, *, max_rows: int) -> List[Any]:
    """Read a JSONL artefact line-by-line, capping at ``max_rows``.

    A malformed line is collected as ``{"_error": "..."}`` instead of
    aborting — the trajectory drawer should still render the legible
    rows when one frame got mangled.
    """
    if not path.is_file():
        return []
    out: List[Any] = []
    try:
        size = path.stat().st_size
        if size > _MAX_FILE_BYTES:
            out.append({
                "_truncated": True,
                "_reason": f"file size {size} > cap {_MAX_FILE_BYTES}",
            })
            return out
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                if len(out) >= max_rows:
                    out.append({"_truncated": True, "_reason": f"row cap {max_rows}"})
                    break
                line = raw.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    out.append({"_error": f"JSONDecodeError: {exc}", "_raw": line[:200]})
    except OSError as exc:
        logger.info("trace _read_jsonl failed path=%s: %r", path, exc)
        out.append({"_error": f"{type(exc).__name__}: {exc}"})
    return out
