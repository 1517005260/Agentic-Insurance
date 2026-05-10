"""Session / message persistence for the chat surface.

Routes call these helpers; runners hand back a result dict that
:func:`build_assistant_metadata_*` turns into a JSON blob ready for
``ChatMessage.metadata_json``. Trace details are NOT stored — only the
relative ``trace_path`` (``<flavor>/<date>/<run_id>``); resolution
goes through :func:`config.settings.trace_run_path`.

Ownership rule: every read / write of a session goes through
:func:`get_session_for_user`, which raises 404 (never 403) when the
session belongs to a different user. We don't leak existence — a
404 is the same response as "no such id at all".
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.models import ChatMessage, ChatSession


# --------------------------------------------------------------- session


async def create_session(
    db: AsyncSession,
    *,
    user_id: int,
    mode: str,
    agent_kind: Optional[str],
    title: str,
    web: bool = False,
) -> ChatSession:
    """Insert a new session. Validation matches DB CHECK constraints."""
    session = ChatSession(
        user_id=user_id,
        title=title,
        mode=mode,
        agent_kind=agent_kind,
        web=1 if web else 0,
    )
    db.add(session)
    await db.flush()  # gives us session.id without an extra round trip
    return session


async def get_session_for_user(
    db: AsyncSession, session_id: int, user_id: int
) -> ChatSession:
    """Load a session, enforcing ownership.

    Returns the session row when ``session.user_id == user_id``;
    otherwise 404 (NOT 403 — we don't leak existence of another
    user's sessions to a probing client).
    """
    res = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    session = res.scalar_one_or_none()
    if session is None or session.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="session not found",
        )
    return session


async def list_sessions_for_user(
    db: AsyncSession, user_id: int, *, limit: int = 50, offset: int = 0
) -> List[ChatSession]:
    """List sessions newest-first by ``updated_at`` (matches the SQLite index)."""
    res = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(res.scalars().all())


async def update_session_title(
    db: AsyncSession, session: ChatSession, *, title: str
) -> ChatSession:
    session.title = title
    # ``updated_at`` is server-side onupdate=now; touching any column
    # is enough. No-op assign forces the UPDATE to run.
    return session


async def delete_session(db: AsyncSession, session: ChatSession) -> None:
    """Hard delete. Messages cascade via FK ON DELETE CASCADE."""
    await db.delete(session)


# --------------------------------------------------------------- messages


async def list_messages(db: AsyncSession, session_id: int) -> List[ChatMessage]:
    res = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.asc())
    )
    return list(res.scalars().all())


async def persist_user_message(
    db: AsyncSession, *, session_id: int, content: str
) -> ChatMessage:
    """Insert a 'user' role message and bump the parent session's updated_at.

    Touching ``ChatSession.updated_at`` keeps the "newest first"
    ordering of ``GET /chat/sessions`` correct — a session with active
    messaging should sort above an old idle one even if its create
    time is older. The bump is done in the same flush as the message
    insert so caller commits land both atomically.
    """
    msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=content,
        metadata_json=None,
    )
    db.add(msg)
    # Touch parent session so onupdate=now fires. We could write
    # session.updated_at = datetime.now() directly but the onupdate
    # column expression already exists; nudging any tracked field
    # forces SQLA to emit the UPDATE.
    res = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    parent = res.scalar_one_or_none()
    if parent is not None:
        parent.updated_at = datetime.now(timezone.utc)
    await db.flush()
    return msg


async def persist_assistant_message(
    db: AsyncSession,
    *,
    session_id: int,
    content: str,
    metadata: Dict[str, Any],
) -> ChatMessage:
    """Insert an 'assistant' role message with serialized metadata. Caller commits.

    ``metadata`` is JSON-encoded with ``ensure_ascii=False`` so CJK
    citation previews stay readable when an admin spot-checks the row.
    Empty or all-None metadata still serialises (`{}`) — keeps the
    column shape consistent so the front-end never has to branch.
    """
    msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=content,
        metadata_json=json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
    )
    db.add(msg)
    await db.flush()
    return msg


# --------------------------------------------------- assistant metadata


def build_rag_assistant_metadata(
    *,
    exit_reason: str,
    trace_path: Optional[str],
    citations: Optional[List[Dict[str, Any]]] = None,
    timings_ms: Optional[Dict[str, int]] = None,
    channels_hit_counts: Optional[Dict[str, int]] = None,
    reranked_count: Optional[int] = None,
    model: Optional[str] = None,
    error: Optional[str] = None,
    original_exit_reason: Optional[str] = None,
    # Web RAG 多轮重写产物 — 用户原 query 与最终送给 Tavily 的
    # standalone search query 通常不同；同时记录 rewrite_error 让
    # 审计/answer detail UI 能解释"为什么这一轮的 search query 没
    # 被重写"。本地 RAG 路径不会传这几项，全为 None → 全跳过。
    search_query: Optional[str] = None,
    original_query: Optional[str] = None,
    rewrite_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Compose the metadata dict for a RAG assistant message.

    Drops keys whose values are None so the stored JSON stays compact
    and the frontend can use ``"key" in metadata`` checks.
    """
    out: Dict[str, Any] = {"exit_reason": exit_reason}
    for k, v in (
        ("trace_path", trace_path),
        ("citations", citations),
        ("timings_ms", timings_ms),
        ("channels_hit_counts", channels_hit_counts),
        ("reranked_count", reranked_count),
        ("model", model),
        ("error", error),
        ("original_exit_reason", original_exit_reason),
        ("search_query", search_query),
        ("original_query", original_query),
        ("rewrite_error", rewrite_error),
    ):
        if v is not None:
            out[k] = v
    return out


def build_agent_assistant_metadata(
    *,
    exit_reason: str,
    loops: int,
    trace_path: Optional[str],
    decision: Optional[str] = None,
    total_cost: Optional[float] = None,
    input_tokens_total: Optional[int] = None,
    cached_tokens_total: Optional[int] = None,
    output_tokens_total: Optional[int] = None,
    model: Optional[str] = None,
    error: Optional[str] = None,
    original_exit_reason: Optional[str] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compose the metadata dict for any agent assistant message (base/proof/graph/web).

    ``citations`` is populated for web-agent runs (kind="web"); other
    kinds leave it None so the stored JSON stays compact.
    """
    out: Dict[str, Any] = {"exit_reason": exit_reason, "loops": loops}
    for k, v in (
        ("trace_path", trace_path),
        ("decision", decision),
        ("total_cost", total_cost),
        ("input_tokens_total", input_tokens_total),
        ("cached_tokens_total", cached_tokens_total),
        ("output_tokens_total", output_tokens_total),
        ("model", model),
        ("error", error),
        ("original_exit_reason", original_exit_reason),
        ("citations", citations),
    ):
        if v is not None:
            out[k] = v
    return out
