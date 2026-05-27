"""Chat session + streaming routes.

Two surfaces share the same runners:

* **Session-aware** (`/chat/sessions/...`): persists user message
  before the stream starts, persists assistant message (with
  citations + trace_path) after the stream finishes.
* **Session-less** (`/rag/stream`, `/agent/stream`): smoke endpoints
  for direct shell scripts; no DB writes.

Both surfaces speak the same SSE protocol (`event: status` →
`retrieval`/`tool_call`/... → `final` → `done`); the frontend can
pipe responses through the same handler regardless of which entrypoint
it hit.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_session
from api.models import User
from api.runners.agent_runner import stream_agent
from api.runners.rag_runner import stream_rag
from api.runners.web_rag_runner import stream_web_rag
from api.schemas.chat import (
    AgentStreamRequest,
    MessageOut,
    MessagePost,
    RagStreamRequest,
    SessionCreate,
    SessionOut,
    SessionUpdate,
    WebRagStreamRequest,
)
from api.services import chat as chat_svc
from api.services.history import load_recent_turns
from tracer import Tracer


logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


# Common SSE response headers — disable proxy buffering so frames flush
# in real time. Used by every streaming endpoint here.
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


# --------------------------------------------------- session lifecycle ----


@router.post("/chat/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: SessionCreate,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> SessionOut:
    session = await chat_svc.create_session(
        db,
        user_id=user.id,
        mode=body.mode,
        agent_kind=body.agent_kind,
        title=body.title,
        web=body.web,
    )
    # get_session commits at request end; chat_svc.flush() already
    # populated the id so it's safe to serialise the row now.
    return SessionOut.model_validate(session)


@router.get("/chat/sessions", response_model=List[SessionOut])
async def list_sessions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> List[SessionOut]:
    rows = await chat_svc.list_sessions_for_user(
        db, user_id=user.id, limit=limit, offset=offset
    )
    return [SessionOut.model_validate(r) for r in rows]


@router.get("/chat/sessions/{session_id}", response_model=dict)
async def get_session_detail(
    session_id: int,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict:
    """Session header + full message list. Light enough for chat history view."""
    session = await chat_svc.get_session_for_user(db, session_id, user.id)
    msgs = await chat_svc.list_messages(db, session_id)
    return {
        "session": SessionOut.model_validate(session).model_dump(),
        "messages": [
            _message_out(m).model_dump() for m in msgs
        ],
    }


@router.patch("/chat/sessions/{session_id}", response_model=SessionOut)
async def patch_session(
    session_id: int,
    body: SessionUpdate,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> SessionOut:
    session = await chat_svc.get_session_for_user(db, session_id, user.id)
    await chat_svc.update_session_title(db, session, title=body.title)
    return SessionOut.model_validate(session)


@router.delete(
    "/chat/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_session(
    session_id: int,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> Response:
    session = await chat_svc.get_session_for_user(db, session_id, user.id)
    await chat_svc.delete_session(db, session)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ------------------------------------------ message stream (per session) ----


@router.post("/chat/sessions/{session_id}/messages")
async def post_session_message(
    session_id: int,
    body: MessagePost,
    request: Request,
    db: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Stream the assistant reply for a new user message in this session.

    Lifecycle:

    1. Validate session + ownership.
    2. Persist user message + **explicitly commit + bump session.updated_at**
       so concurrent ``GET`` sees the new turn even while the stream is
       mid-flight. We don't rely on the ``get_session`` dep tail-commit
       because FastAPI dependency teardown happens after the response
       body finishes iterating, and a client disconnect mid-stream
       would otherwise roll back the user message.
    3. Open the SSE response. The runner runs the pipeline / agent,
       streams events, and resolves a result-future once it returns.
    4. After the bus drains we ``await`` the future (shielded against
       response-task cancellation) and write the assistant message
       inside a fresh DB session.
    """
    session = await chat_svc.get_session_for_user(db, session_id, user.id)

    user_msg = await chat_svc.persist_user_message(
        db, session_id=session.id, content=body.content
    )
    # Explicit commit — see lifecycle docstring above. Don't rely on
    # ``get_session`` tail-commit; it fires after the response body
    # finishes iterating, by which time the client could disconnect
    # mid-stream and roll the user message back.
    await db.commit()

    # Multi-turn: load prior (user, assistant) pairs from
    # chat_messages + trace files now, while ``db`` is still open.
    # The streaming generator below outlives ``get_session``'s tail-
    # commit teardown, so the loader has to run on the *handler*
    # session — closing-over the resulting list is safe.
    cfg = getattr(request.app.state, "config", None)
    n_turns = cfg.chat_history_turns() if cfg is not None else 0
    history = await load_recent_turns(db, session_id=session.id, n_turns=n_turns)

    # Capture only scalar session info — the ORM session attached to
    # ``session`` will close when the route returns, and the streaming
    # generator below outlives that.
    session_snapshot = _SessionSnapshot(
        id=session.id,
        mode=session.mode,
        agent_kind=session.agent_kind,
        web=bool(session.web),
    )

    return StreamingResponse(
        _persist_after_stream(
            request=request,
            session=session_snapshot,
            user_message_id=user_msg.id,
            content=body.content,
            history=history,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _is_done_frame(chunk: bytes) -> bool:
    """Detect the SSE ``event: done`` frame.

    Used by ``_persist_after_stream`` to gate the terminal frame
    behind assistant persistence. Cheap byte-prefix check; the SSE
    encoder (api/sse.py:format_event) always emits exactly
    ``event: done\\ndata: {}\\n\\n`` so a literal prefix match is
    safe. Heartbeats are comment-prefixed (``: keepalive``) and
    won't false-match.
    """
    return chunk.startswith(b"event: done\n")


@dataclass(frozen=True)
class _SessionSnapshot:
    """Plain snapshot of session fields the streaming generator needs.

    Avoids carrying the ORM ``ChatSession`` (whose owning AsyncSession
    closes when the route handler returns) across the response-body
    boundary. The generator runs *after* the request session is gone.
    """

    id: int
    mode: str
    agent_kind: Optional[str]
    web: bool


async def _persist_after_stream(
    *,
    request: Request,
    session: "_SessionSnapshot",
    user_message_id: int,
    content: str,
    history: Optional[List[tuple]] = None,
) -> AsyncIterator[bytes]:
    """Stream bytes; on exit, persist the assistant message in a fresh session.

    Cancellation handling: when the response task is cancelled (client
    disconnect), the worker thread keeps running until the agent /
    pipeline returns and resolves ``result_future``. We schedule
    ``_persist_assistant`` as an independent background task that is
    NOT linked to the response task's lifetime, so the assistant
    message lands even if the client disappeared mid-stream. The body
    we lose is recorded as ``exit_reason='client_disconnect'``.
    """
    loop = asyncio.get_running_loop()
    result_future: "asyncio.Future" = loop.create_future()
    # Trace flavor segregates web runs from local runs so audit
    # doesn't have to grep across mixed trees.
    if session.mode == "rag":
        flavor = "web_rag" if session.web else "rag"
    else:
        flavor = "web_agent" if session.web else "agentic"
    tracer = Tracer(flavor=flavor)

    cfg = getattr(request.app.state, "config", None)
    history_arg = list(history) if history else None
    if session.mode == "rag" and not session.web:
        gen = stream_rag(
            query=content,
            file_ids=None,
            pipeline=request.app.state.rag_pipeline,
            config=cfg,
            tracer=tracer,
            result_future=result_future,
            history=history_arg,
        )
    elif session.mode == "rag" and session.web:
        gen = stream_web_rag(
            query=content,
            llm=request.app.state.rag_pipeline.llm,
            tavily=request.app.state.tavily_client,
            config=cfg,
            tracer=tracer,
            result_future=result_future,
            history=history_arg,
        )
    else:
        # mode='agent'. web=True forces the dedicated web agent
        # singleton (ChatSchema already enforces kind='base' here).
        if session.web:
            agent = request.app.state.web_agent
            stream_kind = "web"
        else:
            agent = _resolve_agent(request, session.agent_kind or "")
            stream_kind = session.agent_kind or "base"
        gen = stream_agent(
            query=content,
            kind=stream_kind,
            agent=agent,
            config=cfg,
            tracer=tracer,
            result_future=result_future,
            history=history_arg,
        )

    # Buffer the terminal ``done`` frame so we can await
    # persistence BEFORE the client sees done. Otherwise the
    # client (frontend useSSE) parses done → cancels the reader →
    # cancels this generator → ``CancelledError`` interrupts the
    # awaited persist, falling back to the detached path and
    # reintroducing the immediate-follow-up history race.
    pending_done: Optional[bytes] = None
    disconnected = False
    try:
        async for chunk in gen:
            if pending_done is not None:
                yield pending_done
                pending_done = None
            if _is_done_frame(chunk):
                pending_done = chunk
                continue
            yield chunk
    except asyncio.CancelledError:
        # Client disconnected mid-stream. Detach the persistence
        # task so FastAPI's response task can fully unwind without
        # cancelling the writer.
        disconnected = True
        asyncio.create_task(
            _persist_assistant_when_ready(
                session=session,
                result_future=result_future,
                disconnected=True,
            )
        )
        raise
    # Clean stream completion. Order matters:
    #   1) await persist — assistant row commits
    #   2) yield done    — only now can the client safely fire a
    #                      follow-up POST that loads multi-turn
    #                      history (the just-finished assistant row
    #                      is guaranteed visible).
    #
    # If the client disconnects DURING the persist await (tab close
    # in the ~tens of ms between drain and yield), CancelledError
    # tears the await down. Detach a fresh writer task so the row
    # still lands; otherwise that tail-disconnect would silently
    # drop the assistant row.
    try:
        await _persist_assistant_when_ready(
            session=session,
            result_future=result_future,
            disconnected=False,
        )
    except asyncio.CancelledError:
        asyncio.create_task(
            _persist_assistant_when_ready(
                session=session,
                result_future=result_future,
                disconnected=True,
            )
        )
        raise
    if pending_done is not None:
        yield pending_done


async def _persist_assistant_when_ready(
    *,
    session: "_SessionSnapshot",
    result_future: "asyncio.Future",
    disconnected: bool,
) -> None:
    """Wait for the worker to finish; persist the assistant row.

    Runs as an independent background task so it survives
    response-task cancellation. ``shield`` guards the future await in
    case the parent shuts the loop down before the worker returns.
    """
    try:
        # Order matters: wait_for(shield(future), ...). The reverse
        # ``shield(wait_for(...))`` still lets wait_for cancel the
        # inner future on timeout, which would also cancel any other
        # consumers of the same future.
        payload = await asyncio.wait_for(asyncio.shield(result_future), timeout=900)
    except asyncio.TimeoutError:
        logger.error("assistant persistence: result future timeout (15min)")
        payload = {"answer": "", "exit_reason": "error", "error": "timeout"}
    except asyncio.CancelledError:
        # Loop is shutting down. Best effort: don't persist, log only.
        logger.warning("assistant persistence task cancelled before result arrived")
        return
    except Exception as exc:
        logger.warning("assistant run failed: %r — persisting error message", exc)
        payload = {"answer": "", "exit_reason": "error", "error": str(exc)}

    if disconnected and payload.get("exit_reason") not in ("error",):
        # The worker completed but the client never saw the tail. Mark
        # so audit can distinguish a clean reply from one nobody read.
        # Each runner uses its own success vocabulary (RAG: "ok";
        # base/graph: "natural"/"max_loops_exceeded"; proof:
        # "finalized"/...) — flatten any non-error to client_disconnect
        # but stash the original for diagnostics.
        original = payload.get("exit_reason")
        payload = {**payload, "exit_reason": "client_disconnect"}
        if original is not None:
            payload["original_exit_reason"] = original

    # Detached task — must catch its own errors. If the session was
    # deleted while the run was in flight, the FK on chat_messages
    # would surface as IntegrityError; we don't want an unobserved
    # task exception polluting the loop.
    try:
        await _persist_assistant(session=session, payload=payload)
    except Exception:
        logger.exception(
            "failed to persist assistant message for session %d (run completed)",
            session.id,
        )


async def _persist_assistant(*, session: "_SessionSnapshot", payload: dict) -> None:
    """Open a fresh DB session and write the assistant message."""
    from api.db import session_scope

    if session.mode == "rag":
        metadata = chat_svc.build_rag_assistant_metadata(
            exit_reason=payload.get("exit_reason", "ok"),
            trace_path=payload.get("trace_path"),
            citations=payload.get("citations"),
            timings_ms=payload.get("timings_ms"),
            channels_hit_counts=payload.get("channels_hit_counts"),
            reranked_count=payload.get("reranked_count"),
            error=payload.get("error"),
            original_exit_reason=payload.get("original_exit_reason"),
            # Only the web_rag runner populates these; local-RAG payload
            # leaves them absent → metadata builder drops them.
            search_query=payload.get("search_query"),
            original_query=payload.get("original_query"),
            rewrite_error=payload.get("rewrite_error"),
        )
    else:
        metadata = chat_svc.build_agent_assistant_metadata(
            exit_reason=payload.get("exit_reason", "ok"),
            loops=payload.get("loops") or 0,
            trace_path=payload.get("trace_path"),
            decision=payload.get("decision"),
            total_cost=payload.get("total_cost"),
            input_tokens_total=payload.get("input_tokens_total"),
            cached_tokens_total=payload.get("cached_tokens_total"),
            output_tokens_total=payload.get("output_tokens_total"),
            error=payload.get("error"),
            original_exit_reason=payload.get("original_exit_reason"),
            # Web-agent runs accumulate WebCitation list; other kinds
            # leave it None and the metadata builder drops the key.
            citations=payload.get("citations"),
        )

    async with session_scope() as db:
        await chat_svc.persist_assistant_message(
            db,
            session_id=session.id,
            content=payload.get("answer", ""),
            metadata=metadata,
        )


def _resolve_agent(request: Request, kind: str):
    """Pick the agent singleton for the requested kind."""
    state = request.app.state
    if kind == "base":
        return state.base_agent
    if kind == "proof":
        return state.proof_agent
    if kind == "graph":
        return state.graph_agent
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"unknown agent_kind: {kind!r}",
    )


# ---------------------------------------------------- session-less stream ----


@router.post("/rag/stream")
async def rag_stream(
    body: RagStreamRequest,
    request: Request,
    _user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Smoke endpoint: streams a RAG answer with no session persistence."""
    return StreamingResponse(
        stream_rag(
            query=body.query,
            file_ids=None,
            pipeline=request.app.state.rag_pipeline,
            config=getattr(request.app.state, "config", None),
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/agent/stream")
async def agent_stream(
    body: AgentStreamRequest,
    request: Request,
    _user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Smoke endpoint: streams a base/proof/graph/web agent run, no persistence.

    ``body.web=True`` swaps in the dedicated web agent singleton with
    its own kwargs map. Schema validation already rejected the
    forbidden combos (web=True + proof/graph).
    """
    if body.web:
        agent = request.app.state.web_agent
        kind = "web"
    else:
        agent = _resolve_agent(request, body.kind)
        kind = body.kind
    return StreamingResponse(
        stream_agent(
            query=body.query,
            kind=kind,
            agent=agent,
            config=getattr(request.app.state, "config", None),
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/web-rag/stream")
async def web_rag_stream(
    body: WebRagStreamRequest,
    request: Request,
    _user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Smoke endpoint: streams a single-call web RAG, no persistence."""
    return StreamingResponse(
        stream_web_rag(
            query=body.query,
            llm=request.app.state.rag_pipeline.llm,
            tavily=request.app.state.tavily_client,
            config=getattr(request.app.state, "config", None),
            include_domains=body.include_domains,
            exclude_domains=body.exclude_domains,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ------------------------------------------------------------- helpers ----


def _message_out(msg) -> MessageOut:
    """Map ORM row to the MessageOut DTO, parsing JSON metadata."""
    return MessageOut(
        id=msg.id,
        role=msg.role,
        content=msg.content,
        metadata=msg.metadata_json,  # validator parses JSON
        created_at=msg.created_at,
    )
