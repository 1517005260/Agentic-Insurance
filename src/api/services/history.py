"""Multi-turn history reconstruction.

The chat-session DB rows are deliberately a thin pointer: each
assistant ``chat_messages`` row carries ``metadata_json.trace_path``
pointing at ``${STORAGE_PATH}/<flavor>/<date>/<run_id>/`` where the
runner persisted ``query.json`` / ``final.json`` / ``trajectory.jsonl``.

Per-turn pair shape:

* **user side** — taken from the immediately-preceding ``role='user'``
  ``chat_messages.content``. This is the *raw* text the user typed; we
  cannot read it from the trace because the runner threads
  ``composed_query`` (raw + prior turns prepended) into ``agent.run`` /
  ``answer()``, and ``final.json.query`` therefore contains that
  composed form. Reading raw user text from the DB row keeps history
  reconstruction non-recursive.
* **assistant side** — taken from ``final.json.answer`` so we honour
  the "trace is the source of truth" rule for assistant output. (It
  matches ``chat_messages.assistant.content`` byte-for-byte under
  normal operation; going through trace keeps the contract uniform and
  leaves room to strip ``assistant.content`` to a pointer later without
  touching this loader.)

Failure mode: a missing / malformed trace, missing user partner row,
etc. silently skips that turn. A user who deleted a trace folder
shouldn't break their later turns; the loader is best-effort. Returns
chronological ``[(user_query, assistant_answer), ...]`` with at most
``n_turns`` items.
"""
import json
import logging
from typing import List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.models import ChatMessage
from config.settings import trace_run_path


logger = logging.getLogger(__name__)


async def _read_assistant_answer(row: ChatMessage) -> Optional[str]:
    """Resolve ``final.json.answer`` from an assistant row's trace_path."""
    meta_raw = row.metadata_json
    if not meta_raw:
        return None
    try:
        meta = json.loads(meta_raw)
    except json.JSONDecodeError:
        logger.debug(
            "history: skipping message %s — malformed metadata_json", row.id
        )
        return None
    trace_path = meta.get("trace_path")
    if not isinstance(trace_path, str) or not trace_path:
        return None
    try:
        run_dir = trace_run_path(trace_path)
    except ValueError:
        logger.warning(
            "history: trace_path %r for message %s resolves outside STORAGE_PATH",
            trace_path,
            row.id,
        )
        return None
    final_path = run_dir / "final.json"
    if not final_path.is_file():
        return None
    try:
        final = json.loads(final_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.debug(
            "history: skipping message %s — final.json unreadable", row.id
        )
        return None
    answer = final.get("answer")
    if not isinstance(answer, str):
        return None
    # Empty (e.g. ABSTAIN with no body) still counts.
    return answer


async def load_recent_turns(
    db: AsyncSession,
    *,
    session_id: int,
    n_turns: int,
) -> List[Tuple[str, str]]:
    """Return the last ``n_turns`` ``(user_query, assistant_answer)`` pairs.

    Walks ``chat_messages`` newest-first, picks each ``role='assistant'``
    row, finds its preceding ``role='user'`` row (largest id smaller
    than the assistant id) and pairs them. Final result is reversed
    into chronological order so callers can prepend in turn order
    without re-reversing.

    A ``n_turns <= 0`` short-circuits to ``[]`` so callers can disable
    multi-turn by setting ``chat.history_turns`` to 0 without branching.

    See module docstring for the user/assistant source-of-truth rules.
    """
    if n_turns <= 0:
        return []

    # Pull every (user, assistant) row newest-first up to a
    # comfortable bound. ``2 * n_turns`` would suffice if every
    # assistant has a partner user, but a stale half-write (user
    # persisted but assistant never landed) would skip turns; pull
    # 4× and walk forward.
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .where(ChatMessage.role.in_(("user", "assistant")))
        .order_by(ChatMessage.id.desc())
        .limit(max(8, n_turns * 4))
    )
    rows = (await db.execute(stmt)).scalars().all()
    # rows is newest → oldest; index by id for partner lookup.
    by_id = {r.id: r for r in rows}

    pairs: List[Tuple[str, str]] = []
    seen_assistants = 0
    for row in rows:
        if seen_assistants >= n_turns:
            break
        if row.role != "assistant":
            continue
        # Find the most recent ``user`` row with id < this assistant
        # row's id. Walk by descending id rather than assuming
        # exactly id-1 (a tool message could land in between in
        # future runners).
        partner: Optional[ChatMessage] = None
        for candidate_id in sorted((rid for rid in by_id if rid < row.id), reverse=True):
            cand = by_id[candidate_id]
            if cand.role == "user":
                partner = cand
                break
        if partner is None:
            continue
        if not isinstance(partner.content, str):
            continue
        answer = await _read_assistant_answer(row)
        if answer is None:
            continue
        pairs.append((partner.content, answer))
        seen_assistants += 1

    # Reverse to chronological order.
    pairs.reverse()
    return pairs
