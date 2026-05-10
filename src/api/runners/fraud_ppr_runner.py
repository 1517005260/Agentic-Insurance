"""Fraud-PPR analysis runner — single PPR retrieval + single streamed LLM call.

Departs from the workbench scaffold: there is **no agent loop and no tool
registry**. The user's question drives one ``GraphService.ppr_subgraph``
call to materialize a candidate subgraph (seeds → actived entities →
passages + induced edges); we then format that subgraph into a system+user
prompt pair and stream ``LLMClient.chat_stream`` directly. The model's job
is to triage the subgraph for fraud signals — concentration, suspicious
paths, risk tier, next-step hints — and abstain when PPR found nothing.

Why not :func:`stream_workbench_agent`? That helper assumes the agent
will discover evidence iteratively via tool calls; here the evidence
arrives in one shot before the model speaks. Reusing it would require
disabling the tool registry, faking ``read`` envelopes for citation
extraction, and synthesising loop counts the model never ran. A direct
streamer is shorter and clearer.
"""
import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from api.runners.events import EventBus, EventType
from api.services.citation import CitationItem
from api.services.graph_service import GraphService, GraphServiceUnavailable
from config.config_store import ConfigStore
from model_client import LLMClient


logger = logging.getLogger(__name__)


_FLAVOR = "fraud_ppr"
_MAX_PASSAGES_IN_PROMPT = 30
_MAX_ENTITIES_IN_PROMPT = 20
# Each passage snippet capped so a 30-passage prompt stays well under
# the LLM's working context. ~280 chars ≈ ~140 zh tokens; 30 × that
# fits even for chain-of-thought-heavy prompt expansions.
_MAX_PASSAGE_SNIPPET_CHARS = 280


async def stream_fraud_ppr(
    *,
    query: str,
    file_ids: Optional[List[str]],
    graph_service: GraphService,
    llm: LLMClient,
    config: ConfigStore,
    result_future: Optional["asyncio.Future[Dict[str, Any]]"] = None,
) -> AsyncIterator[bytes]:
    """Stream one fraud-PPR analysis as SSE bytes.

    Run order:
      1. PPR subgraph (sync, ~300-500ms) on the threadpool — the result
         determines how many passages exist for citation.
      2. Build prompt + citation list.
      3. Stream LLM tokens through the bus.
      4. Emit ``citations → final → done`` per the workbench contract.
    """
    loop = asyncio.get_running_loop()
    bus = EventBus(loop=loop)

    system_prompt = str(config.get("prompt.fraud_ppr"))
    answer_max_tokens = int(config.get("rag.answer_max_tokens"))

    def run_in_thread() -> None:
        result_payload: Dict[str, Any] = {}
        try:
            try:
                subgraph = graph_service.ppr_subgraph(query, file_ids=file_ids)
            except GraphServiceUnavailable as exc:
                # Translate "graph not built yet" into a typed error frame
                # so the frontend can show a friendlier "ingest first" hint
                # rather than a generic 500.
                _emit_failure(
                    bus, loop, result_future, "GraphServiceUnavailable", str(exc)
                )
                return

            passages = subgraph.get("passages") or []
            citations = _build_citations(passages)
            # Pull a short passage snippet for each citation so the LLM
            # can describe what the neighboring conditions actually say
            # — file_id/page_id stubs alone forced the model to invent
            # clause names. The store is in-memory, so this is a dict
            # lookup per passage.
            try:
                passage_text_map = (
                    graph_service._channel.passage_store.hash_id_to_text  # type: ignore[attr-defined]
                )
            except Exception:
                passage_text_map = {}
            user_prompt = _build_user_prompt(
                query, subgraph, citations, passages, passage_text_map
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            answer_chunks: List[str] = []
            usage: Dict[str, Any] = {}
            cost: float = 0.0
            for frame in llm.chat_stream(messages, max_tokens=answer_max_tokens):
                if bus.is_closed:
                    # Client gave up — stop billing for tokens nobody reads.
                    break
                delta = frame.get("delta")
                if delta:
                    answer_chunks.append(delta)
                    bus.push(EventType.TOKEN, {"delta": delta})
                if "usage" in frame:
                    usage = frame.get("usage") or {}
                if "cost" in frame:
                    cost = float(frame.get("cost") or 0.0)

            answer = "".join(answer_chunks)
            citation_items = [c.to_dict() for c in citations]
            bus.push(EventType.CITATIONS, {"items": citation_items})

            mode = subgraph.get("mode", "ppr")
            result_payload = {
                "answer": answer,
                "answer_chars": len(answer),
                "flavor": _FLAVOR,
                "mode": mode,                       # ppr | no_seeds | no_graph
                "subgraph_counts": {
                    "seeds": len(subgraph.get("seeds") or []),
                    "actived_entities": len(subgraph.get("actived_entities") or []),
                    "passages": len(passages),
                    "edges": len(subgraph.get("edges") or []),
                },
                "citations": citation_items,
                "citations_count": len(citation_items),
                "usage": usage,
                "total_cost": cost,
            }
            bus.push(EventType.FINAL, result_payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("fraud-ppr runner failed")
            _emit_failure(
                bus, loop, result_future, type(exc).__name__, str(exc)
            )
            return

        _set_future(loop, result_future, result_payload)
        bus.close()

    loop.run_in_executor(None, run_in_thread)
    async for chunk in bus.stream():
        yield chunk


# --------------------------------------------------------------- prompt build


def _build_citations(passages: List[Dict[str, Any]]) -> List[CitationItem]:
    """Mint CitationItems in PPR-rank order.

    Drops malformed entries silently — bad citations would corrupt the
    legend the frontend renders. ``page_preview`` left empty: PPR does
    not read passage text, so the drawer falls back to the PDF render.
    """
    items: List[CitationItem] = []
    for p in passages[:_MAX_PASSAGES_IN_PROMPT]:
        file_id = p.get("file_id")
        page_id = p.get("page_id")
        if not file_id or not page_id:
            continue
        items.append(
            CitationItem(
                sup=len(items) + 1,
                file_id=str(file_id),
                page_id=str(page_id),
                page_number=_parse_page_number(str(page_id)),
            )
        )
    return items


def _parse_page_number(page_id: str) -> Optional[int]:
    """``p_0007`` → ``7``. Returns None for malformed inputs."""
    if page_id.startswith("p_"):
        try:
            return int(page_id[2:])
        except ValueError:
            return None
    return None


def _build_user_prompt(
    query: str,
    subgraph: Dict[str, Any],
    citations: List[CitationItem],
    passages: List[Dict[str, Any]],
    passage_text_map: Dict[str, str],
) -> str:
    """Render the subgraph as plain text so the LLM can quote it.

    Numbered passages line up with the ``CitationItem.sup`` so the model
    can use ``[^k]`` directly. Each passage carries a truncated snippet
    of its actual text (looked up from the in-memory passage store)
    so the LLM can name specific clauses instead of inventing them.
    Entities are sorted by score so the model's "concentration" rule
    has a stable ordering to walk.
    """
    mode = subgraph.get("mode", "ppr")
    if mode != "ppr":
        return (
            f"## 用户问题\n{query}\n\n"
            f"## 子图摘要\nPPR 返回 mode='{mode}' — 没有可用的实体/段落证据。"
        )

    seeds = subgraph.get("seeds") or []
    actived = sorted(
        subgraph.get("actived_entities") or [],
        key=lambda e: float(e.get("score") or 0.0),
        reverse=True,
    )[:_MAX_ENTITIES_IN_PROMPT]
    edges = subgraph.get("edges") or []

    seeds_block = (
        "\n".join(
            f"- {s.get('surface') or '?'} (sim={float(s.get('similarity') or 0.0):.3f})"
            for s in seeds
        )
        or "- (none)"
    )

    actived_block = (
        "\n".join(
            f"- {e.get('surface') or '?'} (score={float(e.get('score') or 0.0):.3f}, tier={int(e.get('iteration_tier') or 0)})"
            for e in actived
        )
        or "- (none)"
    )

    # ``citations`` is built from ``passages`` by skipping malformed
    # entries (no file_id / no page_id). To zip snippets back onto
    # citations, pre-filter passages with the same predicate so
    # ``citations[i]`` aligns with ``valid_passages[i]``.
    valid_passages = [
        p for p in passages[:_MAX_PASSAGES_IN_PROMPT]
        if p.get("file_id") and p.get("page_id")
    ]
    passage_lines: List[str] = []
    for i, c in enumerate(citations):
        p = valid_passages[i] if i < len(valid_passages) else {}
        hash_id = p.get("hash_id") if isinstance(p, dict) else None
        score = float(p.get("score") or 0.0) if isinstance(p, dict) else 0.0
        snippet_raw = passage_text_map.get(hash_id, "") if hash_id else ""
        snippet = _shorten(snippet_raw, _MAX_PASSAGE_SNIPPET_CHARS)
        head = f"[^{c.sup}] file_id={c.file_id} page={c.page_id} score={score:.3f}"
        if snippet:
            passage_lines.append(f"{head}\n  {snippet}")
        else:
            passage_lines.append(f"{head}  (无可用正文)")
    passages_block = "\n".join(passage_lines) or "(none)"

    edge_type_counts: Dict[str, int] = {}
    for e in edges:
        t = str(e.get("type") or "unknown")
        edge_type_counts[t] = edge_type_counts.get(t, 0) + 1
    edge_summary = ", ".join(f"{k}={v}" for k, v in edge_type_counts.items()) or "(none)"

    return (
        f"## 用户问题\n{query}\n\n"
        f"## 子图摘要\n"
        f"### Seeds ({len(seeds)})\n{seeds_block}\n\n"
        f"### Actived entities (top {len(actived)})\n{actived_block}\n\n"
        f"### Passages (按 PPR 得分排序，可引用)\n{passages_block}\n\n"
        f"### Edges\ntotal={len(edges)}, by_type: {edge_summary}\n"
    )


def _shorten(text: str, n: int) -> str:
    """Trim text to ≤ n chars on a sentence-ish boundary so the snippet
    isn't sliced mid-clause. Falls back to a hard cut + ellipsis."""
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= n:
        return text
    head = text[: n - 1]
    # Prefer the last 。/. before the cap; tolerate up to half-window
    # back-off so we don't return something far shorter than asked.
    for sep in ("。", "；", ";", ".", " "):
        idx = head.rfind(sep)
        if idx >= n // 2:
            return head[: idx + 1] + "…"
    return head + "…"


# --------------------------------------------------------------- bus helpers


def _emit_failure(
    bus: EventBus,
    loop: asyncio.AbstractEventLoop,
    result_future: Optional["asyncio.Future[Dict[str, Any]]"],
    error_type: str,
    message: str,
) -> None:
    """Best-effort failure flush mirroring the workbench scaffold.

    Always emits an empty citations frame before close so the frontend's
    CitationDrawer doesn't carry over stale state from a previous run.
    """
    try:
        bus.push(EventType.CITATIONS, {"items": []})
    except Exception:
        logger.exception("fraud-ppr: empty citations push failed in error path")
    if result_future is not None:
        def _set() -> None:
            if not result_future.done():
                result_future.set_exception(RuntimeError(f"{error_type}: {message}"))
        try:
            loop.call_soon_threadsafe(_set)
        except RuntimeError:
            logger.debug("fraud-ppr: loop closed before exception set")
    bus.close(error=f"{error_type}: {message}", error_type=error_type)


def _set_future(
    loop: asyncio.AbstractEventLoop,
    result_future: Optional["asyncio.Future[Dict[str, Any]]"],
    payload: Dict[str, Any],
) -> None:
    if result_future is None:
        return
    def _set() -> None:
        if not result_future.done():
            result_future.set_result(payload)
    try:
        loop.call_soon_threadsafe(_set)
    except RuntimeError:
        logger.debug("fraud-ppr: loop closed before result set")
