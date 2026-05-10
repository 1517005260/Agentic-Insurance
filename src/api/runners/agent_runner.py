"""Agent runner — base / proof / graph / web behind one streaming entry point.

Same EventBus pattern as :mod:`api.runners.rag_runner`. Adds:

* per-tool ``is_evidence`` tag on ``tool_result`` events (so the
  frontend can render read_page / proof_scan / graph_explore-neighbors
  as inline citation cards instead of generic explore steps);
* tracer attachment so the assistant message can persist a relative
  ``trace_path`` for later detail lookup;
* result accumulation surfaced via an ``asyncio.Future`` so the route
  handler can ``await`` the agent's return value once the bus drains
  and write the assistant message inside the request's async session;
* for ``kind="web"`` only: accumulate ``web_search`` / ``web_fetch``
  envelopes into a URL pool, parse the model's ``## Sources`` section
  for the canonical sup → url legend, and emit one canonical
  ``citations`` SSE event before the runner's own ``final`` (the
  agent-internal ``final`` is swallowed to keep ``citations → final
  → done`` ordering, mirroring :mod:`api.runners._workbench`).
* for ``kind in {base, proof, graph}``: same ``_accumulate_read_citations``
  pattern as :mod:`api.runners._workbench` — every ``read`` envelope
  contributes its ``units`` to a deduped sup-numbered list emitted as
  one ``citations`` event before ``final``. The default chat prompts
  use ``[file_id#page]`` markers (so the citations frame is metadata-
  only); admin-supplied prompts that switch to ``[^k]`` style get
  clickable sup → drawer just like workbench pages do. The frontend's
  evidence-chip path also still works (reverse-derived from
  ``is_evidence=true`` tool_results) — both surfaces are additive.
"""
import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

from agentic.agent.base import BaseAgent
from agentic.agent.proof_agent import ProofAgent, ProofRunResult
from api.runners._tracing import CapturingTracer
from api.runners._workbench import _accumulate_read_citations
from api.runners.events import EventBus, EventType
from api.services.citation import CitationItem
from config.config_store import ConfigStore
from config.settings import relativize_trace_dir
from tracer import Tracer


logger = logging.getLogger(__name__)


# ----------------------------------------------------------- evidence ----

# Tools whose ``tool_result`` payload should carry ``is_evidence=True``.
# Updated after reviewing the actual tool catalog: only the two tools
# that return verbatim page / atom content count. Everything else
# (semantic / bm25 / pattern / graph_explore in any mode) is discovery
# — the model still needs to call ``read`` to commit to a page as
# evidence. ``read`` is the single name registered by ReadTool
# (`agentic/tools/acquisition/read.py:74`); ``proof_scan`` is the
# proof-side atom reader.
_EVIDENCE_TOOL_NAMES: frozenset[str] = frozenset({
    "read",
    "proof_scan",
})


# Tools whose envelope feeds the web citation accumulator. ``web_search``
# returns Tavily ``hits[]``; ``web_fetch`` returns a single (url, title,
# text) triple. Both append to the same deduped URL pool so the model's
# ``[^k]`` markers can be resolved no matter which tool surfaced the
# source.
_WEB_CITATION_TOOLS: frozenset[str] = frozenset({"web_search", "web_fetch"})


def _is_evidence(name: Optional[str], args: Optional[Dict[str, Any]]) -> bool:
    """Decide whether this tool call counts as an inline citation.

    Flat name check: every other tool (search / explore / list) only
    returns candidates the model still has to read separately.
    """
    return name in _EVIDENCE_TOOL_NAMES


# --------------------------------------------------------- main entry ----


def _make_proof_final_payload(result: ProofRunResult) -> Dict[str, Any]:
    return {
        "answer": result.answer,
        "decision": result.decision,
        "exit_reason": result.exit_reason,
        "loops": result.loops,
        "total_cost": result.total_cost,
    }


def _make_base_final_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "answer": result.get("answer", ""),
        "exit_reason": result.get("exit_reason"),
        "loops": result.get("loops"),
        "total_cost": result.get("total_cost"),
        "input_tokens_total": result.get("input_tokens_total"),
        "cached_tokens_total": result.get("cached_tokens_total"),
        "output_tokens_total": result.get("output_tokens_total"),
    }


def _compose_query_with_history(
    history: List[Tuple[str, str]], current_query: str
) -> str:
    """Stitch prior (query, answer) pairs in front of ``current_query``.

    Agent runs are message-list-driven (``[system, user(query)]`` →
    iterate); rather than reach into BaseAgent / ProofAgent and rebuild
    a multi-turn message list with role=assistant turns (which would
    drag tool_call / tool_result accounting along), we keep each agent
    invocation as a self-contained "long internal conversation" and
    inject prior context as a multi-paragraph user prefix.

    The model treats the prior turns as background context against
    which to answer the *current* turn — same effect, simpler plumbing,
    and trace files (one per agent.run) stay independent so audit /
    replay still work per-turn.
    """
    if not history:
        return current_query
    blocks = [
        f"--- previous turn ---\nUser: {q}\n\nAssistant: {a}"
        for q, a in history
    ]
    return "\n\n".join(blocks) + f"\n\n--- current turn ---\nUser: {current_query}"


async def stream_agent(
    *,
    query: str,
    kind: str,                                    # "base" | "proof" | "graph"
    agent: Any,                                   # BaseAgent or ProofAgent singleton
    config: Optional[ConfigStore] = None,
    tracer: Optional[Tracer] = None,
    result_future: Optional["asyncio.Future[Dict[str, Any]]"] = None,
    history: Optional[List[Tuple[str, str]]] = None,
    system_prompt_override: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """Stream one agent run as SSE bytes.

    ``agent`` is the lifespan-built singleton matching ``kind``.
    ``tracer`` (when provided) makes the run write to
    ``STORAGE_PATH/<flavor>/<date>/<run_id>``; the runner resolves that
    folder relative to ``STORAGE_PATH`` and stuffs it into the result
    payload so the route can persist it.

    ``result_future`` (when provided) is set with a dict carrying
    ``answer`` / ``exit_reason`` / ``loops`` / ``decision``? /
    ``total_cost`` / ``trace_path``? once the agent returns. The route
    handler awaits it after the bus drains to write the assistant
    message inside the request's async DB session.

    ``system_prompt_override`` (when not ``None``) replaces the prompt
    that ``materialize_agent_kwargs`` would have pulled from
    ``prompt.<kind>_agent``. Used by wrapper runners (e.g.
    ``risk_predict_runner``) that drive the same kind=``graph`` engine
    but want a workbench-specific instruction set without registering a
    new agent kind.
    """
    if kind not in ("base", "proof", "graph", "web"):
        raise ValueError(f"unsupported agent kind: {kind!r}")

    loop = asyncio.get_running_loop()
    bus = EventBus(loop=loop)
    capturing = CapturingTracer(tracer) if tracer is not None else None

    # Snapshot the config before the worker starts so a concurrent
    # PATCH cannot half-apply mid-run.
    effective_config = config if config is not None else ConfigStore.defaults_only()
    agent_overrides = effective_config.materialize_agent_kwargs(kind)
    if system_prompt_override is not None:
        agent_overrides["system_prompt"] = system_prompt_override
    preview_chars = effective_config.citation_preview_chars()

    # Multi-turn: prepend prior (q, a) pairs to the user query as
    # context. trace files still record only the *current* (query,
    # answer) — see _compose_query_with_history docstring.
    composed_query = _compose_query_with_history(history or [], query)

    # Web-agent citation pool. Indexed by canonical URL → WebCitation
    # field dict (no sup yet). Sup is assigned at flush time by parsing
    # the model's ``## Sources`` section so the legend matches what the
    # LLM put in the answer; if that section is missing, we fall back
    # to first-seen tool order. Other kinds leave this empty.
    url_pool: Dict[str, Dict[str, Any]] = {}
    pool_order: List[str] = []
    # Tool-call → tool_result alias map: web_fetch's request URL may
    # differ from the envelope's ``final_url`` (HTTP redirect). When
    # the result lands we union the two so the LLM can cite either.
    pending_fetch_url: Optional[str] = None
    # Local-agent (base / proof / graph) citation pool: same shape as
    # the workbench accumulator — each ``read`` envelope contributes
    # its ``units`` deduped by (file_id, page_id), sup is first-seen
    # order. Emitted once before ``final`` so the chat ``citations``
    # frame matches the ``citations → final → done`` contract.
    local_cited_units: List[CitationItem] = []
    local_seen_keys: Set[Tuple[str, str]] = set()

    def wrapped_on_event(event_name: str, data: Dict[str, Any]) -> None:
        nonlocal pending_fetch_url
        # Track the requested URL of the most recent web_fetch tool_call
        # so we can alias it against the envelope's final_url on the
        # matching tool_result. BaseAgent emits tool_call → execute →
        # tool_result strictly serially per loop, so a single-pending
        # slot is enough (no FIFO queue needed).
        if (
            kind == "web"
            and event_name == EventType.TOOL_CALL
            and data.get("name") == "web_fetch"
        ):
            args = data.get("args") or {}
            req_url = args.get("url") if isinstance(args, dict) else None
            if isinstance(req_url, str) and req_url:
                pending_fetch_url = req_url

        # Enrich tool_result frames with is_evidence so the frontend
        # can split inline citation cards from generic explore steps.
        # Drop ``_full_result`` (BaseAgent / ProofAgent pass the raw
        # tool envelope under that key for runner-side consumers); the
        # chat surface only needs the 300-char preview.
        if event_name == EventType.TOOL_RESULT:
            full_result = data.get("_full_result")
            tool_name = data.get("name")
            data = {k: v for k, v in data.items() if k != "_full_result"}
            data["is_evidence"] = _is_evidence(tool_name, None)
            if (
                kind == "web"
                and isinstance(full_result, str)
                and tool_name in _WEB_CITATION_TOOLS
            ):
                request_alias = pending_fetch_url if tool_name == "web_fetch" else None
                _accumulate_web_citations(
                    full_result,
                    tool_name,
                    url_pool,
                    pool_order,
                    preview_chars,
                    request_alias=request_alias,
                )
                if tool_name == "web_fetch":
                    pending_fetch_url = None
            # Local kinds: feed every ``read`` envelope into the same
            # citation accumulator the workbench runners use. Frontend
            # gets a sup-numbered legend regardless of whether the
            # admin-tuned prompt actually emits ``[^k]`` markers.
            if (
                kind in ("base", "proof", "graph")
                and tool_name == "read"
                and isinstance(full_result, str)
            ):
                _accumulate_read_citations(
                    full_result, local_cited_units, local_seen_keys, preview_chars
                )
            # Graph kind: pull a canvas-shaped projection out of the
            # graph_explore envelope and emit it on a side channel. The
            # GraphPage subscribes to GRAPH_SUBGRAPH frames to keep its
            # canvas in sync with what the agent actually discovered;
            # without this passthrough the frontend would have to re-
            # query /graph/expand and might surface a different subgraph
            # (PPR vs. BFS routes diverge slightly).
            if (
                kind == "graph"
                and tool_name == "graph_explore"
                and isinstance(full_result, str)
            ):
                projected = _project_graph_explore(full_result, data.get("loop"))
                if projected is not None:
                    bus.push(EventType.GRAPH_SUBGRAPH, projected)

        # Swallow the agent-internal ``final`` frame for kinds that
        # emit their own canonical ``final`` AFTER ``citations``:
        # web (URL pool flush) and the local kinds (read-units flush).
        # Forwarding the agent's final would put it BEFORE citations
        # and break the ``citations → final → done`` ordering the
        # frontend latches onto.
        if event_name == EventType.FINAL and kind in ("web", "base", "proof", "graph"):
            return

        bus.push(event_name, data)

    def run_in_thread() -> None:
        result_payload: Dict[str, Any] = {}
        try:
            if kind == "proof":
                if not isinstance(agent, ProofAgent):
                    raise TypeError("proof kind requires ProofAgent instance")
                proof_result = agent.run(
                    composed_query,
                    tracer=capturing,
                    on_event=wrapped_on_event,
                    cancel_check=lambda: bus.is_closed,
                    **agent_overrides,
                )
                result_payload = _make_proof_final_payload(proof_result)
            else:
                if not isinstance(agent, BaseAgent):
                    raise TypeError(f"{kind!r} kind requires BaseAgent instance")
                base_result = agent.run(
                    composed_query,
                    tracer=capturing,
                    on_event=wrapped_on_event,
                    cancel_check=lambda: bus.is_closed,
                    **agent_overrides,
                )
                result_payload = _make_base_final_payload(base_result)

            if capturing is not None and capturing.last_run_dir is not None:
                try:
                    result_payload["trace_path"] = relativize_trace_dir(
                        capturing.last_run_dir
                    )
                except ValueError:
                    # tracer.root pointed outside STORAGE_PATH (test
                    # override). Skip rather than store an absolute
                    # path that won't survive a STORAGE_PATH change.
                    logger.warning(
                        "trace dir %s is outside STORAGE_PATH; trace_path omitted",
                        capturing.last_run_dir,
                    )

            # Order is ``citations → final → done`` to match the
            # workbench / RAG runner contract. Always emit ``citations``
            # (even when empty) so the frontend can clear stale Drawer
            # state from a previous run on the same session.
            if kind == "web":
                answer_text = str(result_payload.get("answer") or "")
                citation_items = _resolve_web_citation_legend(
                    answer_text, url_pool, pool_order
                )
                bus.push(EventType.CITATIONS, {"items": citation_items})
                result_payload["citations"] = citation_items
                bus.push(EventType.FINAL, dict(result_payload))
            elif kind in ("base", "proof", "graph"):
                citation_items = [c.to_dict() for c in local_cited_units]
                bus.push(EventType.CITATIONS, {"items": citation_items})
                result_payload["citations"] = citation_items
                bus.push(EventType.FINAL, dict(result_payload))
        except Exception as exc:
            logger.exception("agent runner failed (kind=%s)", kind)
            # Flush whatever citations accumulated so the frontend's
            # CitationDrawer can drop stale state from a previous run.
            # Best-effort; never raises, never blocks the close path.
            try:
                if kind == "web":
                    fallback_items = _resolve_web_citation_legend(
                        "", url_pool, pool_order
                    )
                    bus.push(EventType.CITATIONS, {"items": fallback_items})
                elif kind in ("base", "proof", "graph"):
                    bus.push(
                        EventType.CITATIONS,
                        {"items": [c.to_dict() for c in local_cited_units]},
                    )
            except Exception:
                logger.exception(
                    "agent runner: failed to push citations during error path"
                )
            _schedule_future_exception(loop, result_future, exc)
            bus.close(error=f"{type(exc).__name__}: {exc}", error_type=type(exc).__name__)
            return

        _schedule_future_result(loop, result_future, result_payload)
        bus.close()

    loop.run_in_executor(None, run_in_thread)
    async for chunk in bus.stream():
        yield chunk


# --------------------------------------------------- future scheduling ----

# Helpers to set a future from a worker thread without racing the
# loop-side timeout / cancellation. We schedule the ``done()`` check
# AND the set onto the loop in one callback so they run atomically
# from the future's POV — checking ``done()`` from the worker thread
# is meaningless because the answer can change before our scheduled
# callback fires.

def _accumulate_web_citations(
    full_result: str,
    tool_name: str,
    url_pool: Dict[str, Dict[str, Any]],
    pool_order: List[str],
    preview_chars: int,
    *,
    request_alias: Optional[str] = None,
) -> None:
    """Parse a web_search / web_fetch envelope into the URL pool.

    ``web_search`` returns ``results: [{title, url, snippet, score,
    published_date}]`` (the field is ``results`` in the envelope, not
    ``hits`` — that's just the local variable name in the tool); we
    accept ``hits`` too as backward compat in case the field is
    renamed back. ``web_fetch`` returns ``{url, title, text}`` (the
    ``url`` reflects ``final_url`` after redirects).

    The pool is keyed by canonical URL → field dict (no sup yet); sup
    is assigned at flush time by parsing the answer's ``## Sources``
    section. ``request_alias`` is the URL the agent originally asked
    web_fetch to fetch; if it differs from the envelope's final_url
    after redirect, both are mapped to the same pool entry so the
    model can cite either form.

    Failures to parse the envelope are logged at debug level and
    swallowed. A bogus tool result must not abort the agent run; the
    worst case is one fewer chip in the Drawer.
    """
    try:
        payload = json.loads(full_result)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.debug("web citation parse skipped: %s", exc)
        return
    if not isinstance(payload, dict) or not payload.get("ok"):
        return

    if tool_name == "web_search":
        results = payload.get("results")
        if not isinstance(results, list):
            # Test fixtures sometimes serialize hits under "hits" instead
            # of "results"; accept both so unit tests don't have to
            # mirror the production envelope verbatim. Log when the
            # fallback fires so a real envelope drift in production
            # surfaces instead of being silently absorbed.
            results = payload.get("hits")
            if isinstance(results, list):
                logger.debug(
                    "web_search envelope used 'hits' field; "
                    "production tool emits 'results'"
                )
        if not isinstance(results, list):
            return
        for hit in results:
            if not isinstance(hit, dict):
                continue
            _ingest_web_source(
                url_pool,
                pool_order,
                preview_chars,
                url=hit.get("url"),
                title=hit.get("title") or hit.get("url") or "",
                snippet=hit.get("snippet") or hit.get("content"),
                score=hit.get("score"),
                published_date=hit.get("published_date"),
            )
        return

    # tool_name == "web_fetch"
    final_url = payload.get("url")
    title = payload.get("title") or final_url or ""
    snippet_src = payload.get("text") or ""
    snippet = snippet_src[:preview_chars] if snippet_src else None
    _ingest_web_source(
        url_pool,
        pool_order,
        preview_chars,
        url=final_url,
        title=title,
        snippet=snippet,
        score=None,
        published_date=None,
    )
    # Map the original request URL onto the final URL's entry whenever
    # they differ — covers two cases: (a) request_url was never seen
    # (alias only), (b) request_url was seeded by web_search before
    # web_fetch redirected, in which case we merge metadata into the
    # final entry and drop the duplicate from pool_order so fallback
    # numbering doesn't list the same logical source twice.
    if (
        isinstance(request_alias, str)
        and isinstance(final_url, str)
        and request_alias
        and final_url
        and request_alias != final_url
        and final_url in url_pool
    ):
        final_entry = url_pool[final_url]
        prior = url_pool.get(request_alias)
        if prior is not None and prior is not final_entry:
            # Carry forward fields the search hit had but the fetch
            # envelope didn't (notably score / published_date).
            for key, value in prior.items():
                final_entry.setdefault(key, value)
            if request_alias in pool_order:
                pool_order.remove(request_alias)
        url_pool[request_alias] = final_entry


def _ingest_web_source(
    url_pool: Dict[str, Dict[str, Any]],
    pool_order: List[str],
    preview_chars: int,
    *,
    url: Optional[str],
    title: str,
    snippet: Optional[str],
    score: Optional[float],
    published_date: Optional[str],
) -> None:
    """Insert (or enrich) a URL → WebCitation-fields entry in the pool."""
    if not isinstance(url, str) or not url:
        return
    fields: Dict[str, Any] = {
        "kind": "web",
        "title": title or url,
        "url": url,
    }
    if snippet:
        normalized = snippet.replace("\n", " ").strip()
        if normalized:
            fields["snippet"] = normalized[:preview_chars]
    if isinstance(score, (int, float)):
        fields["score"] = round(float(score), 4)
    if isinstance(published_date, str) and published_date:
        fields["published_date"] = published_date

    existing = url_pool.get(url)
    if existing is None:
        url_pool[url] = fields
        pool_order.append(url)
        return
    # Merge: a later web_fetch envelope can supply title/snippet that
    # web_search left blank, but never downgrade existing fields.
    for key, value in fields.items():
        existing.setdefault(key, value)


# Match a Sources section heading, e.g. ``## Sources`` / ``### Sources``
# / ``# 来源`` / ``Sources:`` (case-insensitive). Stops at the next
# heading-like line so we don't bleed into following sections.
_SOURCES_HEADING = re.compile(
    r"(?im)^\s*(?:#{1,3}\s*)?(?:sources|references|来源|参考资料)\s*:?\s*$"
)
# Per-line legend: ``[^N] anything URL`` anchored to start of line so
# an inline ``[^k]`` mid-paragraph is not misread as a legend row.
_LEGEND_LINE = re.compile(r"^\s*\[\^(\d+)\][^\n]*", re.MULTILINE)
_URL_PATTERN = re.compile(r"https?://[^\s)]+")


def _resolve_web_citation_legend(
    answer: str,
    url_pool: Dict[str, Dict[str, Any]],
    pool_order: List[str],
) -> List[Dict[str, Any]]:
    """Build the canonical ``citations.items`` list for the SSE frame.

    Strategy:
    1. Locate the answer's ``## Sources`` section (heading variants OK).
    2. For each ``[^k] ... <url>`` line, take ``url`` as authoritative
       and pull title/snippet/score/published_date from ``url_pool``
       (the accumulated metadata). ``k`` is the sup the model emitted
       in-line, which is what the frontend will look up by index.
    3. If no Sources section is found, fall back to first-seen tool
       order: every URL in ``pool_order`` becomes ``[^1..N]`` so the
       Drawer at least has a non-empty list.

    Either way the returned items are ordered by ascending sup so the
    frontend's lookup ``citations.find(c => c.sup === k)`` works.
    """
    legend = _parse_sources_legend(answer)
    if legend:
        items: List[Dict[str, Any]] = []
        for sup, url, fallback_title in legend:
            entry = url_pool.get(url)
            if entry is not None:
                item = {"sup": sup, **entry}
            else:
                item = {"sup": sup, "kind": "web", "title": fallback_title or url, "url": url}
            items.append(item)
        items.sort(key=lambda x: x["sup"])
        return items

    # Fallback: no Sources section. Number every accumulated URL.
    items = []
    for idx, url in enumerate(pool_order, start=1):
        entry = url_pool.get(url)
        if entry is None:
            continue
        items.append({"sup": idx, **entry})
    return items


def _parse_sources_legend(answer: str) -> List[Tuple[int, str, str]]:
    """Extract ``[(sup, url, fallback_title)]`` from the answer's Sources section.

    Stops scanning at the next markdown heading so adjacent sections
    (``## Notes`` / ``## Appendix``) can't bleed `[^k]` lines into the
    legend. ``_LEGEND_LINE`` is anchored to the line start so an inline
    `[^k]` mid-paragraph is not misread as a legend row.
    """
    if not answer:
        return []
    head = _SOURCES_HEADING.search(answer)
    if head is None:
        return []
    body_start = head.end()
    next_heading = _NEXT_HEADING.search(answer, body_start)
    body_end = next_heading.start() if next_heading is not None else len(answer)
    body = answer[body_start:body_end]
    out: List[Tuple[int, str, str]] = []
    seen: Set[int] = set()
    for line_match in _LEGEND_LINE.finditer(body):
        try:
            sup = int(line_match.group(1))
        except (TypeError, ValueError):
            continue
        if sup in seen:
            continue
        line_text = line_match.group(0)
        url_match = _URL_PATTERN.search(line_text)
        if url_match is None:
            continue
        url = url_match.group(0).rstrip(".,;)。，；")
        # url_match.start() is already an offset *within* line_text
        # (group(0)), so no further adjustment is needed.
        before_url = line_text[: url_match.start()]
        title = before_url[before_url.find("]") + 1 :].strip().strip("—-:|").strip()
        seen.add(sup)
        out.append((sup, url, title))
    return out


# Find the next markdown heading (one or more leading '#') so the
# Sources section doesn't run into adjacent sections.
_NEXT_HEADING = re.compile(r"\n\s*#{1,6}\s+", re.MULTILINE)


# --------------------------------------------- graph_explore projection ----


def _project_graph_explore(
    full_result: str, loop: Optional[int]
) -> Optional[Dict[str, Any]]:
    """Extract a canvas-shaped projection from a graph_explore envelope.

    Returned shape (one of the three modes plus a no-op for non-ok
    envelopes):

      ``{loop, mode: "neighbors", hops, seed_ids: [...], entity_ids: [...],
         page_refs: [{file_id, page_id}], paths: [...]}``
      ``{loop, mode: "ppr", question, seed_surfaces: [...], page_refs: [...]}``
      ``{loop, mode: "entity_lookup", question, candidate_ids: [...]}``

    All ``*_ids`` are vertex hash_ids the frontend can pass straight to
    /graph/expand or use to highlight existing nodes on the canvas.
    Returns ``None`` for unparseable / errored envelopes — the side
    channel stays silent rather than emit a noisy "no data" frame.
    """
    try:
        payload = json.loads(full_result)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or not payload.get("ok"):
        return None
    mode = payload.get("mode")
    if mode == "neighbors":
        seeds_resolved = payload.get("seeds_resolved") or []
        seed_ids = [s.get("hash_id") for s in seeds_resolved if s.get("hash_id")]
        entity_ids = [
            e.get("hash_id")
            for e in (payload.get("candidate_entities") or [])
            if e.get("hash_id")
        ]
        page_refs = [
            {"file_id": p.get("file_id"), "page_id": p.get("page_id")}
            for p in (payload.get("candidate_pages") or [])
            if p.get("file_id") and p.get("page_id")
        ]
        return {
            "loop": loop,
            "mode": "neighbors",
            "hops": payload.get("hops"),
            "seed_ids": seed_ids,
            "entity_ids": entity_ids,
            "page_refs": page_refs,
        }
    if mode == "ppr":
        seed_surfaces = [
            s.get("surface") for s in (payload.get("seeds") or []) if s.get("surface")
        ]
        # PPR mode envelope today carries seeds + candidate_pages but no
        # entity / passage hash_ids — frontends that want to drive a
        # /graph/expand call (RiskExploreCanvas, GraphPage agent mode)
        # need ids. Surface seed_ids when the algorithm-layer envelope
        # exposes them in future without breaking older callers (the
        # field is optional on the FE side).
        seed_ids = [
            s.get("hash_id") for s in (payload.get("seeds") or []) if s.get("hash_id")
        ]
        page_refs = [
            {"file_id": p.get("file_id"), "page_id": p.get("page_id")}
            for p in (payload.get("candidate_pages") or [])
            if p.get("file_id") and p.get("page_id")
        ]
        return {
            "loop": loop,
            "mode": "ppr",
            "question": payload.get("question"),
            "seed_surfaces": seed_surfaces,
            "seed_ids": seed_ids,
            "page_refs": page_refs,
        }
    if mode == "entity_lookup":
        # entity_lookup envelope is {surface, physical: [{hash_id, ...}]} —
        # the agent passes a `surface` query, not a `question`, and hits land
        # under `physical` (the disambiguator's term for "physical entity").
        candidate_ids = [
            c.get("hash_id")
            for c in (payload.get("physical") or [])
            if c.get("hash_id")
        ]
        return {
            "loop": loop,
            "mode": "entity_lookup",
            "question": payload.get("surface"),
            "candidate_ids": candidate_ids,
        }
    return None


def _schedule_future_result(
    loop: asyncio.AbstractEventLoop,
    future: Optional["asyncio.Future"],
    value: Any,
) -> None:
    if future is None:
        return
    def _set() -> None:
        if not future.done():
            future.set_result(value)
    try:
        loop.call_soon_threadsafe(_set)
    except RuntimeError:
        logger.debug("event loop closed; dropping future result")


def _schedule_future_exception(
    loop: asyncio.AbstractEventLoop,
    future: Optional["asyncio.Future"],
    exc: BaseException,
) -> None:
    if future is None:
        return
    def _set() -> None:
        if not future.done():
            future.set_exception(exc)
    try:
        loop.call_soon_threadsafe(_set)
    except RuntimeError:
        logger.debug("event loop closed; dropping future exception")
