"""Single-call web RAG.

A simpler cousin of :class:`rag.pipeline.RAGPipeline`: where the
local pipeline runs four retrieval channels + RRF + rerank + LLM,
this one substitutes the entire retrieval stack with a single
Tavily call. The LLM still does the answering and citing.

Two surfaces:

* :func:`stream_chat` — async generator yielding event tuples for
  SSE conversion. Powers the chat web mode.
* :func:`run_summarized` — synchronous, returns a complete dict.
  Currently has no route caller (the regulation workbench that
  consumed it has been retired); retained for any future
  synchronous workbench that wants the same recipe.

Both share the same retrieve/prompt-build/answer recipe; only the
output transport differs.

Multi-turn note: when ``history`` is non-empty, ``stream_chat`` runs
a cheap LLM rewrite first to convert the (possibly elliptical)
follow-up question into a self-contained Tavily search query. Without
this, "详细讲下病毒" after a turn about "汉他病毒邮轮疫情" searches
the open web for generic biological viruses. The rewrite passes the
**full** prior assistant answer (per user requirement — no
summarization) so coreference resolution sees the whole context.
"""
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from model_client import LLMClient
from model_client.web_search import SearchResult, TavilyClient


logger = logging.getLogger(__name__)


_DEFAULT_SUMMARIZER_PROMPT = (
    "You are an insurance / financial-regulation research assistant. "
    "Answer the user's question using ONLY the numbered web sources "
    "provided below. Every factual claim must carry a citation marker "
    "[^k] referring to the source by its number. If the sources do "
    "not cover the question, say so explicitly — do not invent or "
    "extrapolate. Quote regulation names, article numbers, dates, and "
    "monetary figures verbatim. End your answer with a `## Sources` "
    "section listing each cited source as `[^k] <title> — <url>`."
)


# Coreference / ellipsis resolver for multi-turn follow-ups.
#
# History budget: the full prior assistant body is included (per
# project requirement: no summarization), but we only keep the **last
# 3 turns** to bound prompt cost. Older turns rarely carry the
# coreference target the user is implicitly relying on; if they do,
# the user can re-state. With 5000-token answers each, 3 turns ≈ 15k
# tokens of context — comfortable for any modern model and well below
# 128k limits.
#
# The prompt nudges the LLM toward "concise, effective, faithful to
# user intent" — search engines reward keyword density, not eloquence,
# and adding extra hypothesised entities risks query drift away from
# what the user actually meant.
_REWRITE_HISTORY_TURNS = 3

_REWRITE_SEARCH_PROMPT = """\
You are a search-query rewriter. The user is in a multi-turn conversation. \
Their latest question may use pronouns ("它"), ellipsis ("详细讲下"), or \
implicit references that only make sense given the prior turns.

Your job: rewrite the latest question into ONE self-contained search query \
suitable for a web search engine.

Hard requirements:
- Output must be **concise, effective, and faithful to the user's intent** — \
no padding, no extra commentary, no speculative entities the user did not \
imply. Prefer keywords over full sentences.
- Keep the user's language (Chinese stays Chinese, English stays English; \
mixed input → mixed output).
- Preserve named entities, numbers, dates, and proper nouns verbatim.
- Inject only the disambiguating context the user implicitly relies on — \
e.g. if the prior assistant turn was about "汉他病毒邮轮疫情" and the user \
now asks "详细讲下病毒", rewrite as "汉他病毒 邮轮疫情 详细介绍".
- If the latest question is already self-contained, return it **unchanged**.

Output format (strict): a single line of plain text with no quotes, no \
fences, no commentary, ≤ 30 words.

=== Conversation history (oldest → newest, last {n_turns} turns) ===
{history_block}
=== Latest question ===
{query}

Standalone search query:"""


# How many chars of the rewritten query to ship in the SSE rewrite
# event. The frontend may show "搜索关键词: ..." as a chip; truncate
# defensively in case the LLM ignores the one-line instruction. We
# DO NOT use this as a search-query cap any more — over-long output
# is treated as a rewrite failure (see ``_rewrite_search_query``).
_REWRITE_PREVIEW_CHARS = 400

# Hard cap above which the rewrite is treated as a failure (LLM
# ignored the strict instruction; sending 400 chars of free-form
# prose to Tavily would drift the search away from the original
# intent and could even leak prompt-injection artefacts).
_REWRITE_MAX_QUERY_CHARS = 240

# Cap on the rewrite output. One short line; guards against the LLM
# returning an essay. 200 tokens ≈ 600 chars in CJK is plenty.
_REWRITE_MAX_TOKENS = 200


@dataclass
class WebSource:
    """Numbered source rendered for the LLM and surfaced in the response."""

    sup: int  # 1-based marker, matches [^k] in answer
    title: str
    url: str
    snippet: str
    score: float
    published_date: Optional[str]

    def to_public_dict(self) -> Dict[str, Any]:
        # ``kind`` discriminates web sources from local-RAG citations
        # ({sup,file_id,page_id,...}) when both can land in the same
        # ``citations`` list on a chat message metadata blob; the
        # frontend reads ``kind`` to pick the right drawer (URL preview
        # vs PDF page viewer).
        return {
            "kind": "web",
            "sup": self.sup,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
            "published_date": self.published_date,
        }


@dataclass
class WebRagResult:
    answer: str
    sources: List[WebSource]
    n_results: int
    search_query: str
    used_include_domains: Optional[List[str]]


# ---------------------------------------------------------------- rewrite


def _format_rewrite_history(history: List[Tuple[str, str]]) -> str:
    """Render the most recent ``_REWRITE_HISTORY_TURNS`` (user, assistant)
    pairs as plain text for the rewriter.

    Per project requirement: assistant answers are passed **whole**, no
    summarization. This is the only way "详细讲下病毒" can be expanded
    into "汉他病毒 邮轮疫情 详细介绍" — the entity "汉他病毒" only
    exists inside the prior assistant body.

    Window: only the **last 3 turns** are kept. Older turns rarely
    carry the coreference target the user implicitly relies on, and
    cap the rewrite-prompt cost at ~15 k tokens (3 turns × ~5 k chars
    of typical assistant output).

    Pairs are rendered oldest → newest. Each turn shows ``T<i> 用户:``
    and ``T<i> 助手:`` blocks separated by a blank line, mirroring how
    chat transcripts are usually exchanged so the LLM has a clear
    coreference frame.
    """
    if not history:
        return "(无历史)"
    # Slice tail for chronological "last N turns" (history is already
    # chronological per ``load_recent_turns`` contract).
    tail = history[-_REWRITE_HISTORY_TURNS:]
    lines: List[str] = []
    for i, (q, a) in enumerate(tail, start=1):
        q_clean = (q or "").strip()
        a_clean = (a or "").strip()
        lines.append(f"T{i} 用户: {q_clean}")
        if a_clean:
            lines.append(f"T{i} 助手: {a_clean}")
        lines.append("")  # blank line between turns
    return "\n".join(lines).rstrip()


def _rewrite_search_query(
    *,
    llm: LLMClient,
    query: str,
    history: List[Tuple[str, str]],
) -> Tuple[str, Optional[str]]:
    """Return (rewritten_query, error_or_none).

    Best-effort: any failure (LLM raise, empty output, suspiciously
    long fence-wrapped junk) falls back to ``query`` unchanged. Caller
    should not branch on the error string — it's purely for the SSE
    event so the frontend can hint "rewrite degraded, using original".
    """
    history_block = _format_rewrite_history(history)
    prompt = _REWRITE_SEARCH_PROMPT.format(
        history_block=history_block,
        query=query,
        n_turns=_REWRITE_HISTORY_TURNS,
    )
    # We rely on LLMClient's built-in (10, 120) connect/read timeout —
    # adding a per-call timeout here would require shoving the call
    # behind a thread + future, and the rewrite is already gated by
    # ``max_tokens=_REWRITE_MAX_TOKENS`` so the worst case is one
    # short LLM round-trip. The caller emits an SSE rewrite event with
    # ``elapsed_ms``; long delays surface in the trace, not as silent
    # tail-latency hidden under "answering".
    try:
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=_REWRITE_MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("web_rag rewrite call failed: %s", exc)
        return query, f"{type(exc).__name__}: {exc}"

    raw = (response.get("message") or {}).get("content") or ""
    raw = raw.strip()
    if not raw:
        return query, "empty rewrite"
    # LLM occasionally still wraps in fences or quotes despite the
    # instruction — strip a single leading/trailing layer.
    if raw.startswith("```"):
        # take the first non-fence line
        lines = [ln for ln in raw.splitlines() if ln and not ln.startswith("```")]
        raw = lines[0] if lines else query
    raw = raw.strip().strip('"').strip("'").strip()
    if not raw:
        return query, "empty after strip"
    # one-line guard: if the model wrote prose, keep first line only.
    raw = raw.splitlines()[0].strip()
    if not raw:
        return query, "empty after one-line"
    # Over-long output → treat as failure and fall back to original.
    # Earlier we silently truncated to 400 chars, but that risks
    # sending a chunk of explanatory prose (or worse, prompt-injection
    # leakage) to the search engine. Failing soft preserves the
    # user's intent at the cost of one missed coreference resolution.
    if len(raw) > _REWRITE_MAX_QUERY_CHARS:
        return query, f"rewrite too long ({len(raw)}>{_REWRITE_MAX_QUERY_CHARS})"
    return raw, None


# ---------------------------------------------------------------- assembly


def _build_messages(
    *,
    system_prompt: str,
    query: str,
    sources: Sequence[WebSource],
    history: Optional[List[Tuple[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Compose chat messages: system + (optional history) + sources/question.

    Sources are rendered as a numbered list so the LLM has only one
    way to cite — `[^k]` referencing `[Source k]`. Putting the
    numbering in the prompt removes the LLM's freedom to invent its
    own scheme; it also matches the local-RAG inline-cite contract.

    ``history`` (chronological prior (user, assistant) pairs) is
    spliced between the system prompt and the current user turn so
    the model can answer follow-up questions in context. Sup numbering
    is scoped to one turn — prior turns' citations don't carry over.
    """
    if not sources:
        sources_block = "(no sources retrieved — answer with abstain)"
    else:
        lines = []
        for s in sources:
            head = f"[Source {s.sup}] {s.title}".rstrip()
            url_line = f"URL: {s.url}"
            date_line = f"Published: {s.published_date}" if s.published_date else ""
            body = s.snippet or "(no snippet)"
            block_lines = [head, url_line]
            if date_line:
                block_lines.append(date_line)
            block_lines.append("")
            block_lines.append(body)
            lines.append("\n".join(block_lines))
        sources_block = "\n\n---\n\n".join(lines)

    user_content = (
        f"Question: {query}\n\n"
        f"Sources:\n\n{sources_block}\n\n"
        "Answer in the same language as the question. "
        "Cite every fact with [^k]."
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    if history:
        for prev_q, prev_a in history:
            messages.append({"role": "user", "content": prev_q})
            messages.append({"role": "assistant", "content": prev_a})
    messages.append({"role": "user", "content": user_content})
    return messages


def _retrieve(
    *,
    tavily: TavilyClient,
    query: str,
    max_results: int,
    search_depth: str,
    include_domains: Optional[Sequence[str]],
    exclude_domains: Optional[Sequence[str]],
    days: Optional[int] = None,
) -> List[WebSource]:
    """Tavily search → WebSource list. Empty list on Tavily unavailable
    so the caller can still emit a graceful abstain answer."""
    if not tavily.available():
        logger.warning("tavily client unavailable; returning no sources")
        return []
    # ``days`` is honored by Tavily only when ``topic='news'``. Pass
    # it through opportunistically; Tavily rejects gracefully when
    # not applicable, which the SearchResult adapter swallows as
    # "no results".
    extra: Dict[str, Any] = {}
    if days is not None:
        extra["days"] = int(days)
        extra["topic"] = "news"
    try:
        raw: List[SearchResult] = tavily.search_results(
            query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=list(include_domains) if include_domains else None,
            exclude_domains=list(exclude_domains) if exclude_domains else None,
            **extra,
        )
    except Exception:
        logger.exception("tavily search failed; returning no sources")
        return []
    return [
        WebSource(
            sup=i + 1,
            title=r.title,
            url=r.url,
            snippet=r.content,
            score=round(float(r.score), 4),
            published_date=r.published_date,
        )
        for i, r in enumerate(raw)
    ]


# ---------------------------------------------------------------- entry points


def run_summarized(
    *,
    llm: LLMClient,
    tavily: TavilyClient,
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: Optional[Sequence[str]] = None,
    exclude_domains: Optional[Sequence[str]] = None,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    days: Optional[int] = None,
) -> WebRagResult:
    """Synchronous web-RAG: retrieve + single LLM call + return.

    No route currently consumes this entrypoint (the regulation
    workbench that did has been retired); kept available for future
    synchronous callers and unit testing of the recipe. Caller is
    expected to wrap this in ``run_in_threadpool`` because the LLM
    call blocks.
    """
    sources = _retrieve(
        tavily=tavily,
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        days=days,
    )
    messages = _build_messages(
        system_prompt=system_prompt or _DEFAULT_SUMMARIZER_PROMPT,
        query=query,
        sources=sources,
    )
    response = llm.chat(messages=messages, max_tokens=max_tokens)
    message = response.get("message") or {}
    answer = (message.get("content") or "").strip()
    return WebRagResult(
        answer=answer,
        sources=sources,
        n_results=len(sources),
        search_query=query,
        used_include_domains=list(include_domains) if include_domains else None,
    )


def stream_chat(
    *,
    llm: LLMClient,
    tavily: TavilyClient,
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: Optional[Sequence[str]] = None,
    exclude_domains: Optional[Sequence[str]] = None,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    history: Optional[List[Tuple[str, str]]] = None,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Generator yielding (event_name, data) tuples.

    Sequence (always in this order):

    0. ``status`` ``{phase: "rewriting"}`` — *only* when ``history`` is
       non-empty; turn 1 skips straight to ``searching``.
    0'. ``rewrite`` ``{original, rewritten, elapsed_ms, error?}`` —
       paired with the rewriting status.
    1. ``status`` ``{phase: "searching"}``
    2. ``retrieval`` ``{n_results, sources: [...]}``  (after Tavily returns)
    3. ``status`` ``{phase: "answering"}``
    4. ``token``  one per LLM streaming chunk
    5. ``citations`` ``{items: [...]}``  (extracted sources used)
    6. ``final``  ``{answer_chars, n_results, search_query,
                     timings_ms: {rewrite, retrieve, answer, total}, ...}``

    Yielding tuples (not SSE bytes) keeps the runner responsible for
    serialization, so the same generator can drive different
    transports (SSE today, websocket later).
    """
    t_total0 = time.perf_counter()
    rewrite_ms = 0
    retrieve_ms = 0
    answer_ms = 0
    rewrite_error: Optional[str] = None

    # ---- (0) optional rewrite step (multi-turn coreference) ----
    search_query = query
    if history:
        yield ("status", {"phase": "rewriting"})
        t_rewrite0 = time.perf_counter()
        search_query, rewrite_error = _rewrite_search_query(
            llm=llm, query=query, history=history
        )
        rewrite_ms = int((time.perf_counter() - t_rewrite0) * 1000)
        yield (
            "rewrite",
            {
                "original": query,
                "rewritten": search_query,
                "elapsed_ms": rewrite_ms,
                "error": rewrite_error,
            },
        )

    # ---- (1) retrieve ----
    yield ("status", {"phase": "searching"})
    t_retr0 = time.perf_counter()
    sources = _retrieve(
        tavily=tavily,
        query=search_query,
        max_results=max_results,
        search_depth=search_depth,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )
    retrieve_ms = int((time.perf_counter() - t_retr0) * 1000)
    yield (
        "retrieval",
        {
            "channel": "web",
            "n_results": len(sources),
            "search_query": search_query,
            "elapsed_ms": retrieve_ms,
            "sources": [s.to_public_dict() for s in sources],
        },
    )

    # ---- (2) answer ----
    yield ("status", {"phase": "answering"})
    messages = _build_messages(
        system_prompt=system_prompt or _DEFAULT_SUMMARIZER_PROMPT,
        query=query,  # answer phase still sees the user's original wording
        sources=sources,
        history=history,
    )

    answer_parts: List[str] = []
    usage: Dict[str, Any] = {}
    cost: float = 0.0
    finish_reason: str = ""
    t_ans0 = time.perf_counter()
    for chunk in llm.chat_stream(messages=messages, max_tokens=max_tokens):
        if cancel_check is not None and cancel_check():
            # Closed bus → drop the remainder. The LLM client's session
            # close happens when the stream iterator is GC'd, which
            # happens after this function exits.
            break
        delta = chunk.get("delta")
        if delta:
            answer_parts.append(delta)
            yield ("token", {"delta": delta})
        if chunk.get("finish_reason"):
            finish_reason = chunk["finish_reason"]
        if "usage" in chunk:
            usage = chunk["usage"]
        if "cost" in chunk:
            cost = chunk["cost"]
    answer_ms = int((time.perf_counter() - t_ans0) * 1000)

    answer = "".join(answer_parts).strip()
    # Fallback for "stream closed cleanly but emitted no content"
    # (relays that ship hidden reasoning frames only, then a bare
    # finish_reason) — the user gets ``answer=""`` and 2+ minutes
    # wasted. Retry once via non-streaming chat() which returns the
    # full message body in a single response. Surface the recovered
    # text as one synthesized token frame so the UI updates.
    if not answer and (cancel_check is None or not cancel_check()):
        logger.warning(
            "web_rag stream finished with empty answer "
            "(finish_reason=%r); falling back to non-streaming chat()",
            finish_reason,
        )
        try:
            resp = llm.chat(messages=messages, max_tokens=max_tokens)
            fallback = ((resp.get("message") or {}).get("content") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.exception("web_rag fallback chat() raised: %s", exc)
            fallback = ""
        if fallback:
            answer = fallback
            yield ("token", {"delta": fallback})
            # Roll usage/cost forward — the original stream may have
            # reported zero (or never reported); the fallback's usage
            # is the authoritative one for what the user actually got.
            if isinstance(resp.get("input_tokens"), int):
                usage = {
                    "input_tokens": resp.get("input_tokens", 0),
                    "cached_tokens": resp.get("cached_tokens", 0),
                    "output_tokens": resp.get("output_tokens", 0),
                }
            if "cost" in resp:
                cost = float(resp.get("cost") or 0.0)
    cited = _extract_cited_sources(answer, sources)
    yield (
        "citations",
        {
            "items": [c.to_public_dict() for c in cited],
        },
    )
    total_ms = int((time.perf_counter() - t_total0) * 1000)
    yield (
        "final",
        {
            # ``answer`` carries the full assembled body so the frontend
            # can recover when token frames were dropped (network jitter
            # or hidden-reasoning relays). ``answer_chars`` alone would
            # leave the UI without the text to render.
            "answer": answer,
            "answer_chars": len(answer),
            "n_results": len(sources),
            "n_cited": len(cited),
            "usage": usage,
            "cost": cost,
            "search_query": search_query,
            "original_query": query,
            "rewrite_error": rewrite_error,
            "finish_reason": finish_reason or None,
            "timings_ms": {
                "rewrite": rewrite_ms,
                "retrieve": retrieve_ms,
                "answer": answer_ms,
                "total": total_ms,
            },
        },
    )
    # Sentinel so the runner can pick up the assembled answer for
    # the result_future without re-joining the parts list itself.
    yield (
        "__assembled__",
        {
            "answer": answer,
            "sources": [s.to_public_dict() for s in sources],
            "cited": [c.to_public_dict() for c in cited],
            "search_query": search_query,
            "original_query": query,
            "rewrite_error": rewrite_error,
            "timings_ms": {
                "rewrite": rewrite_ms,
                "retrieve": retrieve_ms,
                "answer": answer_ms,
                "total": total_ms,
            },
        },
    )


# ---------------------------------------------------------------- citation extract


def _extract_cited_sources(
    answer: str, sources: Sequence[WebSource]
) -> List[WebSource]:
    """Pick up [^k] markers actually present in the assembled answer."""
    if not answer or not sources:
        return []
    seen: List[WebSource] = []
    used: set[int] = set()
    by_sup = {s.sup: s for s in sources}
    # Match [^1], [^12], etc.
    import re
    for m in re.finditer(r"\[\^(\d+)\]", answer):
        try:
            k = int(m.group(1))
        except ValueError:
            continue
        if k in used:
            continue
        src = by_sup.get(k)
        if src is None:
            continue
        used.add(k)
        seen.append(src)
    return seen
