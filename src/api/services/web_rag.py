"""Single-call web RAG.

A simpler cousin of :class:`rag.pipeline.RAGPipeline`: where the
local pipeline runs four retrieval channels + RRF + rerank + LLM,
this one substitutes the entire retrieval stack with a single
Tavily call. The LLM still does the answering and citing.

Two surfaces:

* :func:`run_summarized` — synchronous, returns a complete dict.
  Used by the regulation-search workbench (no SSE; <5s response).
* :func:`stream_chat` — async generator yielding event tuples for
  SSE conversion. Used by the chat-web-rag mode.

Both share the same retrieve/prompt-build/answer recipe; only the
output transport differs.
"""
import json
import logging
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


# ---------------------------------------------------------------- assembly


def _build_messages(
    *, system_prompt: str, query: str, sources: Sequence[WebSource]
) -> List[Dict[str, str]]:
    """Compose chat messages: system + 'sources block + question' user turn.

    Sources are rendered as a numbered list so the LLM has only one
    way to cite — `[^k]` referencing `[Source k]`. Putting the
    numbering in the prompt removes the LLM's freedom to invent its
    own scheme; it also matches the local-RAG inline-cite contract.
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
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


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

    Used by the regulation-search workbench. Caller is expected to be
    in an event-loop-friendly context (route handler offloads to
    ``run_in_threadpool`` since the LLM call is blocking).
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
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Generator yielding (event_name, data) tuples.

    Sequence (always in this order):

    1. ``status`` ``{phase: "searching"}``
    2. ``retrieval`` ``{n_results, sources: [...]}``  (after Tavily returns)
    3. ``status`` ``{phase: "answering"}``
    4. ``token``  one per LLM streaming chunk
    5. ``citations`` ``{items: [...]}``  (extracted sources used)
    6. ``final``  ``{answer_chars, n_results, ...}``

    Yielding tuples (not SSE bytes) keeps the runner responsible for
    serialization, so the same generator can drive different
    transports (SSE today, websocket later).
    """
    yield ("status", {"phase": "searching"})
    sources = _retrieve(
        tavily=tavily,
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )
    yield (
        "retrieval",
        {
            "channel": "web",
            "n_results": len(sources),
            "sources": [s.to_public_dict() for s in sources],
        },
    )

    yield ("status", {"phase": "answering"})
    messages = _build_messages(
        system_prompt=system_prompt or _DEFAULT_SUMMARIZER_PROMPT,
        query=query,
        sources=sources,
    )

    answer_parts: List[str] = []
    usage: Dict[str, Any] = {}
    cost: float = 0.0
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
        if "usage" in chunk:
            usage = chunk["usage"]
        if "cost" in chunk:
            cost = chunk["cost"]

    answer = "".join(answer_parts).strip()
    cited = _extract_cited_sources(answer, sources)
    yield (
        "citations",
        {
            "items": [c.to_public_dict() for c in cited],
        },
    )
    yield (
        "final",
        {
            "answer_chars": len(answer),
            "n_results": len(sources),
            "n_cited": len(cited),
            "usage": usage,
            "cost": cost,
            "search_query": query,
        },
    )
    # Sentinel so the runner can pick up the assembled answer for
    # the result_future without re-joining the parts list itself.
    yield ("__assembled__", {"answer": answer, "sources": [s.to_public_dict() for s in sources], "cited": [c.to_public_dict() for c in cited]})


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
