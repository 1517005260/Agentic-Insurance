"""Final-answer LLM call.

The reader receives ``query`` plus the reranked pages serialized as a
labeled block. We instruct the model to (1) reply in the user's language(s)
(2) use only the provided pages, (3) admit ignorance when the answer is not
present.

The function has a streaming mode (``on_event`` + ``stream=True``) that
fires per-token deltas via the supplied callback. The non-streaming
path returns the full answer string and is what the experiment scripts
keep using; the API runner takes the streaming path.

``system_prompt`` and ``citation_legend`` are optional injection points
the web layer uses to swap in a stricter business prompt and append
the ``[^k]`` legend after the question. Algorithm callers pass neither.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from config import RAGConfig
from model_client import LLMClient
from rag.rerank import RerankedPage


logger = logging.getLogger(__name__)


_SYSTEM = """\
You are a document QA assistant. Answer the user's question based ONLY on \
the provided page content.

LANGUAGE MATCHING (critical):
- Reply in the SAME language(s) as the user's question.
- Chinese question -> Chinese answer. English -> English. Mixed -> mixed.

Rules:
- Use ONLY information explicitly present in the pages. Do not fabricate.
- If the answer cannot be derived from the pages, say so honestly in the \
user's language (e.g. "根据提供的内容，无法回答" or "Based on the provided \
content, I cannot answer this").
- Be concise; quote specific values / phrases when they appear in the pages.\
"""

_USER_TEMPLATE = """\
Question: {query}

Pages:
{pages_block}

Answer:"""

# When the runner injects a ``[^k]`` legend, it lands between the
# pages block and the "Answer:" cue so the model sees both the page
# content and the labels it must use.
_USER_TEMPLATE_WITH_LEGEND = """\
Question: {query}

Pages:
{pages_block}

{legend}

Answer:"""


def _format_pages(pages: Sequence[RerankedPage]) -> str:
    parts: List[str] = []
    for r in pages:
        p = r.page
        parts.append(f"----- Page {p.file_id}#{p.page_id} -----\n{p.text_markdown}\n")
    return "\n".join(parts) if parts else "(no pages found)"


def answer(
    query: str,
    pages: Sequence[RerankedPage],
    *,
    config: Optional[RAGConfig] = None,
    llm: Optional[LLMClient] = None,
    system_prompt: Optional[str] = None,
    citation_legend: Optional[str] = None,
    pages_block_override: Optional[str] = None,
    stream: bool = False,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    history: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Run the answer LLM and return its plain-text response.

    Default behavior (no kwargs beyond config / llm) is identical to
    the original blocking path — what the experiment scripts call.

    When ``stream=True``, the LLM is invoked with ``chat_stream`` and
    each delta is forwarded to ``on_event("token", {"delta": ...})``;
    the function still returns the assembled full text at the end.
    ``stream=True`` without ``on_event`` is allowed (deltas just go
    nowhere) — useful for smoke testing.

    ``system_prompt`` overrides the default ``_SYSTEM``.
    ``citation_legend`` (already-rendered text from
    ``CitationBuilder``) is appended to the user message between the
    pages block and the "Answer:" cue.

    ``history`` (optional) is a chronological list of
    ``(prior_user_query, prior_assistant_answer)`` pairs that gets
    spliced between the system prompt and the current user message.
    The current turn's pages-block + legend stays anchored to the
    *current* query — sup numbering is scoped to one turn, so prior
    turns' citations are not carried over. Default ``None`` keeps the
    single-turn behaviour for experiment scripts.
    """
    cfg = config or RAGConfig()
    client = llm or LLMClient()

    sys_prompt = system_prompt if system_prompt is not None else _SYSTEM
    pages_block = (
        pages_block_override if pages_block_override is not None else _format_pages(pages)
    )
    if citation_legend:
        user_msg = _USER_TEMPLATE_WITH_LEGEND.format(
            query=query, pages_block=pages_block, legend=citation_legend
        )
    else:
        user_msg = _USER_TEMPLATE.format(query=query, pages_block=pages_block)

    messages: List[Dict[str, Any]] = [{"role": "system", "content": sys_prompt}]
    if history:
        for prev_q, prev_a in history:
            messages.append({"role": "user", "content": prev_q})
            messages.append({"role": "assistant", "content": prev_a})
    messages.append({"role": "user", "content": user_msg})

    if stream:
        return _answer_stream(client, messages, cfg, on_event, query, cancel_check)

    result = client.chat(
        messages=messages,
        temperature=0.0,
        max_tokens=cfg.answer_max_tokens,
    )
    content = (result["message"].get("content") or "").strip()
    finish_reason = ""
    try:
        finish_reason = (
            result.get("raw_response", {})
            .get("choices", [{}])[0]
            .get("finish_reason", "")
        )
    except Exception:
        pass
    if not content or finish_reason == "length":
        # ``length`` means we hit ``answer_max_tokens`` — for reasoning
        # models the visible answer can be empty or truncated mid-sentence
        # while reasoning_tokens consumed the entire budget. Surface a
        # warning so this isn't silently misinterpreted as a real answer.
        logger.warning(
            "answer LLM truncated or empty "
            "(finish_reason=%r, output_tokens=%d, content_chars=%d, query=%r)",
            finish_reason,
            result.get("output_tokens", 0),
            len(content),
            query[:120],
        )
    return content


def _answer_stream(
    client: LLMClient,
    messages: List[Dict[str, Any]],
    cfg: RAGConfig,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]],
    query: str,
    cancel_check: Optional[Callable[[], bool]],
) -> str:
    """Stream the answer tokens; assemble + return the full text.

    ``cancel_check`` is polled before forwarding each delta. When it
    returns True (typically because the SSE consumer disconnected and
    the runner-side ``EventBus`` flipped its closed flag), the
    generator is closed early so the underlying HTTPS keep-alive
    connection drops, the relay stops billing tokens nobody will read,
    and the partial text is returned.
    """
    chunks: List[str] = []
    finish_reason: str = ""
    stream_iter = client.chat_stream(
        messages=messages,
        temperature=0.0,
        max_tokens=cfg.answer_max_tokens,
    )
    try:
        for frame in stream_iter:
            if cancel_check is not None and cancel_check():
                # Closing the generator triggers GeneratorExit inside
                # chat_stream → exits the ``with response`` block →
                # the underlying urllib3 connection is released, so the
                # relay sees TCP close and aborts generation.
                stream_iter.close()
                break
            delta = frame.get("delta")
            if delta:
                chunks.append(delta)
                if on_event is not None:
                    try:
                        on_event("token", {"delta": delta})
                    except Exception:
                        logger.exception("token on_event callback failed")
            if frame.get("finish_reason"):
                finish_reason = frame["finish_reason"]
    finally:
        # Idempotent — close on any exit path so a mid-stream exception
        # also drops the upstream socket.
        stream_iter.close()
    full = "".join(chunks).strip()
    if not full or finish_reason == "length":
        logger.warning(
            "answer stream truncated or empty (finish_reason=%r, query=%r)",
            finish_reason,
            query[:120],
        )
    return full
