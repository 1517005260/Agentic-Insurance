"""Final-answer LLM call.

The reader receives ``query`` plus the reranked pages serialized as a
labeled block. We instruct the model to (1) reply in the user's language(s)
(2) use only the provided pages, (3) admit ignorance when the answer is not
present.
"""

import logging
from typing import List, Optional, Sequence

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
) -> str:
    """Run the answer LLM and return its plain-text response."""
    cfg = config or RAGConfig()
    client = llm or LLMClient()
    user_msg = _USER_TEMPLATE.format(query=query, pages_block=_format_pages(pages))
    result = client.chat(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
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
