"""Case-insensitive substring search over page Markdown.

The unit of retrieval is the page. The score is a length-weighted match count
across all keywords; per-page snippets are sentences containing any matched
keyword. Snippets only — full text comes from `read_page`.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, TYPE_CHECKING

import tiktoken

from agentic.tools.base import BaseTool
from ingestion.index._sentence import split_sentences
from storage.page_store import PageStore

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class KeywordSearchTool(BaseTool):
    def __init__(self, page_source: Union[str, Path, PageStore]):
        if isinstance(page_source, PageStore):
            self.page_store = page_source
        else:
            self.page_store = PageStore(page_source)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    @property
    def name(self) -> str:
        return "keyword_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "keyword_search",
                "description": (
                    "Search pages by case-insensitive keyword matching. Returns page "
                    "IDs and abbreviated sentence snippets where the keywords appear.\n\n"
                    "Use SHORT, SPECIFIC terms (1-3 words). Each keyword is matched "
                    "independently. Avoid full sentences or questions.\n\n"
                    "Snippets are abbreviated — call read_page on promising IDs to get "
                    "the full Markdown before answering."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "1-3 word terms, e.g. ['Einstein', 'relativity', '1905'].",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top-ranked pages (default: 5, max: 20).",
                            "default": 5,
                        },
                    },
                    "required": ["keywords"],
                },
            },
        }

    def execute(
        self, context: "AgentContext", keywords: List[str], top_k: int = 5
    ) -> Tuple[str, Dict[str, Any]]:
        top_k = min(top_k, 20)

        scored_pages = []
        for global_id in self.page_store.ids():
            page = self.page_store.get(global_id)
            if page is None:
                continue
            text = page.text_markdown
            if not text:
                continue
            text_lower = text.lower()

            matches: List[str] = []
            total_score = 0
            for keyword in keywords:
                kw_lower = keyword.lower()
                count = text_lower.count(kw_lower)
                if count > 0:
                    matches.append(keyword)
                    total_score += count * len(keyword)

            if total_score == 0:
                continue

            sentences = split_sentences(text)
            matched_sentences = [
                s for s in sentences if any(kw.lower() in s.lower() for kw in matches)
            ]

            scored_pages.append(
                {
                    "global_id": global_id,
                    "score": total_score,
                    "matched_sentences": matched_sentences[:5],
                    "keywords_found": matches,
                }
            )

        scored_pages.sort(key=lambda x: x["score"], reverse=True)
        top_pages = scored_pages[:top_k]

        if not top_pages:
            return (
                f"No results found for keywords: {keywords}",
                {"retrieved_tokens": 0, "pages_found": 0},
            )

        result_parts = []
        for item in top_pages:
            if item["matched_sentences"]:
                matched_text = "... " + " ... ".join(item["matched_sentences"]) + " ..."
            else:
                matched_text = "(no exact sentence match)"
            result_parts.append(
                f"Page: {item['global_id']}, Matched keywords: {matched_text}"
            )
        tool_result = "\n\n".join(result_parts)

        all_matched_sentences = []
        for item in top_pages:
            all_matched_sentences.extend(item["matched_sentences"])
        retrieved_tokens = (
            len(self.tokenizer.encode("\n".join(all_matched_sentences)))
            if all_matched_sentences
            else 0
        )

        context.add_retrieval_log(
            tool_name="keyword_search",
            tokens=retrieved_tokens,
            metadata={
                "keywords": keywords,
                "pages_found": len(top_pages),
                "page_ids": [p["global_id"] for p in top_pages],
            },
        )

        return tool_result, {
            "retrieved_tokens": retrieved_tokens,
            "pages_found": len(top_pages),
        }
