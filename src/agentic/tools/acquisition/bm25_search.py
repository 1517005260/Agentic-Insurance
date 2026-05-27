"""Lexical retrieval over the global tantivy BM25 index.

A single sub-query — the agent decides how to phrase. We deliberately do
NOT re-run the original / rewrite / HyDE fan-out that ``rag/channels/bm25``
does for the standalone-RAG pipeline; in the agentic loop the LLM is the
planner and is expected to issue a follow-up query if the first one is
weak. Rolling fan-out into a single tool call would conflate two LLM
turns and bury the cost.

`file_ids` / `page_range` follow the cross-tool scope contract from
``_common.parse_scope`` — both optional, intersected when both supplied,
empty list treated as "no filter".
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import tantivy
from config.shared import shared_tiktoken_encoder

from agentic.tools.acquisition._common import (
    err,
    keyword_snippet,
    ok,
    parse_scope,
)
from agentic.tools.base import BaseTool
from config.settings import bm25_root
from storage.inventory_store import InventoryStore
from storage.page_store import PageStore

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


logger = logging.getLogger(__name__)


# Tantivy QueryParser specials. We strip them rather than escape because
# the agent's queries are natural-language phrases, not boolean
# expressions; honoring the special syntax would surprise the model more
# often than help it.
_TANTIVY_SPECIAL = '+-!(){}[]^"~*?:\\/'


def _sanitize_query(text: str) -> str:
    return "".join(" " if ch in _TANTIVY_SPECIAL else ch for ch in text or "")


class Bm25SearchTool(BaseTool):
    def __init__(
        self,
        page_store: Optional[PageStore] = None,
        index_path: Optional[Path] = None,
        inventory: Optional[InventoryStore] = None,
    ):
        self.page_store = page_store
        self.inventory = inventory
        self.index_path = Path(index_path) if index_path else (bm25_root() / "index")
        self._index: Optional[tantivy.Index] = None
        self._tokenizer = shared_tiktoken_encoder("gpt-4o")

    @property
    def name(self) -> str:
        return "bm25_search"

    @property
    def index(self) -> Optional[tantivy.Index]:
        if self._index is None and self.index_path.is_dir():
            try:
                self._index = tantivy.Index.open(str(self.index_path))
            except Exception as exc:
                logger.warning("bm25_search: failed to open tantivy index: %s", exc)
        return self._index

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "bm25_search",
                "description": (
                    "Lexical (BM25) retrieval over page Markdown. Strongest "
                    "for exact terms, numbers, codes, abbreviations, proper "
                    "nouns. Returns up to `top_k` page hits with abbreviated "
                    "snippets — `read` the page before quoting. Scope filters "
                    "(`file_ids`, `page_range`, `section_ids`) intersect."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Free-text query; punctuation stripped.",
                        },
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional file id allow-list.",
                        },
                        "page_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "Optional [start, end] inclusive 1-based page-number filter."
                            ),
                        },
                        "section_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of section ids "
                                "(e.g. '<file_id>:sec_003') from `toc`. "
                                "A page must lie inside at least one to qualify."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max hits to return; default 10, max 50.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        query: str,
        file_ids: Optional[List[str]] = None,
        page_range: Optional[List[int]] = None,
        section_ids: Optional[List[str]] = None,
        top_k: int = 10,
    ):
        if not query or not str(query).strip():
            return err(
                "invalid_argument",
                "`query` must be a non-empty string.",
                remediation="Pass `query` as a non-empty free-text string; punctuation is stripped automatically.",
                valid_example={"query": "AFYP rebate"},
            ), {"error": "invalid_argument"}
        scope, scope_err = parse_scope(
            file_ids, page_range, section_ids, inventory=self.inventory
        )
        if scope_err is not None:
            return err(
                "invalid_argument",
                scope_err,
                remediation="Fix the scope arguments per the message: file_ids must come from list_files; page_range must be [start, end]; section_ids must come from toc.",
                valid_example={"file_ids": ["<file_id>"], "page_range": [1, 50]},
            ), {"error": "invalid_argument"}

        try:
            top_k_int = int(top_k)
        except (TypeError, ValueError):
            return err(
                "invalid_argument",
                "`top_k` must be an integer.",
                remediation="Pass `top_k` as a positive integer (default 10, max 50).",
                valid_example={"top_k": 10},
            ), {"error": "invalid_argument"}
        if top_k_int < 1:
            return err(
                "invalid_argument",
                "`top_k` must be >= 1.",
                remediation="Set `top_k` to a positive integer (default 10, max 50).",
                valid_example={"top_k": 10},
            ), {"error": "invalid_argument"}
        limit = min(top_k_int, 50)

        idx = self.index
        if idx is None:
            return (
                err(
                    "index_unavailable",
                    "BM25 index is not built or unreadable.",
                    remediation="The BM25 index is not built; fall back to semantic_search or pattern_search for this query, or ask the operator to build the BM25 index.",
                    index_path=str(self.index_path),
                ),
                {"error": "index_unavailable"},
            )

        sanitized = _sanitize_query(query).strip()
        if not sanitized:
            return (
                err(
                    "invalid_argument",
                    "`query` reduces to empty after punctuation stripping.",
                    remediation="Re-issue with at least one alphanumeric token in the query (the tokenizer strips +-!(){}[]^\"~*?:\\/ before parsing).",
                    valid_example={"query": "Premium USD"},
                ),
                {"error": "invalid_argument"},
            )
        try:
            q = idx.parse_query(sanitized, default_field_names=["text"])
        except Exception as exc:
            return (
                err(
                    "query_parse_failed",
                    f"Tantivy could not parse the query: {exc}",
                    remediation="Simplify the query to plain alphanumeric tokens; remove anything that looks like Lucene/tantivy operator syntax.",
                    query=sanitized,
                ),
                {"error": "query_parse_failed"},
            )

        # Pull deeper than `limit` so the in-memory scope filter still has
        # enough rows after dropping out-of-scope hits.
        scope_narrows = bool(scope.file_ids or scope.page_range or scope.section_ranges)
        depth = limit * (4 if scope_narrows else 1)
        searcher = idx.searcher()
        try:
            raw_hits = searcher.search(q, limit=depth).hits
        except Exception as exc:
            return (
                err(
                    "search_failed",
                    f"Tantivy search raised: {exc}",
                    remediation="Retry with a simpler query (fewer tokens, no special characters); if the failure repeats, fall back to semantic_search or pattern_search.",
                    query=sanitized,
                ),
                {"error": "search_failed"},
            )

        results: List[Dict[str, Any]] = []
        needles = [tok for tok in sanitized.split() if tok]
        for score, doc_addr in raw_hits:
            if len(results) >= limit:
                break
            doc = searcher.doc(doc_addr)
            file_id = (doc.get_first("file_id") or "")
            page_id = (doc.get_first("page_id") or "")
            if not file_id or not page_id:
                continue
            page_number = self._page_number(file_id, page_id)
            if not scope.contains(file_id, page_number):
                continue
            results.append(
                {
                    "file_id": str(file_id),
                    "page_id": str(page_id),
                    "page_number": page_number,
                    "score": round(float(score), 4),
                    "snippet": self._snippet(file_id, page_id, needles),
                }
            )

        retrieved_tokens = (
            len(self._tokenizer.encode("\n".join(r["snippet"] for r in results)))
            if results
            else 0
        )
        log_meta = {
            "query": query,
            "scope": scope.as_dict(),
            "top_k": limit,
            "hits": len(results),
        }
        context.add_retrieval_log(tool_name="bm25_search", tokens=retrieved_tokens, metadata=log_meta)

        return (
            ok(
                "PageSearchObservation",
                tool="bm25_search",
                query=query,
                scope=scope.as_dict(),
                results=results,
            ),
            {"retrieved_tokens": retrieved_tokens, "hits": len(results)},
        )

    # ----------------------------------------------------------- internals

    def _page_number(self, file_id: str, page_id: str) -> Optional[int]:
        if self.page_store is None:
            return None
        page = self.page_store.get(f"{file_id}/{page_id}")
        return page.page_number if page else None

    def _snippet(self, file_id: str, page_id: str, needles: List[str]) -> str:
        if self.page_store is None:
            return ""
        page = self.page_store.get(f"{file_id}/{page_id}")
        if page is None:
            return ""
        return keyword_snippet(page.text_markdown or "", needles)
