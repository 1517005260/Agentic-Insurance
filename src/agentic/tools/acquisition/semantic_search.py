"""Sentence-level embedding search backed by the global faiss dense store.

Each indexed sentence carries ``(page_id, file_id)`` metadata. The query is
embedded, faiss returns the top-K candidate sentences, and we aggregate to
the page level (per-page score = max sentence similarity).
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import tiktoken

from agentic.tools.base import BaseTool
from config.settings import faiss_dense_dir
from model_client import EmbeddingClient
from storage import EmbeddingStore

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


class SemanticSearchTool(BaseTool):
    _embedding_lock = threading.Lock()

    def __init__(
        self,
        store_dir: Optional[Path] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.store_dir = Path(store_dir) if store_dir is not None else faiss_dense_dir()
        self.embedding_client = embedding_client or EmbeddingClient()
        self.store = EmbeddingStore(self.store_dir, namespace="dense")
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    @property
    def name(self) -> str:
        return "semantic_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": (
                    "Semantic search via embedding similarity. Use when keyword "
                    "search misses or the exact wording is unknown.\n\n"
                    "Returns page-level snippets — call read_page on promising IDs "
                    "to get the full Markdown."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of most relevant pages (default: 5, max: 20).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(
        self, context: "AgentContext", query: str, top_k: int = 5
    ) -> Tuple[str, Dict[str, Any]]:
        top_k = min(top_k, 20)
        if len(self.store) == 0:
            return f"No results for: {query}", {"retrieved_tokens": 0, "pages_found": 0}

        with self._embedding_lock:
            query_embedding = self.embedding_client.encode(query)

        # Pull more sentences than pages so per-page max sim aggregates well.
        sentence_topk = max(top_k * 5, 10)
        hits = self.store.topk(query_embedding, sentence_topk)

        # Aggregate by global_id ("file_id/page_id") so cross-file collisions
        # on plain page_id don't merge.
        page_sentences: Dict[str, List[Dict[str, Any]]] = {}
        for hash_id, score in hits:
            row = self.store.get_meta_row(hash_id)
            page_id = row.get("page_id")
            file_id = row.get("file_id") or ""
            sentence = row.get("text", "")
            if not page_id:
                continue
            global_id = f"{file_id}/{page_id}" if file_id else page_id
            page_sentences.setdefault(global_id, []).append(
                {"sentence": sentence, "similarity": float(score)}
            )

        page_scores = sorted(
            (
                (gid, max(s["similarity"] for s in sents), sents)
                for gid, sents in page_sentences.items()
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        top_pages = page_scores[:top_k]
        if not top_pages:
            return f"No results for: {query}", {"retrieved_tokens": 0, "pages_found": 0}

        result_parts = []
        for global_id, max_sim, sents in top_pages:
            sents_sorted = sorted(sents, key=lambda x: -x["similarity"])
            matched_text = "... " + " ... ".join(s["sentence"] for s in sents_sorted) + " ..."
            result_parts.append(
                f"Page: {global_id} (Similarity: {max_sim:.3f})\nMatched: {matched_text}"
            )
        tool_result = "\n\n".join(result_parts)

        all_matched = [s["sentence"] for _, _, sents in top_pages for s in sents]
        retrieved_tokens = (
            len(self.tokenizer.encode("\n".join(all_matched))) if all_matched else 0
        )

        context.add_retrieval_log(
            tool_name="semantic_search",
            tokens=retrieved_tokens,
            metadata={"query": query, "pages_found": len(top_pages)},
        )

        return tool_result, {
            "retrieved_tokens": retrieved_tokens,
            "pages_found": len(top_pages),
        }
