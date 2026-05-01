"""BM25 channel — three sub-queries fused per page.

Sub-queries (run in parallel against the same tantivy index):

* ``query`` (original)
* ``rewrite`` (LLM paraphrase)
* ``hyde`` (LLM hypothetical answer)

Each sub-query returns top-K (file_id, page_id, bm25_score). The three
result lists are pooled and aggregated per page via ``Σ score / sqrt(N+1)``.

Cross-language note: BM25 does not generalize across languages. A Chinese
query against an English-only corpus will see weak signal. The HyDE
sub-query helps because the hypothetical answer often introduces vocabulary
in both languages; we accept the residual gap as a known limitation (the
embedding channels carry cross-lingual recall).
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import tantivy

from config import RAGConfig
from config.settings import bm25_root
from rag.channels.base import BaseChannel, ChannelHit, RawHit, aggregate_per_page
from rag.preprocess import QueryContext


# Characters tantivy's QueryParser treats as syntax. We strip them from
# free-text sub-queries so user-provided punctuation can't accidentally
# turn the input into a malformed query expression.
_TANTIVY_SPECIAL = '+-!(){}[]^"~*?:\\/'


def _sanitize_query(text: str) -> str:
    if not text:
        return ""
    return "".join(" " if ch in _TANTIVY_SPECIAL else ch for ch in text)


class BM25Channel(BaseChannel):
    name = "bm25"

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        index_path: Optional[Path] = None,
    ):
        self.config = config or RAGConfig()
        self.index_path = index_path or (bm25_root() / "index")
        self._index: Optional[tantivy.Index] = None

    @property
    def index(self) -> Optional[tantivy.Index]:
        if self._index is None and self.index_path.is_dir():
            self._index = tantivy.Index.open(str(self.index_path))
        return self._index

    def retrieve(self, ctx: QueryContext) -> List[ChannelHit]:
        idx = self.index
        if idx is None:
            return []
        cfg = self.config
        sub_queries = [q for q in (ctx.query, ctx.rewrite, ctx.hyde) if q and q.strip()]

        searcher = idx.searcher()
        with ThreadPoolExecutor(max_workers=len(sub_queries) or 1) as pool:
            futures = [
                pool.submit(self._run_one, idx, searcher, q, cfg.bm25_topk_per_query, ctx.file_ids)
                for q in sub_queries
            ]
            raw: List[RawHit] = []
            for f in futures:
                raw.extend(f.result())

        return aggregate_per_page(raw, top_k=cfg.bm25_channel_topk)

    @staticmethod
    def _run_one(
        idx: tantivy.Index,
        searcher,
        query_text: str,
        top_k: int,
        file_ids: Optional[List[str]],
    ) -> List[RawHit]:
        sanitized = _sanitize_query(query_text).strip()
        if not sanitized:
            return []
        try:
            query = idx.parse_query(sanitized, default_field_names=["text"])
        except Exception:
            return []
        # Pull deeper when filtering by file_ids so the post-filter still
        # has top_k rows (heuristic: 4×).
        depth = top_k * 4 if file_ids else top_k
        try:
            hits = searcher.search(query, limit=depth).hits
        except Exception:
            return []

        file_id_filter = set(file_ids) if file_ids else None
        out: List[RawHit] = []
        for score, doc_addr in hits:
            doc = searcher.doc(doc_addr)
            file_id = (doc.get_first("file_id") or "")
            page_id = (doc.get_first("page_id") or "")
            if not file_id or not page_id:
                continue
            if file_id_filter and file_id not in file_id_filter:
                continue
            out.append(
                RawHit(
                    file_id=str(file_id),
                    page_id=str(page_id),
                    score=float(score),
                    evidence={"sub_query": query_text[:80]},
                )
            )
            if len(out) >= top_k:
                break
        return out
