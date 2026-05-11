"""Rerank candidate pages with the local Qwen3-Reranker cross-encoder."""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from config import RAGConfig
from model_client import RerankClient, get_cached_rerank_client
from storage.page_store import PageAsset


@dataclass
class RerankedPage:
    page: PageAsset
    score: float


def rerank_pages(
    query: str,
    pages: Sequence[PageAsset],
    *,
    config: Optional[RAGConfig] = None,
    client: Optional[RerankClient] = None,
) -> List[RerankedPage]:
    """Send page Markdowns to the reranker and return the top-N.

    Each page's Markdown is truncated to ``config.rerank_doc_max_chars`` so
    a single huge page can't blow the request budget. Local Qwen3-Reranker
    is true pairwise (no cross-request normalization) so the
    ``relevance_score`` is directly comparable across calls — caller can
    cache or threshold without rescoring.
    """
    cfg = config or RAGConfig()
    rc = client or get_cached_rerank_client()
    if not pages or not rc.available():
        return [RerankedPage(page=p, score=0.0) for p in pages[: cfg.rerank_top_n]]

    docs: List[str] = []
    for p in pages:
        text = p.text_markdown or ""
        if len(text) > cfg.rerank_doc_max_chars:
            text = text[: cfg.rerank_doc_max_chars]
        docs.append(text)

    results = rc.rerank(query=query, documents=docs, top_n=cfg.rerank_top_n)
    out: List[RerankedPage] = []
    for r in results:
        idx = r["index"]
        if 0 <= idx < len(pages):
            out.append(RerankedPage(page=pages[idx], score=r["relevance_score"]))
    return out
