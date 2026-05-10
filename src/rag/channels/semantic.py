"""Semantic channel — three sub-paths fused per page.

Sub-paths (run in parallel):

1. text_dense + embed(query)     — sentence-level cos sim
2. text_dense + embed(HyDE doc)  — sentence-level cos sim
3. vision_dense + embed(query)   — page-image cos sim

Within each sub-path each hit is one row of an ``EmbeddingStore``:

* text_dense rows are sentences with meta ``file_id`` / ``page_id``
* vision_dense rows are pages, with meta ``file_id`` / ``page_id``

All raw hits are pooled and aggregated per page via ``Σ cos / sqrt(N+1)``.
Cross-language is free here — both embedding models are multilingual.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from config import RAGConfig
from config.settings import faiss_dense_dir, faiss_visual_dir
from model_client import (
    EmbeddingClient,
    VisualEmbeddingClient,
    get_cached_embedding_client,
    get_cached_visual_embedding_client,
)
from rag.channels.base import BaseChannel, ChannelHit, RawHit, aggregate_per_page
from rag.preprocess import QueryContext
from storage import EmbeddingStore
from storage.embedding_store import get_or_create_store


class SemanticChannel(BaseChannel):
    name = "semantic"

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        visual_client: Optional[VisualEmbeddingClient] = None,
        text_store: Optional[EmbeddingStore] = None,
        visual_store: Optional[EmbeddingStore] = None,
    ):
        self.config = config or RAGConfig()
        self.embedding_client = embedding_client or get_cached_embedding_client()
        self.visual_client = visual_client or get_cached_visual_embedding_client()
        self._text_store = text_store
        self._visual_store = visual_store

    @property
    def text_store(self) -> EmbeddingStore:
        if self._text_store is None:
            # Pull from process cache so the dense store is the same
            # in-memory faiss as the ingest builder writes into.
            self._text_store = get_or_create_store(faiss_dense_dir(), namespace="dense")
        return self._text_store

    @property
    def visual_store(self) -> EmbeddingStore:
        if self._visual_store is None:
            self._visual_store = get_or_create_store(faiss_visual_dir(), namespace="visual")
        return self._visual_store

    # ---------------------------------------------------------- retrieve ----

    def retrieve(self, ctx: QueryContext) -> List[ChannelHit]:
        cfg = self.config
        topk = cfg.semantic_topk_per_subpath

        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_text_orig = pool.submit(self._text_dense, ctx.query, topk, ctx.file_ids)
            fut_text_hyde = pool.submit(
                self._text_dense, ctx.hyde or ctx.query, topk, ctx.file_ids
            )
            fut_vision = pool.submit(self._vision_dense, ctx.query, topk, ctx.file_ids)

            raw: List[RawHit] = []
            raw.extend(fut_text_orig.result())
            raw.extend(fut_text_hyde.result())
            raw.extend(fut_vision.result())

        return aggregate_per_page(raw, top_k=cfg.semantic_channel_topk)

    # --------------------------------------------------------- sub-paths ----

    def _text_dense(
        self,
        query_text: str,
        top_k: int,
        file_ids: Optional[List[str]],
    ) -> List[RawHit]:
        store = self.text_store
        if len(store) == 0 or not query_text.strip():
            return []
        emb = self.embedding_client.encode(query_text)
        # Pull a deeper top-k when post-filtering by file_ids so we still
        # have ``top_k`` rows after the filter — heuristic: 4×.
        depth = top_k * 4 if file_ids else top_k
        scored = store.topk(emb, depth)
        return self._materialize_text_hits(store, scored, top_k, file_ids)

    def _vision_dense(
        self,
        query_text: str,
        top_k: int,
        file_ids: Optional[List[str]],
    ) -> List[RawHit]:
        store = self.visual_store
        if len(store) == 0 or not self.visual_client.available() or not query_text.strip():
            return []
        try:
            emb = self.visual_client.encode_text(query_text)
        except Exception:
            return []
        depth = top_k * 4 if file_ids else top_k
        scored = store.topk(emb, depth)
        return self._materialize_vision_hits(store, scored, top_k, file_ids)

    # --------------------------------------------------------- materialize ----

    @staticmethod
    def _materialize_text_hits(
        store: EmbeddingStore,
        scored: list,
        top_k: int,
        file_ids: Optional[List[str]],
    ) -> List[RawHit]:
        out: List[RawHit] = []
        file_id_filter = set(file_ids) if file_ids else None
        for hash_id, score in scored:
            row = store.get_meta_row(hash_id)
            file_id = row.get("file_id")
            page_id = row.get("page_id")
            if not file_id or not page_id:
                continue
            if file_id_filter and file_id not in file_id_filter:
                continue
            out.append(
                RawHit(
                    file_id=str(file_id),
                    page_id=str(page_id),
                    score=float(score),
                    evidence={"sentence": row.get("text", ""), "sub_path": "text"},
                )
            )
            if len(out) >= top_k:
                break
        return out

    @staticmethod
    def _materialize_vision_hits(
        store: EmbeddingStore,
        scored: list,
        top_k: int,
        file_ids: Optional[List[str]],
    ) -> List[RawHit]:
        out: List[RawHit] = []
        file_id_filter = set(file_ids) if file_ids else None
        for hash_id, score in scored:
            row = store.get_meta_row(hash_id)
            file_id = row.get("file_id")
            page_id = row.get("page_id")
            if not file_id or not page_id:
                continue
            if file_id_filter and file_id not in file_id_filter:
                continue
            out.append(
                RawHit(
                    file_id=str(file_id),
                    page_id=str(page_id),
                    score=float(score),
                    evidence={"image_path": row.get("image_path"), "sub_path": "vision"},
                )
            )
            if len(out) >= top_k:
                break
        return out
