"""Page-image embedding index.

Writes page-image embeddings into a global faiss store at
``STORAGE_PATH/faiss/visual/``. The store's ``text`` field carries the
``"<file_id>/<page_id>"`` sentinel (used for dedup hashing); the absolute
image path lives in the meta column ``image_path``.

When ``VISUAL_EMBEDDING_MODEL`` is unset the builder no-ops with
``skipped_reason`` so the pipeline keeps going.
"""
from pathlib import Path
from typing import List, Optional

import numpy as np

from config.settings import faiss_visual_dir, paddle_ocr_root
from ingestion.index.base import IndexBuilder, IndexBuildResult
from model_client import VisualEmbeddingClient, get_cached_visual_embedding_client
from storage import EmbeddingStore
from storage.embedding_store import get_or_create_store
from storage.page_store import PageAsset


class VisionDenseIndexBuilder(IndexBuilder):
    name = "vision_dense"

    def __init__(self, visual_client: Optional[VisualEmbeddingClient] = None):
        self.visual_client = visual_client or get_cached_visual_embedding_client()
        self._store: Optional[EmbeddingStore] = None

    @property
    def output_dir(self) -> Path:
        return faiss_visual_dir()

    def _get_store(self) -> EmbeddingStore:
        if self._store is None:
            # Process-cached store — see text_dense._get_store for
            # rationale (avoid re-loading a large faiss index per ingest).
            self._store = get_or_create_store(self.output_dir, namespace="visual")
        return self._store

    def _build(self, file_id: str, pages: List[PageAsset]) -> IndexBuildResult:
        store = self._get_store()

        if not self.visual_client.available():
            return IndexBuildResult(
                index_name=self.name,
                file_id=file_id,
                output_dir=str(self.output_dir),
                item_count=0,
                skipped_reason="VISUAL_EMBEDDING_MODEL not configured",
            )

        file_root = paddle_ocr_root() / file_id
        sentinels: List[str] = []
        page_ids: List[str] = []
        abs_image_paths: List[Path] = []
        for page in pages:
            if not page.page_image_path:
                continue
            abs_path = file_root / page.page_image_path
            if not abs_path.is_file():
                continue
            sentinels.append(f"{file_id}/{page.page_id}")
            page_ids.append(page.page_id)
            abs_image_paths.append(abs_path)

        if not abs_image_paths:
            return IndexBuildResult(
                index_name=self.name,
                file_id=file_id,
                output_dir=str(self.output_dir),
                item_count=0,
                skipped_reason="no rendered page images on disk",
            )

        # Skip embedding for images we already have (hash dedup).
        new_indices = [i for i, s in enumerate(sentinels) if not store.has(store.hash_for(s))]
        if not new_indices:
            return IndexBuildResult(
                index_name=self.name,
                file_id=file_id,
                output_dir=str(self.output_dir),
                item_count=0,
                skipped_reason="all page images already embedded",
                extra={"store_size": len(store)},
            )

        new_paths = [abs_image_paths[i] for i in new_indices]
        new_sentinels = [sentinels[i] for i in new_indices]
        new_page_ids = [page_ids[i] for i in new_indices]

        embeddings = self.visual_client.encode_paths(new_paths)
        if isinstance(embeddings, list):
            embeddings = np.asarray(embeddings, dtype=np.float32)

        hash_ids = [store.hash_for(s) for s in new_sentinels]
        added = store.add(
            hash_ids,
            new_sentinels,
            embeddings,
            extra_metadata={
                "page_id": new_page_ids,
                "file_id": [file_id] * len(new_sentinels),
                "image_path": [str(p) for p in new_paths],
            },
        )
        # EmbeddingStore.add() no longer saves implicitly; persist now
        # so the per-file builder contract holds.
        store.save()

        return IndexBuildResult(
            index_name=self.name,
            file_id=file_id,
            output_dir=str(self.output_dir),
            item_count=len(added),
            extra={
                "embedding_model": self.visual_client.model,
                "store_size": len(store),
            },
        )
