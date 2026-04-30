"""Sentence-level text-embedding index.

Writes sentence embeddings into a global faiss store at
``STORAGE_PATH/faiss/dense/``. Each row's meta carries ``(page_id, file_id,
lang)``. The dedup key is ``"{file_id}|{page_id}|{sentence}"`` — same
sentence on different pages or in different files yields distinct rows so
per-(file, page) retrieval is unambiguous.
"""
from pathlib import Path
from typing import List, Optional

import numpy as np

from config.settings import faiss_dense_dir
from ingestion.index._sentence import split_sentences
from ingestion.index.base import IndexBuilder, IndexBuildResult
from model_client import EmbeddingClient
from storage import EmbeddingStore
from storage.page_store import PageAsset


class TextDenseIndexBuilder(IndexBuilder):
    name = "text_dense"

    def __init__(self, embedding_client: Optional[EmbeddingClient] = None):
        self.embedding_client = embedding_client or EmbeddingClient()
        self._store: Optional[EmbeddingStore] = None

    @property
    def output_dir(self) -> Path:
        return faiss_dense_dir()

    def _get_store(self) -> EmbeddingStore:
        if self._store is None:
            self._store = EmbeddingStore(self.output_dir, namespace="dense")
        return self._store

    def _build(self, file_id: str, pages: List[PageAsset]) -> IndexBuildResult:
        store = self._get_store()

        sentences: List[str] = []
        page_ids: List[str] = []
        file_ids: List[str] = []
        for page in pages:
            for sent in split_sentences(page.text_markdown):
                sentences.append(sent)
                page_ids.append(page.page_id)
                file_ids.append(file_id)

        if not sentences:
            return IndexBuildResult(
                index_name=self.name,
                file_id=file_id,
                output_dir=str(self.output_dir),
                item_count=0,
                skipped_reason="no sentences extracted",
            )

        # Compose a unique dedup key per (file_id, page_id, sentence).
        keys = [f"{f}|{p}|{s}" for f, p, s in zip(file_ids, page_ids, sentences)]
        hash_ids = [store.hash_for(k) for k in keys]

        # Embed only the rows we don't yet have.
        new_local_idx = [i for i, h in enumerate(hash_ids) if not store.has(h)]
        if not new_local_idx:
            return IndexBuildResult(
                index_name=self.name,
                file_id=file_id,
                output_dir=str(self.output_dir),
                item_count=0,
                skipped_reason="all sentences already embedded for this file",
                extra={"store_size": len(store)},
            )

        new_sentences = [sentences[i] for i in new_local_idx]
        new_hash_ids = [hash_ids[i] for i in new_local_idx]
        new_page_ids = [page_ids[i] for i in new_local_idx]
        new_file_ids = [file_ids[i] for i in new_local_idx]

        embeddings = self.embedding_client.encode(new_sentences)
        if isinstance(embeddings, list):
            embeddings = np.asarray(embeddings, dtype=np.float32)

        added = store.add(
            new_hash_ids,
            new_sentences,
            embeddings,
            extra_metadata={"page_id": new_page_ids, "file_id": new_file_ids},
        )

        return IndexBuildResult(
            index_name=self.name,
            file_id=file_id,
            output_dir=str(self.output_dir),
            item_count=len(added),
            extra={
                "embedding_model": self.embedding_client.model,
                "store_size": len(store),
            },
        )
