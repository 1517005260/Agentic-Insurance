"""Four retrieval indexes; each backed by a global cross-file store.

* ``text_dense``    — sentence text embeddings  → faiss store at ``STORAGE_PATH/faiss/dense/``
* ``vision_dense``  — page-image embeddings     → faiss store at ``STORAGE_PATH/faiss/visual/``
* ``bm25``          — tantivy index             → ``STORAGE_PATH/bm25/``
* ``graph``         — LinearRAG relation-free entity graph → ``STORAGE_PATH/faiss/graph/``

Every builder consumes a ``PageAsset`` list and a ``file_id``. Stores are
**global**: each call appends, ``file_id`` is a meta column for filtering
and per-file removal. Build-time the four builders are independent.
"""

from ingestion.index.base import IndexBuilder, IndexBuildResult
from ingestion.index.bm25_tantivy import BM25IndexBuilder
from ingestion.index.graph_linearrag import GraphIndexBuilder
from ingestion.index.maintenance import indexed_file_ids, purge_file_artifacts
from ingestion.index.text_dense import TextDenseIndexBuilder
from ingestion.index.vision_dense import VisionDenseIndexBuilder

__all__ = [
    "IndexBuilder",
    "IndexBuildResult",
    "BM25IndexBuilder",
    "GraphIndexBuilder",
    "TextDenseIndexBuilder",
    "VisionDenseIndexBuilder",
    "purge_file_artifacts",
    "indexed_file_ids",
]
