"""LinearRAG build-time graph construction.

Wraps the LinearRAG algorithm onto local primitives: the faiss-backed
:class:`storage.EmbeddingStore` (global, cross-file) replaces a
SentenceTransformer-backed store, and configuration is centralised in
``config.LinearRAGConfig``. Retrieval / PPR live with the query path,
not here.
"""

from ingestion.index.linear_rag.disambig import (
    add_alias_edges,
    compute_clusters,
    get_clusters,
    gradient_topk_candidates,
    invalidate_clusters,
    mutual_topk_filter,
    write_clusters,
)
from ingestion.index.linear_rag.linear_rag import LinearRAG
from ingestion.index.linear_rag.maintenance import remove_file, split_cluster, unalias
from ingestion.index.linear_rag.ner import SpacyNER
from ingestion.index.linear_rag.normalize import (
    canonical_form,
    cleanup,
    is_junk,
    normalize_for_hash,
)
from ingestion.index.linear_rag.utils import compute_mdhash_id

__all__ = [
    "LinearRAG",
    "SpacyNER",
    "compute_mdhash_id",
    "gradient_topk_candidates",
    "mutual_topk_filter",
    "add_alias_edges",
    "compute_clusters",
    "get_clusters",
    "write_clusters",
    "invalidate_clusters",
    "unalias",
    "split_cluster",
    "remove_file",
    "cleanup",
    "is_junk",
    "canonical_form",
    "normalize_for_hash",
]
