"""LinearRAG build-time graph construction.

Wraps the LinearRAG algorithm onto local primitives: the faiss-backed
:class:`storage.EmbeddingStore` (global, cross-file) replaces a
SentenceTransformer-backed store, and configuration is centralised in
``config.LinearRAGConfig``. Retrieval / PPR live with the query path,
not here.

This package intentionally re-exports nothing: the submodules pull in
spaCy + torch transitively, so importing the package would force the
~600 MB native baseline even for callers that only want a small
helper (e.g. ``normalize_for_hash``). Import the specific submodule
you need (``from ingestion.index.linear_rag.linear_rag import
LinearRAG``).
"""

__all__: list[str] = []
