"""Graph index built by LinearRAG.

Thin adapter around :class:`ingestion.index.linear_rag.LinearRAG`. Each
PageAsset becomes one passage (its plain ``text_markdown``); the
``(file_id, page_number)`` identity is carried as meta columns on the
passage embedding store, NOT prefixed into the text — that keeps the
text fed to embedding / NER / lang routing free of metadata pollution.

Build artifacts land under ``STORAGE_PATH/faiss/graph/``:

    passage/, entity/, sentence/   (faiss EmbeddingStore each)
    ner_results.json
    LinearRAG.graphml
"""
from pathlib import Path
from typing import List, Optional

from config import LinearRAGConfig
from config.settings import faiss_graph_dir
from ingestion.index.base import IndexBuilder, IndexBuildResult
from ingestion.index.linear_rag.linear_rag import LinearRAG
from model_client import EmbeddingClient, get_cached_embedding_client
from storage.page_store import PageAsset


class GraphIndexBuilder(IndexBuilder):
    name = "graph"

    def __init__(
        self,
        embedding_client: Optional[EmbeddingClient] = None,
        max_workers: int = 4,
        linear_config: Optional[LinearRAGConfig] = None,
        reuse_graph: bool = False,
    ):
        self.embedding_client = embedding_client or get_cached_embedding_client()
        self.max_workers = max_workers
        # Default False = a fresh LinearRAG per _build (per-file API
        # path: each upload must load+persist its own graphml, unchanged).
        # True = one persistent LinearRAG reused across _build calls so
        # a bulk corpus build loads graphml ONCE and writes on the
        # config's graphml_flush_every cadence instead of a full O(V+E)
        # read+write per doc (the 650-build O(N²) fix). Opt-in only.
        self.reuse_graph = reuse_graph
        self._lr = None
        # Carries the admin-tuned literal-backfill flags + GLiNER knobs.
        # The embedding_client / max_workers fields here will be overwritten
        # in _build() with the per-call values; the backfill / NER knobs
        # survive from config-store hydration. ``None`` keeps the built-in
        # defaults so experiment scripts that construct the builder
        # directly need no changes.
        self.linear_config = linear_config

    @property
    def output_dir(self) -> Path:
        return faiss_graph_dir()

    def _build(self, file_id: str, pages: List[PageAsset]) -> IndexBuildResult:
        from dataclasses import replace as _replace

        eligible = [p for p in pages if p.text_markdown.strip()]
        passages = [p.text_markdown for p in eligible]
        page_numbers = [
            p.page_number if p.page_number is not None else 0 for p in eligible
        ]

        # Compose: take the admin-provided knobs (literal-backfill + GLiNER
        # labels / threshold / model id) from self.linear_config when
        # present, then override the plumbing-only fields with the
        # per-call values. ``replace`` keeps the dataclass
        # immutable-by-construction.
        base_lc = self.linear_config or LinearRAGConfig()
        config = _replace(
            base_lc,
            embedding_client=self.embedding_client,
            max_workers=self.max_workers,
        )

        if self.reuse_graph:
            if self._lr is None:
                self._lr = LinearRAG(config)
            graph = self._lr
        else:
            graph = LinearRAG(config)
        added = graph.index(passages, file_id=file_id, page_numbers=page_numbers)

        return IndexBuildResult(
            index_name=self.name,
            file_id=file_id,
            output_dir=str(self.output_dir),
            item_count=added["passages"],
            extra={
                "ner_model": config.gliner_model_id,
                "embedding_model": self.embedding_client.model,
                "graphml": str(self.output_dir / "LinearRAG.graphml"),
                "graph_v": graph.graph.vcount(),
                "graph_e": graph.graph.ecount(),
                "added": added,
            },
        )

    def flush(self) -> None:
        """Force-persist the reused graph + all deferred writes.

        Bulk driver: call at the end and before any checkpoint that
        reads the on-disk graphml / faiss / parquet artifacts. No-op
        unless ``reuse_graph`` and a graph has been built.

        Drains the cadence-deferred work (3 embedding stores, NER JSON,
        literal backfill, graphml) via ``LinearRAG.flush_all``.
        """
        if self._lr is not None:
            self._lr.flush_all()
