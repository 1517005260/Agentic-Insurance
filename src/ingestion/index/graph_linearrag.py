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
from config.settings import faiss_graph_dir, models_root
from ingestion.index.base import IndexBuilder, IndexBuildResult
from ingestion.index.linear_rag.linear_rag import LinearRAG
from model_client import EmbeddingClient, get_cached_embedding_client
from storage.page_store import PageAsset


class GraphIndexBuilder(IndexBuilder):
    name = "graph"

    def __init__(
        self,
        embedding_client: Optional[EmbeddingClient] = None,
        spacy_model_name: str = "en_core_web_trf",
        zh_spacy_model_name: Optional[str] = "zh_core_web_trf",
        max_workers: int = 4,
        linear_config: Optional[LinearRAGConfig] = None,
    ):
        self.embedding_client = embedding_client or get_cached_embedding_client()
        self.spacy_model_name = spacy_model_name
        self.zh_spacy_model_name = zh_spacy_model_name
        self.max_workers = max_workers
        # Carries the admin-tuned literal-backfill flags. The
        # embedding_client / spacy paths / max_workers fields here will
        # be overwritten in _build() with the per-call values; only the
        # backfill knobs survive. None preserves pre-Phase-6 behaviour
        # for experiment scripts that construct the builder directly.
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

        # spacy.load() interprets relative names as installed-package names
        # before falling back to paths. Resolve to absolute so the model is
        # always loaded from disk, regardless of cwd.
        spacy_model_path = (models_root() / self.spacy_model_name).resolve()

        # ZH model path is opt-in; only set when the model directory actually
        # exists on disk so we don't crash on EN-only deployments.
        zh_path: Optional[str] = None
        if self.zh_spacy_model_name:
            candidate = (models_root() / self.zh_spacy_model_name).resolve()
            if (candidate / "config.cfg").is_file():
                zh_path = str(candidate)

        # Compose: take the admin-provided literal-backfill knobs from
        # self.linear_config when present, then override the
        # plumbing-only fields (embedding_client, spaCy paths,
        # max_workers) with the per-call values. ``replace`` keeps
        # the dataclass immutable-by-construction.
        base_lc = self.linear_config or LinearRAGConfig()
        config = _replace(
            base_lc,
            embedding_client=self.embedding_client,
            spacy_model=str(spacy_model_path),
            zh_spacy_model=zh_path,
            max_workers=self.max_workers,
        )

        graph = LinearRAG(config)
        added = graph.index(passages, file_id=file_id, page_numbers=page_numbers)

        return IndexBuildResult(
            index_name=self.name,
            file_id=file_id,
            output_dir=str(self.output_dir),
            item_count=added["passages"],
            extra={
                "spacy_model": self.spacy_model_name,
                "embedding_model": self.embedding_client.model,
                "graphml": str(self.output_dir / "LinearRAG.graphml"),
                "graph_v": graph.graph.vcount(),
                "graph_e": graph.graph.ecount(),
                "added": added,
            },
        )
