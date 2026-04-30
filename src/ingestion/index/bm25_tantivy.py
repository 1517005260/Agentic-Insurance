"""Global tantivy BM25 index over page Markdown.

Stores one shared index at ``STORAGE_PATH/bm25/index/``. New file ingest
appends new documents; ``file_id`` is a stored field for filtered retrieval
and per-file removal.

Schema:
    file_id : TEXT (stored, not tokenized)
    page_id : TEXT (stored, not tokenized)
    text    : TEXT (default tokenizer; the field BM25 ranks against)
"""
import json
from pathlib import Path
from typing import List

import tantivy

from config.settings import bm25_root
from ingestion.index.base import IndexBuilder, IndexBuildResult
from storage.page_store import PageAsset


class BM25IndexBuilder(IndexBuilder):
    name = "bm25"

    @property
    def output_dir(self) -> Path:
        return bm25_root()

    def _build(self, file_id: str, pages: List[PageAsset]) -> IndexBuildResult:
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        index_path = out_dir / "index"
        meta_path = out_dir / "meta.json"

        existing_meta: dict = {}
        if meta_path.exists():
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        file_counts: dict = existing_meta.get("file_counts", {})

        # Per-file dedup: if this file_id already has documents in the index,
        # skip the rerun. ``maintenance.remove_file`` should be called first
        # if you want to re-ingest a file from scratch.
        if file_id in file_counts:
            return IndexBuildResult(
                index_name=self.name,
                file_id=file_id,
                output_dir=str(out_dir),
                item_count=0,
                skipped_reason="file_id already in BM25 index — call remove_file to rebuild",
                extra={"file_counts": file_counts},
            )

        if index_path.exists():
            index = tantivy.Index.open(str(index_path))
        else:
            index_path.mkdir(parents=True, exist_ok=True)
            schema_builder = tantivy.SchemaBuilder()
            schema_builder.add_text_field("file_id", stored=True)
            schema_builder.add_text_field("page_id", stored=True)
            schema_builder.add_text_field("text", stored=True)
            schema = schema_builder.build()
            index = tantivy.Index(schema, path=str(index_path))

        writer = index.writer()
        for page in pages:
            doc = tantivy.Document()
            doc.add_text("file_id", file_id)
            doc.add_text("page_id", page.page_id)
            doc.add_text("text", page.text_markdown)
            writer.add_document(doc)
        writer.commit()
        writer.wait_merging_threads()

        file_counts[file_id] = len(pages)
        meta = {"fields": ["file_id", "page_id", "text"], "file_counts": file_counts}
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return IndexBuildResult(
            index_name=self.name,
            file_id=file_id,
            output_dir=str(out_dir),
            item_count=len(pages),
            extra={"file_counts": file_counts},
        )
