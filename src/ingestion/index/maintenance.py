"""Top-level index maintenance facade.

The heavy lifting (passage / entity / sentence orphan sweeps in the
LinearRAG graph) lives in :mod:`ingestion.index.linear_rag.maintenance`,
because those operations need intimate knowledge of the graph schema.
This module re-exports the cross-index entry points the web layer
actually needs, under names that don't pretend the code is graph-only:

* :func:`purge_file_artifacts`  — delete every disk artifact tagged with
  ``file_id`` across all four indexes (dense / visual / bm25 / graph) plus
  the upstream caches (page_assets, inventory, paddle_ocr, uploads).
* :func:`indexed_file_ids`      — set of file_ids that currently appear in
  any index (drift detector against the ``files`` SQL row).

Both are pure-disk operations, no DB / network I/O. The web layer wraps
them with a job row + audit entry.
"""
import json
from typing import Dict, Set

from config.settings import (
    bm25_root,
    faiss_dense_dir,
    faiss_graph_passage_dir,
    faiss_visual_dir,
    inventory_atoms_root,
    inventory_root,
    page_assets_root,
    paddle_ocr_root,
    uploads_root,
)
from ingestion.index.linear_rag.maintenance import remove_file as _remove_file_impl
from storage import EmbeddingStore


def purge_file_artifacts(
    file_id: str,
    *,
    keep_upload: bool = False,
    upload_suffix: str | None = None,
) -> Dict[str, int]:
    """Wipe every on-disk artifact tagged with ``file_id``.

    ``keep_upload=True`` preserves ``uploads/<file_id>.*`` so a re-ingest
    can drop indexes and then re-feed the same blob to the parser. Set
    ``False`` (default) for an outright delete.

    ``upload_suffix`` (e.g. ``".pdf"``) makes the uploads-dir delete
    EXACT. Pass it whenever the caller knows the suffix (DB row → the
    delete route does this). Without it, the linear_rag layer falls
    back to a stem match that's still safe but only finds files whose
    full stem equals ``file_id``.

    Idempotent: each step short-circuits if its target is already absent,
    so calling on a half-deleted file is safe.

    Returns a dict of per-step counts (rows dropped, dirs removed, etc.)
    suitable for the ``ingest_jobs.log_tail`` audit field.
    """
    return _remove_file_impl(
        file_id, keep_upload=keep_upload, upload_suffix=upload_suffix
    )


def indexed_file_ids() -> Set[str]:
    """Union of file_ids present in any index store or upstream cache.

    Useful for drift detection: ``indexed_file_ids() - {row.file_id for row in files}``
    is the set of files that exist on disk but are absent from the
    ``files`` table (orphan artifacts a previous failure left behind).
    """
    seen: Set[str] = set()

    # Per-file JSON / directory caches. Stems carry the file_id; one
    # entry per file.
    for cache_root, scan in (
        (page_assets_root(), "json"),
        (inventory_root(), "json"),
        (inventory_atoms_root("passages"), "json"),
        (inventory_atoms_root("table_rows"), "json"),
        (paddle_ocr_root(), "dir"),
    ):
        if not cache_root.exists():
            continue
        if scan == "json":
            for fp in cache_root.glob("*.json"):
                seen.add(fp.stem)
        else:
            for d in cache_root.iterdir():
                if d.is_dir():
                    seen.add(d.name)

    # Uploads: ``<file_id><suffix>``. The suffix is unknown so we strip
    # the last extension from each entry. Bare-no-suffix files are kept
    # as-is. Skip the ``.<file_id>.<rand>.part`` temp files staging
    # leaves behind on a crashed write — those would otherwise surface
    # as bogus orphan file_ids.
    up_root = uploads_root()
    if up_root.exists():
        for entry in up_root.iterdir():
            if not entry.is_file():
                continue
            if entry.name.startswith(".") and entry.suffix == ".part":
                continue
            # ``.with_suffix('').name`` strips only the final ext, which
            # matches the inverse of ``upload_path(file_id, suffix)``.
            seen.add(entry.with_suffix("").name)

    # Faiss meta columns — one row per (file_id, *).
    for store_dir, ns in (
        (faiss_dense_dir(), "dense"),
        (faiss_visual_dir(), "visual"),
        (faiss_graph_passage_dir(), "passage"),
    ):
        if not store_dir.exists():
            continue
        try:
            store = EmbeddingStore(store_dir, namespace=ns)
        except Exception:
            # A partial-write corruption is loud (raises in EmbeddingStore.__init__);
            # callers handling the exception above us should report; here we
            # just skip this store rather than crash the audit.
            continue
        seen.update(fid for fid in store.meta_column("file_id") if isinstance(fid, str) and fid)

    # BM25 meta.json — file_counts dict.
    bm25_meta = bm25_root() / "meta.json"
    if bm25_meta.exists():
        try:
            data = json.loads(bm25_meta.read_text(encoding="utf-8"))
            seen.update((data.get("file_counts") or {}).keys())
        except json.JSONDecodeError:
            pass

    return seen


__all__ = ["purge_file_artifacts", "indexed_file_ids"]
