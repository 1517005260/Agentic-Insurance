"""Local persistence primitives — page assets, embedding stores, evidence/trace."""

from storage.embedding_store import EmbeddingStore
from storage.inventory_store import InventoryStore, Section
from storage.page_store import PageAsset, PageStore, make_global_id, split_global_id
from storage.passage_store import Passage, PassageCacheMissing, PassageStore
from storage.table_row_store import TableRow, TableRowCacheMissing, TableRowStore


def build_inventory_atoms_for_file(
    file_id: str,
    page_store: PageStore,
    inventory: InventoryStore,
) -> None:
    """Eagerly build the passage + table_row caches for one file.

    Called from the ingest pipeline (``ingestion.page_assets.build_page_assets``)
    immediately after a file's page assets are persisted. Idempotent:
    re-running on a fresh corpus is a fast no-op (page_count tripwire
    matches → existing cache loaded). Invoked at ingest time so the
    proof gate's read path stays cache-only — no on-the-fly rebuild
    inside an agent loop.
    """
    inventory.passage_store.build(file_id)
    inventory.table_row_store.build(file_id)


__all__ = [
    "EmbeddingStore",
    "InventoryStore",
    "PageAsset",
    "PageStore",
    "Passage",
    "PassageCacheMissing",
    "PassageStore",
    "Section",
    "TableRow",
    "TableRowCacheMissing",
    "TableRowStore",
    "build_inventory_atoms_for_file",
    "make_global_id",
    "split_global_id",
]
