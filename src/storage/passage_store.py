"""Per-file passage inventory.

A *passage* is one PaddleOCR layout block whose ``block_label`` is
content-bearing prose: ``text`` (paragraph) or ``paragraph_title``
(section sub-heading the LLM may want to cite separately from the
parent section). Header/footer/figure-caption/aside blocks are
excluded — they are layout decoration, not answer-bearing content.

Each passage is identified by a stable global id of the form

    <file_id>/<page_id>:p_<block_id_4digit>

so an LLM can pin a passage across tool calls without re-discovery,
and the proof gate's ScanCover can compose passage-level scans
underneath a parent page cover.

Build / read split (no lazy build on read):

* :meth:`build` walks the file's :class:`PageAsset` ``layout_blocks``,
  derives ``parent_section_id`` from an :class:`InventoryStore`,
  and persists the result under
  ``STORAGE_PATH/inventory_atoms/passages/<file_id>.json``. Called
  once during ingest, not on the read path.
* :meth:`passages_for_file` ONLY loads from disk. If the cache is
  missing the method raises :class:`PassageCacheMissing` so callers
  see a clear "not indexed yet" signal instead of silent recompute
  during agent loops.

Page-count tripwire: the cache stores ``page_count`` from build
time; a re-ingest that adds/drops pages forces the next ``build``
call to overwrite (loaders see the mismatch and raise the same
missing error so the caller knows to re-ingest).
"""
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from config.settings import inventory_atoms_root
from storage.page_store import PageAsset, PageStore

if TYPE_CHECKING:
    from storage.inventory_store import InventoryStore


logger = logging.getLogger(__name__)


# Block labels we treat as content-bearing prose. Header/footer/etc.
# are decoration; image/table are tracked by their own stores.
_PASSAGE_LABELS: frozenset = frozenset({"text", "paragraph_title"})

# Cache version tag — bump on shape change so stale caches rebuild.
_PASSAGE_VERSION = 1


@dataclass(frozen=True)
class Passage:
    """A passage within a file's page layout."""

    passage_id: str           # "<file_id>/<page_id>:p_<NNNN>"
    file_id: str
    page_id: str
    page_number: Optional[int]
    block_id: int
    block_label: str
    text: str
    parent_section_id: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Passage":
        return cls(
            passage_id=str(data["passage_id"]),
            file_id=str(data["file_id"]),
            page_id=str(data["page_id"]),
            page_number=data.get("page_number"),
            block_id=int(data.get("block_id", 0)),
            block_label=str(data.get("block_label", "text")),
            text=str(data.get("text", "")),
            parent_section_id=data.get("parent_section_id"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "file_id": self.file_id,
            "page_id": self.page_id,
            "page_number": self.page_number,
            "block_id": self.block_id,
            "block_label": self.block_label,
            "text": self.text,
            "parent_section_id": self.parent_section_id,
        }


def make_passage_id(file_id: str, page_id: str, block_id: int) -> str:
    return f"{file_id}/{page_id}:p_{block_id:04d}"


class PassageCacheMissing(LookupError):
    """Raised when a caller asks for passages of a file whose cache
    has not been built yet (or whose page_count tripwire fired).
    The caller should run :meth:`PassageStore.build` for that file
    — usually during ingest — before reading.
    """


class PassageStore:
    """Persistent passage index keyed by passage_id. Read-only on the
    query path; :meth:`build` is the only writer and is invoked from
    the ingest pipeline after a file's page assets are persisted.
    """

    def __init__(
        self,
        page_store: PageStore,
        *,
        atoms_dir: Optional[Path] = None,
        inventory: Optional["InventoryStore"] = None,
    ):
        self.page_store = page_store
        self.atoms_dir = Path(atoms_dir) if atoms_dir else inventory_atoms_root("passages")
        self._inventory = inventory   # used only for parent_section_id at build()
        self._by_file: Dict[str, List[Passage]] = {}
        self._by_id: Dict[str, Passage] = {}
        self._lock = threading.Lock()

    # ----------------------------------------------------------- read

    def passages_for_file(self, file_id: str) -> List[Passage]:
        """Return cached passages for ``file_id``. Raises
        :class:`PassageCacheMissing` if no cache exists or if the
        cache's page_count tripwire is stale (re-ingested file).
        """
        with self._lock:
            cached = self._by_file.get(file_id)
            if cached is not None:
                return list(cached)
            pages = self._iter_pages(file_id)
            disk = self._load_cache(file_id, expected_page_count=len(pages))
            if disk is None:
                raise PassageCacheMissing(
                    f"passage cache for file_id={file_id!r} not built; "
                    "run PassageStore.build(file_id) during ingest first."
                )
            self._by_file[file_id] = disk
            for p in disk:
                self._by_id[p.passage_id] = p
            return list(disk)

    def passages_for_page(self, file_id: str, page_id: str) -> List[Passage]:
        return [p for p in self.passages_for_file(file_id) if p.page_id == page_id]

    def get(self, passage_id: str) -> Optional[Passage]:
        with self._lock:
            cached = self._by_id.get(passage_id)
        if cached is not None:
            return cached
        if "/" not in passage_id:
            return None
        file_id, _ = passage_id.split("/", 1)
        try:
            self.passages_for_file(file_id)
        except PassageCacheMissing:
            return None
        return self._by_id.get(passage_id)

    # ----------------------------------------------------------- write

    def build(self, file_id: str, *, force: bool = True) -> List[Passage]:
        """Walk the file's pages, extract every content-bearing layout
        block, persist the result, and update the in-memory index.

        ``force=True`` (default for ingest callers) ALWAYS rebuilds
        and overwrites the cache. Page-count parity is too weak a
        tripwire for re-ingest: a re-OCR can keep the page count
        stable but change cell content, and we'd silently certify
        over the old universe. Pass ``force=False`` only from a
        warmup script that wants to skip already-built files.
        """
        with self._lock:
            pages = self._iter_pages(file_id)
            sorted_pages = sorted(
                pages, key=lambda p: (p.page_number or 0, p.page_id),
            )
            if not force:
                disk = self._load_cache(file_id, expected_page_count=len(pages))
                if disk is not None:
                    self._by_file[file_id] = disk
                    for p in disk:
                        self._by_id[p.passage_id] = p
                    return list(disk)
            passages = list(self._build(file_id, sorted_pages))
            self._persist(file_id, passages, page_count=len(pages))
            self._by_file[file_id] = passages
            for p in passages:
                self._by_id[p.passage_id] = p
            return list(passages)

    # ----------------------------------------------------------- internals

    def _iter_pages(self, file_id: str) -> List[PageAsset]:
        prefix = f"{file_id}/"
        out: List[PageAsset] = []
        for gid in self.page_store.ids():
            if not gid.startswith(prefix):
                continue
            page = self.page_store.get(gid)
            if page is not None:
                out.append(page)
        return out

    def _build(self, file_id: str, pages: List[PageAsset]) -> List[Passage]:
        out: List[Passage] = []
        for page in pages:
            blocks = page.layout_blocks or []
            for block in blocks:
                label = (block.get("block_label") or block.get("block_type") or "").lower()
                if label not in _PASSAGE_LABELS:
                    continue
                content = block.get("block_content") or block.get("content") or ""
                text = str(content).strip()
                if not text:
                    continue
                try:
                    block_id = int(
                        block.get("block_id")
                        if block.get("block_id") is not None
                        else block.get("block_order", 0)
                    )
                except (TypeError, ValueError):
                    block_id = len(out)
                pid = make_passage_id(file_id, page.page_id, block_id)
                parent_sid = self._parent_section(file_id, page) if self._inventory else None
                out.append(Passage(
                    passage_id=pid,
                    file_id=file_id,
                    page_id=page.page_id,
                    page_number=page.page_number,
                    block_id=block_id,
                    block_label=label,
                    text=text,
                    parent_section_id=parent_sid,
                ))
        return out

    def _parent_section(self, file_id: str, page: PageAsset) -> Optional[str]:
        if self._inventory is None or page.page_number is None:
            return None
        # Late lookup — inventory was injected separately.
        for sec in self._inventory.sections_for_file(file_id):
            if sec.page_start <= page.page_number <= sec.page_end:
                return sec.section_id
        return None

    # ----------------------------------------------------------- persistence

    def _cache_path(self, file_id: str) -> Path:
        return self.atoms_dir / f"{file_id}.json"

    def _load_cache(
        self, file_id: str, expected_page_count: int,
    ) -> Optional[List[Passage]]:
        cache_path = self._cache_path(file_id)
        if not cache_path.is_file():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if data.get("version") != _PASSAGE_VERSION:
            return None
        if data.get("page_count") != expected_page_count:
            return None
        try:
            return [Passage.from_dict(d) for d in data.get("passages", [])]
        except (KeyError, ValueError, TypeError):
            return None

    def _persist(
        self, file_id: str, passages: List[Passage], *, page_count: int,
    ) -> None:
        try:
            self.atoms_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("PassageStore: failed to create %s: %s", self.atoms_dir, exc)
            return
        payload = {
            "version": _PASSAGE_VERSION,
            "file_id": file_id,
            "page_count": page_count,
            "passages": [p.to_dict() for p in passages],
        }
        path = self._cache_path(file_id)
        # Atomic write: temp file + rename. Otherwise a crash mid-
        # write leaves a half-JSON cache that the read path then
        # has to error on. ``os.replace`` is atomic on POSIX and on
        # Windows when the destination exists.
        import os
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        except OSError as exc:
            logger.warning("PassageStore: failed to write %s: %s", path, exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
