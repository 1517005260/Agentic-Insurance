"""Persistent per-file structural inventory.

For now the only inventory type is **sections** — page-spans derived from
the Markdown heading structure that PaddleOCR emits. Section IDs are
globally-unique strings of the form ``"<file_id>:sec_NNN"`` where ``N``
is the heading's document-order rank, so an agent can quote the same id
across tool calls without re-discovery.

The store lazily builds a per-file inventory on first request and
persists it under ``STORAGE_PATH/inventory/<file_id>.json``. Re-ingest
of a file overwrites the cache; mid-process changes are NOT watched —
construct a fresh ``InventoryStore`` if the underlying ``PageStore`` is
swapped out.

This module is intentionally narrow. ``DocumentInventory`` per
``docs/algorithm.md`` §1.2 is broader (entities, tables, clauses,
terms, …); we add types here as later phases need them.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import inventory_root
from storage.page_store import PageAsset, PageStore


logger = logging.getLogger(__name__)


_INVENTORY_VERSION = 1
# CommonMark allows up to 3 leading spaces before an ATX heading. Beyond
# that the line is an indented code block, not a heading. PaddleOCR
# emits flush-left headings in practice but cheap correctness here
# costs nothing and protects against quirky parses.
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*$")
_FENCE_PREFIXES = ("```", "~~~")


@dataclass(frozen=True)
class Section:
    """A page-span derived from a single Markdown heading."""

    section_id: str  # "<file_id>:sec_NNN"
    file_id: str
    title: str
    depth: int
    page_start: int
    page_end: int
    parent_section_id: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Section":
        return cls(
            section_id=str(data["section_id"]),
            file_id=str(data["file_id"]),
            title=str(data.get("title", "")),
            depth=int(data["depth"]),
            page_start=int(data["page_start"]),
            page_end=int(data["page_end"]),
            parent_section_id=data.get("parent_section_id"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "file_id": self.file_id,
            "title": self.title,
            "depth": self.depth,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "parent_section_id": self.parent_section_id,
        }


class InventoryStore:
    """Lazy + persistent section inventory keyed by section_id.

    Construction is cheap — sections are loaded the first time a file is
    referenced, either from the ``inventory/<file_id>.json`` cache or
    derived from the ``PageStore``. A single internal lock serialises
    builders / cache writes; the class is thread-safe under the common
    read pattern (one agent loop, parallel tool fan-out) but does not
    attempt per-file fairness — concurrent first-time loads of
    different files briefly serialise on each other.
    """

    def __init__(
        self,
        page_store: PageStore,
        inventory_dir: Optional[Path] = None,
    ):
        self.page_store = page_store
        self.inventory_dir = Path(inventory_dir) if inventory_dir else inventory_root()
        self._sections_by_file: Dict[str, List[Section]] = {}
        self._section_by_id: Dict[str, Section] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- public API

    def sections_for_file(self, file_id: str) -> List[Section]:
        """Return all sections for ``file_id`` (empty list if no headings).

        For an unknown ``file_id`` (no pages indexed) we return ``[]``
        without persisting anything — a typo'd lookup must not leak an
        empty cache file under a bogus name.

        The on-disk cache stores ``page_count`` alongside the sections;
        if a re-ingest changed the page count the cache is rejected and
        the inventory is rebuilt. Same-page-count edits (rare: same
        page-set, different headings) still hit the stale cache —
        users who need a guaranteed-fresh inventory should delete the
        per-file cache file.
        """
        with self._lock:
            cached = self._sections_by_file.get(file_id)
            if cached is not None:
                return cached
            pages = self._pages_for(file_id)
            if not pages:
                # Don't memoize the empty result — a later ingest of
                # this file_id should be picked up on the next call.
                return []
            disk = self._load_cache(file_id, expected_page_count=len(pages))
            if disk is not None:
                sections = disk
            else:
                last_page = pages[-1].page_number or len(pages)
                headings = _collect_headings(pages)
                sections = _resolve_spans(file_id, headings, last_page=last_page)
                self._persist(file_id, sections, page_count=len(pages))
            self._sections_by_file[file_id] = sections
            for s in sections:
                self._section_by_id[s.section_id] = s
            return sections

    def get(self, section_id: str) -> Optional[Section]:
        """Resolve a section_id to its :class:`Section`, or ``None``."""
        with self._lock:
            cached = self._section_by_id.get(section_id)
        if cached is not None:
            return cached
        # IDs have the form ``<file_id>:sec_NNN``. Pull off the prefix so
        # we can hydrate the right file without scanning the whole store.
        file_id, sep, _ = section_id.rpartition(":")
        if not sep or not file_id:
            return None
        self.sections_for_file(file_id)
        with self._lock:
            return self._section_by_id.get(section_id)

    def warm_up(self) -> Dict[str, int]:
        """Force-load every file's sections.

        Returns ``{file_id: section_count}``. The agent's ``warm_up``
        hook calls this so first-query latency is bounded — building
        sections fresh for a 30-page Markdown file is sub-millisecond,
        but we still want the persisted file to exist when the first
        retrieval comes in (it's the contract for ``section_ids``).
        """
        out: Dict[str, int] = {}
        for gid in self.page_store.ids():
            file_id = gid.split("/", 1)[0]
            if file_id and file_id not in out:
                out[file_id] = len(self.sections_for_file(file_id))
        return out

    # ------------------------------------------------------------- internals

    def _cache_path(self, file_id: str) -> Path:
        return self.inventory_dir / f"{file_id}.json"

    def _load_cache(
        self, file_id: str, expected_page_count: int
    ) -> Optional[List[Section]]:
        """Load ``inventory/<file_id>.json``; reject if version or
        ``page_count`` doesn't match the current PageStore snapshot.

        ``page_count`` is a cheap re-ingest tripwire: a re-parse that
        adds or drops pages will update the stored count, which then
        mismatches the cached value and triggers a rebuild. Same-count
        re-parses still hit a stale cache (acceptable for v1 — caller
        can manually delete the file).
        """
        path = self._cache_path(file_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("InventoryStore: cache for %s is unreadable (%s); rebuilding", file_id, exc)
            return None
        if data.get("version") != _INVENTORY_VERSION:
            return None
        # Missing page_count means a pre-tripwire cache file: treat as
        # stale so the rebuild populates the new field. Mismatch means
        # a re-ingest changed the page set and the cached spans are no
        # longer authoritative.
        cached_page_count = data.get("page_count")
        if cached_page_count is None or cached_page_count != expected_page_count:
            logger.info(
                "InventoryStore: cache for %s is stale (page_count %s != %s); rebuilding",
                file_id, cached_page_count, expected_page_count,
            )
            return None
        return [Section.from_dict(s) for s in data.get("sections", [])]

    def _persist(self, file_id: str, sections: List[Section], page_count: int) -> None:
        try:
            self.inventory_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": _INVENTORY_VERSION,
                "file_id": file_id,
                "page_count": page_count,
                "sections": [s.to_dict() for s in sections],
            }
            self._cache_path(file_id).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            # Persistence is best-effort — a sandboxed read-only test
            # should still get correct results from the in-memory cache.
            logger.warning("InventoryStore: failed to persist %s: %s", file_id, exc)

    def _pages_for(self, file_id: str) -> List[PageAsset]:
        out: List[PageAsset] = []
        for gid in self.page_store.ids():
            if not gid.startswith(file_id + "/"):
                continue
            page = self.page_store.get(gid)
            if page is None or page.file_id != file_id:
                continue
            out.append(page)
        out.sort(key=lambda p: (p.page_number or 0, p.page_id))
        return out


# --------------------------------------------------------------------- helpers


def _collect_headings(pages: List[PageAsset]) -> List[Tuple[int, int, str]]:
    """Walk every page in document order, yielding ``(page_no, depth, title)``.

    Headings inside fenced code blocks are skipped — Markdown spec says
    a ``#`` inside ``````` is not a heading, and PaddleOCR
    occasionally emits Python comment listings that would otherwise
    confuse the section extractor.
    """
    out: List[Tuple[int, int, str]] = []
    for page in pages:
        text = page.text_markdown or ""
        if "#" not in text:
            continue
        page_no = page.page_number if page.page_number is not None else 0
        in_fence = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(_FENCE_PREFIXES):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = _HEADING_RE.match(line)
            if not m:
                continue
            depth = len(m.group(1))
            title = m.group(2).strip()
            if title:
                out.append((page_no, depth, title))
    return out


def _resolve_spans(
    file_id: str,
    headings: List[Tuple[int, int, str]],
    last_page: int,
) -> List[Section]:
    """Close each heading at the next equal-or-shallower one (or EOF).

    ``parent_section_id`` is threaded via a depth stack so the inventory
    surfaces the heading tree the agent's ``toc`` tool already computes.
    Section IDs include the file_id so they survive round-trips through
    tool arguments without an extra ``file_id`` field.
    """
    if not headings:
        return []
    sections: List[Section] = []
    stack: List[Tuple[int, str]] = []
    for i, (page_no, depth, title) in enumerate(headings):
        section_id = f"{file_id}:sec_{i+1:03d}"
        while stack and stack[-1][0] >= depth:
            stack.pop()
        parent = stack[-1][1] if stack else None
        end = last_page
        for j in range(i + 1, len(headings)):
            np, nd, _ = headings[j]
            if nd <= depth:
                end = max(page_no, np - 1) if np > page_no else page_no
                break
        sections.append(
            Section(
                section_id=section_id,
                file_id=file_id,
                title=title,
                depth=depth,
                page_start=page_no,
                page_end=end,
                parent_section_id=parent,
            )
        )
        stack.append((depth, section_id))
    return sections
