"""Persistent per-file structural inventory.

The inventory tracks **sections** — page-spans derived from the Markdown
heading structure that PaddleOCR emits. Section IDs are globally-unique
strings of the form ``"<file_id>:sec_NNN"`` where ``N`` is the heading's
document-order rank, so an agent can quote the same id across tool calls
without re-discovery.

Each section also carries a ``provenance`` flag indicating how the
boundary was derived (``toc_explicit`` from a real PDF outline,
``heading_extracted`` from the OCR markdown structure, or
``heuristic_split`` from a fallback split). ``confidence`` follows from
provenance and gates which proof obligations may close at section level.
``is_page_exclusive`` records whether the section owns at least one page
no other section covers — closure rules use it to refuse certifying a
section-level scan when boundaries are ambiguous.

The store lazily builds a per-file inventory on first request and
persists it under ``STORAGE_PATH/inventory/<file_id>.json``. Re-ingest
of a file overwrites the cache; mid-process changes are NOT watched —
construct a fresh ``InventoryStore`` if the underlying ``PageStore`` is
swapped out.

This module is intentionally narrow. Richer document units (entities,
tables, clauses, terms) live in their own stores; this one is the
file/section axis only.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from config.settings import inventory_root
from storage.page_store import PageAsset, PageStore


logger = logging.getLogger(__name__)


# Cached inventories tag themselves with this version; a mismatch forces
# rebuild so callers never read caches that lack fields the current
# Section dataclass expects.
_INVENTORY_VERSION = 3
# CommonMark allows up to 3 leading spaces before an ATX heading. Beyond
# that the line is an indented code block, not a heading. PaddleOCR
# emits flush-left headings in practice but cheap correctness here
# costs nothing and protects against quirky parses.
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*$")
_FENCE_PREFIXES = ("```", "~~~")


Provenance = Literal["toc_explicit", "heading_extracted", "heuristic_split"]
Confidence = Literal["high", "medium", "low"]
UnitType = Literal["file", "section", "page", "passage", "table_row"]
# ``passage`` and ``table_row`` use sibling stores; the others come from
# this inventory plus the underlying page_store directly.
_WIRED_UNIT_TYPES: frozenset = frozenset(
    {"file", "section", "page", "passage", "table_row"}
)


# Provenance → confidence mapping. Confidence is a derived field so we
# don't accept user overrides; a low-confidence section that "feels"
# high to a caller still must not close completeness obligations.
_PROVENANCE_CONFIDENCE: Dict[Provenance, Confidence] = {
    "toc_explicit": "high",
    "heading_extracted": "medium",
    "heuristic_split": "low",
}


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
    provenance: Provenance
    confidence: Confidence
    is_page_exclusive: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Section":
        provenance = data.get("provenance") or "heading_extracted"
        confidence = data.get("confidence") or _PROVENANCE_CONFIDENCE[provenance]
        return cls(
            section_id=str(data["section_id"]),
            file_id=str(data["file_id"]),
            title=str(data.get("title", "")),
            depth=int(data["depth"]),
            page_start=int(data["page_start"]),
            page_end=int(data["page_end"]),
            parent_section_id=data.get("parent_section_id"),
            provenance=provenance,  # type: ignore[arg-type]
            confidence=confidence,  # type: ignore[arg-type]
            is_page_exclusive=bool(data.get("is_page_exclusive", False)),
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
            "provenance": self.provenance,
            "confidence": self.confidence,
            "is_page_exclusive": self.is_page_exclusive,
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
        *,
        atoms_dir: Optional[Path] = None,
    ):
        self.page_store = page_store
        self.inventory_dir = Path(inventory_dir) if inventory_dir else inventory_root()
        self._sections_by_file: Dict[str, List[Section]] = {}
        self._section_by_id: Dict[str, Section] = {}
        # Sibling stores for sub-page units. Constructed eagerly here
        # (cheap: just a wrapper around page_store), but read-only on
        # the query path — caches are written exclusively by ingest
        # via :func:`build_inventory_atoms_for_file`. ``atoms_dir``
        # defaults to ``<STORAGE_PATH>/inventory_atoms``; callers
        # (notably tests) can override to scope caches into a
        # workdir.
        from storage.passage_store import PassageStore
        from storage.table_row_store import TableRowStore
        passage_dir = Path(atoms_dir) / "passages" if atoms_dir else None
        table_row_dir = Path(atoms_dir) / "table_rows" if atoms_dir else None
        self.passage_store = PassageStore(
            page_store, inventory=self, atoms_dir=passage_dir,
        )
        self.table_row_store = TableRowStore(
            page_store, inventory=self, atoms_dir=table_row_dir,
        )
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

    # ----------------------------------------------------- proof-layer API

    def section_for_page(self, page_global_id: str) -> Optional[str]:
        """Map a page global_id (``<file_id>/<page_id>``) to the
        deepest section whose span contains the page.

        Returns ``None`` when the page is in an unknown file or falls
        outside every section span (e.g., a cover page before any heading).
        Used by auto-extractors to aggregate page-level pattern hits up
        to section-level ScanClaims.
        """
        file_id, sep, page_id = page_global_id.partition("/")
        if not sep:
            return None
        page = self.page_store.get(page_global_id)
        if page is None:
            return None
        page_no = page.page_number
        if page_no is None:
            return None
        sections = self.sections_for_file(file_id)
        if not sections:
            return None
        # Pick the deepest containing section so a leaf wins over its
        # parent in nested headings.
        best: Optional[Section] = None
        for s in sections:
            if s.page_start <= page_no <= s.page_end:
                if best is None or s.depth > best.depth:
                    best = s
        return best.section_id if best is not None else None

    def units(
        self,
        unit_type: UnitType,
        *,
        file_ids: Optional[List[str]] = None,
        section_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Enumerate the addressable unit ids inside a scope.

        For ``unit_type="file"`` the result is the supplied ``file_ids``
        (or the union of files that own ``section_ids``).
        For ``unit_type="section"`` the result is the section ids
        contained in the resolved scope. ``section_ids`` is preserved
        verbatim if supplied; otherwise we expand each ``file_id`` to
        all of its sections.

        Closure rules use this to compute the universe a ScanClaim must
        partition. Returning a stable, sorted list keeps the partition
        diff cheap for diagnostics.
        """
        if unit_type not in _WIRED_UNIT_TYPES:
            raise ValueError(
                f"unit_type must be one of {sorted(_WIRED_UNIT_TYPES)!r}; "
                f"got {unit_type!r}"
            )
        if unit_type == "file":
            if file_ids:
                return sorted({f for f in file_ids if f})
            if section_ids:
                derived = set()
                for sid in section_ids:
                    sec = self.get(sid)
                    if sec is not None:
                        derived.add(sec.file_id)
                return sorted(derived)
            return []
        if unit_type == "page":
            # Page-level enumeration. ``file_ids`` and ``section_ids``
            # gate which pages we return; section_ids dominate when
            # both are supplied (section_ids ∩ file_ids is implied).
            if section_ids:
                pages: List[str] = []
                for sid in section_ids:
                    sec = self.get(sid)
                    if sec is None:
                        continue
                    for p in range(sec.page_start, sec.page_end + 1):
                        gid = f"{sec.file_id}/p_{p:04d}"
                        if self.page_store.get(gid) is not None:
                            pages.append(gid)
                return sorted(set(pages))
            if not file_ids:
                return []
            allowed = set(file_ids)
            return sorted(
                gid for gid in self.page_store.ids()
                if "/" in gid and gid.split("/", 1)[0] in allowed
            )
        if unit_type == "passage":
            return self._enumerate_atoms(
                self.passage_store, "passage_id",
                file_ids=file_ids, section_ids=section_ids,
            )
        if unit_type == "table_row":
            return self._enumerate_atoms(
                self.table_row_store, "table_row_id",
                file_ids=file_ids, section_ids=section_ids,
            )
        # unit_type == "section"
        if section_ids:
            cleaned = sorted({s for s in section_ids if s})
            return [s for s in cleaned if self.get(s) is not None]
        out: List[str] = []
        for fid in file_ids or []:
            for s in self.sections_for_file(fid):
                out.append(s.section_id)
        return sorted(out)

    def _enumerate_atoms(
        self,
        store: Any,
        id_attr: str,
        *,
        file_ids: Optional[List[str]],
        section_ids: Optional[List[str]],
    ) -> List[str]:
        """Shared enumeration for passage / table_row stores.

        ``section_ids`` dominates ``file_ids`` (section_ids ∩ file
        is implied via the section's owning file). Each atom is
        included iff its ``parent_section_id`` matches one of the
        requested sections OR (when no section filter) its file
        matches one of ``file_ids``.
        """
        if section_ids:
            section_set = {sid for sid in section_ids if sid}
            sec_files = set()
            for sid in section_set:
                sec = self.get(sid)
                if sec is not None:
                    sec_files.add(sec.file_id)
            out: List[str] = []
            for fid in sec_files:
                for atom in store.passages_for_file(fid) if id_attr == "passage_id" else store.rows_for_file(fid):
                    if atom.parent_section_id in section_set:
                        out.append(getattr(atom, id_attr))
            return sorted(out)
        if not file_ids:
            return []
        out: List[str] = []
        for fid in file_ids:
            atoms = (
                store.passages_for_file(fid) if id_attr == "passage_id"
                else store.rows_for_file(fid)
            )
            out.extend(getattr(a, id_attr) for a in atoms)
        return sorted(out)

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
        re-parses still hit a stale cache; the caller can delete the
        file manually to force a rebuild.
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
    tool arguments without an extra ``file_id`` field. After spans are
    resolved we compute :func:`_compute_page_exclusivity` so leaves and
    siblings carry an authoritative ``is_page_exclusive`` flag rather
    than a per-call recomputation.
    """
    if not headings:
        return []
    raw: List[Tuple[int, int, str, int, Optional[str]]] = []  # (start, depth, title, end, parent)
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
        raw.append((page_no, depth, title, end, parent))
        stack.append((depth, section_id))

    # We need is_page_exclusive — does any non-ancestor section overlap
    # this section's pages? Compute pairwise so the answer is stable
    # regardless of traversal order.
    exclusivity = _compute_page_exclusivity(raw, file_id)

    return [
        Section(
            section_id=f"{file_id}:sec_{i+1:03d}",
            file_id=file_id,
            title=title,
            depth=depth,
            page_start=start,
            page_end=end,
            parent_section_id=parent,
            provenance="heading_extracted",
            confidence=_PROVENANCE_CONFIDENCE["heading_extracted"],
            is_page_exclusive=exclusivity[i],
        )
        for i, (start, depth, title, end, parent) in enumerate(raw)
    ]


def _compute_page_exclusivity(
    raw: List[Tuple[int, int, str, int, Optional[str]]],
    file_id: str,
) -> List[bool]:
    """``is_page_exclusive(s)`` iff *no* page in ``s.page_range`` is
    covered by any other section in the same file.

    This is the property section-level scan certificates require: any
    overlap with a sibling or descendant means a page-hit cannot be
    unambiguously attributed to a single section, so the section is
    unsafe as a closure unit. Nested headings — where a parent's range
    contains a child's range — therefore yield a non-exclusive parent,
    by design.

    Implementation: difference-array sweep. ``counts[p]`` = number of
    sections covering page p; build by ``+1`` at start and ``-1`` at
    end+1 then prefix-sum. Section i is exclusive iff every page in its
    span has count exactly 1 (just itself). O(n + max_page) — drops the
    original O(n²) double loop while staying allocation-light for
    documents with thousands of sections.
    """
    if not raw:
        return []
    max_page = max(end for _, _, _, end, _ in raw)
    counts: List[int] = [0] * (max_page + 2)
    for start, _, _, end, _ in raw:
        counts[start] += 1
        counts[end + 1] -= 1
    running = 0
    for p in range(len(counts)):
        running += counts[p]
        counts[p] = running
    out: List[bool] = []
    for start, _, _, end, _ in raw:
        exclusive = True
        for p in range(start, end + 1):
            if counts[p] != 1:
                exclusive = False
                break
        out.append(exclusive)
    return out
