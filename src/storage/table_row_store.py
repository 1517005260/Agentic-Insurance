"""Per-file table-row inventory.

A *table row* is a single ``<tr>`` element extracted from one of a
page's rendered HTML tables. The HTML lives on layout blocks whose
``block_label == 'table'`` (PaddleOCR PP-StructureV3) and on
``PageAsset.table_blocks[].html`` (PP-Structure v2). Each row gets a
stable global id

    <file_id>/<page_id>:t_<table_index>:r_<row_index>

so a ScanCover over ``unit_type='table_row'`` can name precisely
which rows are positive / negative for a predicate.

Build / read split (no lazy build on read): the ingest pipeline
calls :meth:`build` after a file's page assets are persisted;
:meth:`rows_for_file` only loads from the on-disk cache and raises
:class:`TableRowCacheMissing` otherwise. HTML parsing uses
stdlib :mod:`html.parser` (no third-party dep).
"""
import json
import logging
import re
import threading
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from config.settings import inventory_atoms_root
from storage.page_store import PageAsset, PageStore

if TYPE_CHECKING:
    from storage.inventory_store import InventoryStore


logger = logging.getLogger(__name__)


_TABLE_ROW_VERSION = 2


@dataclass(frozen=True)
class TableRow:
    """A single row inside a page's HTML table."""

    table_row_id: str         # "<file_id>/<page_id>:t_<NN>:r_<NN>"
    file_id: str
    page_id: str
    page_number: Optional[int]
    table_index: int
    row_index: int
    html: str
    text: str
    is_header_row: bool
    parent_section_id: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableRow":
        return cls(
            table_row_id=str(data["table_row_id"]),
            file_id=str(data["file_id"]),
            page_id=str(data["page_id"]),
            page_number=data.get("page_number"),
            table_index=int(data.get("table_index", 0)),
            row_index=int(data.get("row_index", 0)),
            html=str(data.get("html", "")),
            text=str(data.get("text", "")),
            is_header_row=bool(data.get("is_header_row", False)),
            parent_section_id=data.get("parent_section_id"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_row_id": self.table_row_id,
            "file_id": self.file_id,
            "page_id": self.page_id,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "row_index": self.row_index,
            "html": self.html,
            "text": self.text,
            "is_header_row": self.is_header_row,
            "parent_section_id": self.parent_section_id,
        }


def make_table_row_id(file_id: str, page_id: str, table_index: int, row_index: int) -> str:
    return f"{file_id}/{page_id}:t_{table_index:02d}:r_{row_index:02d}"


# ---------------------------------------------------------------- HTML

class _RowExtractor(HTMLParser):
    """Pull every ``<tr>`` (and its inner cells) out of a table HTML
    string. We stay inside ``html.parser`` so the store has zero
    third-party deps; PaddleOCR's table HTML is well-formed enough
    that a SAX-style sweep is sufficient."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[Dict[str, Any]] = []
        self._depth_in_tr = 0
        self._current_html: List[str] = []
        self._current_text: List[str] = []
        self._cell_kinds: List[str] = []
        self._in_cell = False
        self._current_cell_kind: Optional[str] = None

    def handle_starttag(self, tag: str, attrs):
        if tag == "tr":
            self._depth_in_tr += 1
            if self._depth_in_tr == 1:
                self._current_html = ["<tr>"]
                self._current_text = []
                self._cell_kinds = []
            return
        if self._depth_in_tr == 0:
            return
        attr_str = "".join(f' {k}="{v}"' for k, v in attrs if v is not None)
        self._current_html.append(f"<{tag}{attr_str}>")
        if tag in ("th", "td"):
            self._in_cell = True
            self._current_cell_kind = tag
            self._cell_kinds.append(tag)

    def handle_endtag(self, tag: str):
        if tag == "tr":
            if self._depth_in_tr == 1:
                self._current_html.append("</tr>")
                row_html = "".join(self._current_html)
                row_text = " ".join(t.strip() for t in self._current_text if t.strip())
                kinds = list(self._cell_kinds)
                is_header = bool(kinds) and all(k == "th" for k in kinds)
                self.rows.append({
                    "html": row_html,
                    "text": row_text,
                    "is_header_row": is_header,
                })
            self._depth_in_tr = max(0, self._depth_in_tr - 1)
            return
        if self._depth_in_tr == 0:
            return
        self._current_html.append(f"</{tag}>")
        if tag in ("th", "td"):
            self._in_cell = False
            self._current_cell_kind = None

    def handle_data(self, data: str):
        if self._depth_in_tr == 0:
            return
        self._current_html.append(data)
        if self._in_cell:
            self._current_text.append(data)


def _extract_rows(html: str) -> List[Dict[str, Any]]:
    if not html:
        return []
    parser = _RowExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:
        logger.warning("TableRowStore: HTML parse failed: %s", exc)
        return []
    return parser.rows


# ---------------------------------------------------------------- Markdown
#
# Markdown's table syntax is GitHub-flavored:
#
#     | Col A | Col B | Col C |
#     | :---  | :---: | ----: |
#     | a     | b     | c     |
#     | d     | e     | f     |
#
# Variations we handle:
#   - leading/trailing pipes optional;
#   - alignment markers ``:---`` / ``:---:`` / ``---:``;
#   - escaped pipes inside cells (``\|``);
#   - whitespace-padded cells;
#   - inline HTML inside a cell (preserved verbatim in the cell text);
#   - tables embedded inside fenced code blocks are skipped.
#
# We synthesise a normalised ``<tr><th|td>...</tr>`` HTML for the row
# so downstream cell-aware predicates see one shape regardless of
# the source format.

_MD_SEP_RE = re.compile(
    r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$"
)
_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")


def _split_md_row(line: str) -> List[str]:
    """Split a Markdown row line into trimmed cell strings, honouring
    escaped pipes (``\\|``). Leading/trailing empty cells produced by
    optional outer pipes are dropped."""
    placeholder = "\x00"
    masked = line.replace(r"\|", placeholder)
    cells = masked.split("|")
    if cells and cells[0].strip() == "":
        cells = cells[1:]
    if cells and cells[-1].strip() == "":
        cells = cells[:-1]
    return [c.strip().replace(placeholder, "|") for c in cells]


def _row_to_html(cells: List[str], is_header: bool) -> str:
    """Synthesise ``<tr>`` HTML from Markdown cells. Cell content is
    escaped — a Markdown cell containing a literal ``<tr>`` token
    must become the text ``&lt;tr&gt;``, not new HTML structure.
    Tag name and attribute escaping aren't needed because we control
    the tag set (th/td)."""
    import html as _htmllib
    tag = "th" if is_header else "td"
    inner = "".join(
        f"<{tag}>{_htmllib.escape(c, quote=False)}</{tag}>" for c in cells
    )
    return f"<tr>{inner}</tr>"


def _extract_md_rows(md_text: str) -> List[Dict[str, Any]]:
    """Find every Markdown table in ``md_text`` and return its rows
    in document order, normalised into the same shape HTML extraction
    yields. Tables inside fenced code blocks are skipped — they are
    documentation, not real data."""
    if not md_text or "|" not in md_text:
        return []
    lines = md_text.splitlines()
    out: List[Dict[str, Any]] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            i += 1
            continue
        if in_fence or "|" not in line:
            i += 1
            continue
        # Header candidate: this line has pipes AND the next is a
        # separator. GFM requires both — without the separator a
        # bare pipe line is just prose.
        if i + 1 >= len(lines) or not _MD_SEP_RE.match(lines[i + 1]):
            i += 1
            continue
        header_cells = _split_md_row(line)
        if not header_cells:
            i += 1
            continue
        out.append({
            "html": _row_to_html(header_cells, is_header=True),
            "text": " ".join(c for c in header_cells if c),
            "is_header_row": True,
        })
        j = i + 2
        while j < len(lines):
            nxt = lines[j]
            stripped = nxt.strip()
            if not stripped:
                break
            if _FENCE_RE.match(stripped):
                break
            if "|" not in nxt:
                break
            data_cells = _split_md_row(nxt)
            if not data_cells:
                break
            out.append({
                "html": _row_to_html(data_cells, is_header=False),
                "text": " ".join(c for c in data_cells if c),
                "is_header_row": False,
            })
            j += 1
        i = j
    return out


# ---------------------------------------------------------------- store

class TableRowCacheMissing(LookupError):
    """Raised when a caller asks for table rows of a file whose cache
    has not been built yet (or whose page_count tripwire fired)."""


class TableRowStore:
    """Persistent table-row index keyed by table_row_id. Read-only on
    the query path; :meth:`build` is the only writer."""

    def __init__(
        self,
        page_store: PageStore,
        *,
        atoms_dir: Optional[Path] = None,
        inventory: Optional["InventoryStore"] = None,
    ):
        self.page_store = page_store
        self.atoms_dir = Path(atoms_dir) if atoms_dir else inventory_atoms_root("table_rows")
        self._inventory = inventory
        self._by_file: Dict[str, List[TableRow]] = {}
        self._by_id: Dict[str, TableRow] = {}
        self._lock = threading.Lock()

    # ----------------------------------------------------------- read

    def rows_for_file(self, file_id: str) -> List[TableRow]:
        with self._lock:
            cached = self._by_file.get(file_id)
            if cached is not None:
                return list(cached)
            pages = self._iter_pages(file_id)
            disk = self._load_cache(file_id, expected_page_count=len(pages))
            if disk is None:
                raise TableRowCacheMissing(
                    f"table_row cache for file_id={file_id!r} not built; "
                    "run TableRowStore.build(file_id) during ingest first."
                )
            self._by_file[file_id] = disk
            for r in disk:
                self._by_id[r.table_row_id] = r
            return list(disk)

    def rows_for_page(self, file_id: str, page_id: str) -> List[TableRow]:
        return [r for r in self.rows_for_file(file_id) if r.page_id == page_id]

    def get(self, table_row_id: str) -> Optional[TableRow]:
        with self._lock:
            cached = self._by_id.get(table_row_id)
        if cached is not None:
            return cached
        if "/" not in table_row_id:
            return None
        file_id, _ = table_row_id.split("/", 1)
        try:
            self.rows_for_file(file_id)
        except TableRowCacheMissing:
            return None
        return self._by_id.get(table_row_id)

    # ----------------------------------------------------------- write

    def build(self, file_id: str, *, force: bool = True) -> List[TableRow]:
        """Extract every row, persist, and refresh the in-memory index.

        ``force=True`` (the default; what ingest uses) overwrites the
        disk cache unconditionally. Page-count parity is too weak a
        tripwire for re-ingest — same page count can hide a re-OCR
        that changed every cell. ``force=False`` is for warmup
        scripts that want to skip already-cached files.
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
                    for r in disk:
                        self._by_id[r.table_row_id] = r
                    return list(disk)
            rows = list(self._build(file_id, sorted_pages))
            self._persist(file_id, rows, page_count=len(pages))
            self._by_file[file_id] = rows
            for r in rows:
                self._by_id[r.table_row_id] = r
            return list(rows)

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

    def _build(self, file_id: str, pages: List[PageAsset]) -> List[TableRow]:
        """Walk every page and harvest rows from three source kinds:

        1. ``page.table_blocks[].html`` — top-level ``tables`` array
           emitted by PP-Structure v2.
        2. ``page.layout_blocks[].block_content`` for blocks whose
           ``block_label == 'table'``. The content is HTML on most
           PP-StructureV3 outputs but Markdown-pipe syntax on others.
        3. ``page.text_markdown`` — full-page sweep that catches
           Markdown tables outside an explicit table block (e.g.
           model-emitted MD that didn't get a layout_block label).

        Format detection is heuristic: if the text contains a
        ``<table`` substring we treat it as HTML; otherwise we try
        Markdown. Cross-source dedup is by row text (normalised
        whitespace + lowercase) so an HTML table re-emitted as MD
        doesn't double-count.
        """
        out: List[TableRow] = []
        for page in pages:
            # Collect per-source row lists separately. Within a single
            # source we never dedup — two genuinely-distinct rows in
            # different tables that share text must both stay
            # enumerated. Cross-source dedup is byte-exact on the
            # row's html string only (see the harvest loop below).
            # Bag-subset dedup is avoided because it would drop a
            # legitimate source whose row-text bag happens to be a
            # subset of another source's.
            sources: List[List[Dict[str, Any]]] = []
            for tbl in (page.table_blocks or []):
                html = tbl.get("html") or ""
                rows = self._rows_from_blob(html) if html else []
                if rows:
                    sources.append(rows)
            for block in (page.layout_blocks or []):
                label = (block.get("block_label") or block.get("block_type") or "").lower()
                if label != "table":
                    continue
                content = block.get("block_content") or block.get("content") or ""
                rows = self._rows_from_blob(content) if content else []
                if rows:
                    sources.append(rows)
            if page.text_markdown and "|" in page.text_markdown:
                rows = _extract_md_rows(page.text_markdown)
                if rows:
                    sources.append(rows)
            harvested: List[Dict[str, Any]] = []
            seen_html: set[str] = set()
            for src in sources:
                for row in src:
                    h = row.get("html") or ""
                    # Byte-exact html dedup — soundness > completeness.
                    # A bag-subset heuristic could drop a genuinely
                    # distinct source whose row-text bag happens to be
                    # a subset of another, making SealClaim seal an
                    # under-enumerated row universe. Exact-html dedup
                    # keeps every distinct row, accepting at most 2x
                    # over-enumeration when PaddleOCR emits a table as
                    # both HTML and Markdown.
                    if h and h in seen_html:
                        continue
                    if h:
                        seen_html.add(h)
                    harvested.append(row)
            parent_sid = self._parent_section(file_id, page) if self._inventory else None
            # Table numbering: every row whose ``is_header_row=True``
            # or whose source had no prior rows opens a new table.
            # Subsequent data rows belong to the same table.
            table_index = -1
            row_index = 0
            for row in harvested:
                if row.get("is_header_row") or table_index < 0:
                    table_index += 1
                    row_index = 0
                rid = make_table_row_id(
                    file_id, page.page_id, table_index, row_index,
                )
                out.append(TableRow(
                    table_row_id=rid,
                    file_id=file_id,
                    page_id=page.page_id,
                    page_number=page.page_number,
                    table_index=table_index,
                    row_index=row_index,
                    html=str(row.get("html", "")),
                    text=str(row.get("text", "")),
                    is_header_row=bool(row.get("is_header_row", False)),
                    parent_section_id=parent_sid,
                ))
                row_index += 1
        return out

    @staticmethod
    def _rows_from_blob(blob: str) -> List[Dict[str, Any]]:
        """Pick HTML or Markdown extraction based on the blob's shape.
        ``<table`` substring → HTML; otherwise Markdown if the blob
        carries pipe characters."""
        lower = blob.lower()
        if "<table" in lower or "<tr" in lower:
            return _extract_rows(blob)
        if "|" in blob:
            return _extract_md_rows(blob)
        return []

    def _parent_section(self, file_id: str, page: PageAsset) -> Optional[str]:
        if self._inventory is None or page.page_number is None:
            return None
        for sec in self._inventory.sections_for_file(file_id):
            if sec.page_start <= page.page_number <= sec.page_end:
                return sec.section_id
        return None

    # ----------------------------------------------------------- persistence

    def _cache_path(self, file_id: str) -> Path:
        return self.atoms_dir / f"{file_id}.json"

    def _load_cache(
        self, file_id: str, expected_page_count: int,
    ) -> Optional[List[TableRow]]:
        cache_path = self._cache_path(file_id)
        if not cache_path.is_file():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if data.get("version") != _TABLE_ROW_VERSION:
            return None
        if data.get("page_count") != expected_page_count:
            return None
        try:
            return [TableRow.from_dict(d) for d in data.get("rows", [])]
        except (KeyError, ValueError, TypeError):
            return None

    def _persist(
        self, file_id: str, rows: List[TableRow], *, page_count: int,
    ) -> None:
        try:
            self.atoms_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("TableRowStore: failed to create %s: %s", self.atoms_dir, exc)
            return
        payload = {
            "version": _TABLE_ROW_VERSION,
            "file_id": file_id,
            "page_count": page_count,
            "rows": [r.to_dict() for r in rows],
        }
        path = self._cache_path(file_id)
        import os
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        except OSError as exc:
            logger.warning("TableRowStore: failed to write %s: %s", path, exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
