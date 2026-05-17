"""Build PageAsset records from a Double-Bench bundled ``ocr_text`` tree.

An alternative page-asset source to :mod:`ingestion.page_assets` (which
reads our own PaddleOCR ``ParseResult``). Double-Bench ships per-document
OCR under ``ocr_text/<lang>/<id>/``:

* ``text/NNN.txt``        — page ``NNN`` (0-based) text, ATX-heading markdown,
                            with ``<!-- image -->`` placeholders.
* ``table_text/NNN_M.txt`` — VLM prose describing table ``M`` on page ``NNN``.
* ``figure_text/NNN_M.txt``— VLM prose describing figure ``M`` on page ``NNN``.

The dataset carries no structured tables and no per-page raster, so:

* ``page_number`` is 1-based (``page_index + 1``) to match every repo
  contract (``Scope.page_range`` rejects ``<1``, preview subtracts one,
  citation forwards it). The Double-Bench 0-based ``reference_page`` is
  converted at the eval boundary only (``reference_page + 1``).
* ``text_markdown`` and ``layout_blocks`` are rendered from one shared
  ordered block list, so a cited span lives in both the page text and
  the passage text (the closure gate substring-matches each).
* Table / figure prose is folded into the page as ``text`` blocks
  (retrievable, graph-visible, citable). ``table_blocks`` stays empty —
  there is no structured-row source — so ``TableRowStore`` yields rows
  only from any genuine GFM pipe table already in the text.
* Real document headings render as ATX ``#``/``##`` (the inventory TOC
  scans those). Synthetic ``Figure N.M`` / ``Table N.M`` labels render
  as plain text but stay ``paragraph_title`` blocks for passage
  readability, so they do not pollute the section graph.
* ``page_image_path`` points at a per-page raster rendered from the
  image-only source PDF into ``paddle_ocr_root()/<file_id>/pages/`` so
  the vision channel and the VLM reader resolve it under their fixed
  ``paddle_ocr_root()/<file_id>/`` join.
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import paddle_ocr_root
from ingestion.page_assets import persist_and_warm_page_assets
from ingestion.page_mode import PageModeSignals, classify_page_mode
from storage.page_store import PageAsset

_PASSAGE_CHAR_BUDGET = 1200
_PAGE_IMAGE_LONG_PX = 1600
_ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}\s+\S")
_PLACEHOLDER_RE = re.compile(r"^\s*<!--\s*(?:image|formula-not-decoded)\s*-->\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CHECKPOINT = ".ipynb_checkpoints"


class _Block:
    """One ordered content unit: feeds both ``layout_blocks`` and the
    rendered ``text_markdown`` so the two never diverge."""

    __slots__ = ("label", "content", "is_atx")

    def __init__(self, label: str, content: str, is_atx: bool):
        self.label = label          # "paragraph_title" | "text"
        self.content = content
        self.is_atx = is_atx        # render as a real ATX heading line


def _numbered_txt(subdir: Path) -> List[Tuple[int, int, Path]]:
    """``NNN.txt`` / ``NNN_M.txt`` → sorted ``(page_index, item, path)``.

    ``item`` is ``-1`` for the single-page ``text/NNN.txt`` form.
    ``.ipynb_checkpoints`` pollution is skipped.
    """
    out: List[Tuple[int, int, Path]] = []
    if not subdir.is_dir():
        return out
    for p in subdir.iterdir():
        if p.name == _CHECKPOINT or p.suffix != ".txt" or not p.is_file():
            continue
        stem = p.stem
        if "_" in stem:
            a, _, b = stem.partition("_")
            try:
                out.append((int(a), int(b), p))
            except ValueError:
                continue
        else:
            try:
                out.append((int(stem), -1, p))
            except ValueError:
                continue
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _split_to_budget(text: str, budget: int = _PASSAGE_CHAR_BUDGET) -> List[str]:
    """Split prose into <=``budget``-char chunks on sentence boundaries
    (hard-splitting any single oversized sentence)."""
    text = text.strip()
    if len(text) <= budget:
        return [text] if text else []
    chunks: List[str] = []
    cur = ""
    for sent in _SENTENCE_SPLIT_RE.split(text):
        if not sent:
            continue
        if len(sent) > budget:
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(sent), budget):
                chunks.append(sent[i : i + budget])
            continue
        if cur and len(cur) + 1 + len(sent) > budget:
            chunks.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def _blocks_from_page_text(raw: str) -> List[_Block]:
    """ATX heading lines → ``paragraph_title`` (kept as ATX); blank-line
    separated paragraphs → ``text`` (budget-split). Placeholder comment
    lines are dropped."""
    blocks: List[_Block] = []
    lines = [ln for ln in raw.splitlines() if not _PLACEHOLDER_RE.match(ln)]
    para: List[str] = []

    def flush_para() -> None:
        joined = " ".join(s.strip() for s in para if s.strip()).strip()
        para.clear()
        for chunk in _split_to_budget(joined):
            blocks.append(_Block("text", chunk, is_atx=False))

    for ln in lines:
        if _ATX_HEADING_RE.match(ln):
            flush_para()
            blocks.append(_Block("paragraph_title", ln.strip(), is_atx=True))
        elif ln.strip():
            para.append(ln)
        else:
            flush_para()
    flush_para()
    return blocks


def _aux_blocks(label: str, item: int, prose: str) -> List[_Block]:
    """A synthetic ``Figure N.M`` / ``Table N.M`` label (plain-text
    ``paragraph_title`` — not ATX, so it stays out of the TOC) plus the
    budget-split VLM prose."""
    out = [_Block("paragraph_title", f"{label} {item}", is_atx=False)]
    out += [_Block("text", c, is_atx=False) for c in _split_to_budget(prose)]
    return out


def _render_page_image(pdf_path: Path, page_index0: int, dest: Path) -> bool:
    """Render one PDF page (image-only source) to a JPEG. Best-effort:
    returns False if the source/renderer is unavailable so the vision
    channel simply skips the page rather than failing ingest."""
    try:
        import pypdfium2 as pdfium
    except Exception:
        return False
    try:
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            if page_index0 >= len(doc):
                return False
            page = doc[page_index0]
            # Render scale is pixels-per-point. The assembled image-only
            # PDFs declare an oversized page box (one giant embedded
            # raster), so a fixed DPI would explode to tens of MP/page.
            # Cap the long side instead — pin-sharp enough for the VLM
            # reader / vision embedder, bounded cost regardless of box.
            w_pt, h_pt = page.get_size()
            scale = _PAGE_IMAGE_LONG_PX / max(w_pt, h_pt, 1.0)
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil().convert("RGB")
            dest.parent.mkdir(parents=True, exist_ok=True)
            pil.save(dest, "JPEG", quality=90)
            return True
        finally:
            doc.close()
    except Exception:
        return False


def build_pageassets_from_dbocr(
    doc_dir: Path,
    file_id: str,
    *,
    source_pdf: Optional[Path] = None,
    persist: bool = True,
) -> List[PageAsset]:
    """Build (and optionally persist + warm) PageAssets for one document
    from its Double-Bench ``ocr_text`` directory.

    ``doc_dir`` is ``.../ocr_text/<lang>/<id>``; ``source_pdf`` is the
    image-only assembled PDF used to render per-page rasters for the
    vision channel.
    """
    doc_dir = Path(doc_dir)
    text_items = _numbered_txt(doc_dir / "text")
    if not text_items:
        raise FileNotFoundError(f"no text/ pages under {doc_dir}")

    by_page_aux: Dict[int, List[Tuple[str, int, Path]]] = {}
    for label, sub in (("Table", "table_text"), ("Figure", "figure_text")):
        for pidx, item, path in _numbered_txt(doc_dir / sub):
            by_page_aux.setdefault(pidx, []).append((label, item, path))

    pages: List[PageAsset] = []
    for pidx, _item, tpath in text_items:
        page_number = pidx + 1            # repo contract is 1-based
        page_id = f"p_{page_number:04d}"

        blocks = _blocks_from_page_text(tpath.read_text(encoding="utf-8", errors="replace"))
        aux = sorted(by_page_aux.get(pidx, []), key=lambda t: (t[0], t[1]))
        has_table = any(lbl == "Table" for lbl, _, _ in aux)
        has_figure = any(lbl == "Figure" for lbl, _, _ in aux)
        for lbl, item, apath in aux:
            blocks += _aux_blocks(
                lbl, item, apath.read_text(encoding="utf-8", errors="replace")
            )

        layout_blocks: List[Dict[str, Any]] = []
        md_parts: List[str] = []
        for bid, b in enumerate(blocks):
            layout_blocks.append(
                {
                    "block_label": b.label,
                    "block_content": b.content,
                    "block_id": bid,
                    "block_order": bid,
                }
            )
            md_parts.append(b.content)
        text_markdown = "\n\n".join(md_parts).strip()

        image_blocks = [
            {
                "image_id": f"{file_id}/{page_id}/figure_{item}",
                "path": None,
                "type": "figure",
            }
            for lbl, item, _ in aux
            if lbl == "Figure"
        ]

        page_image_path: Optional[str] = None
        if source_pdf is not None:
            rel = f"pages/{page_id}.jpg"
            dest = paddle_ocr_root() / file_id / rel
            if dest.is_file() or _render_page_image(Path(source_pdf), pidx, dest):
                page_image_path = rel

        signals = PageModeSignals(
            has_table=has_table,
            has_chart=False,
            has_figure=has_figure,
            low_ocr_confidence=False,
            scanned_page=False,
            dense_layout=len(layout_blocks) >= 12,
        )
        pages.append(
            PageAsset(
                page_id=page_id,
                file_id=file_id,
                page_number=page_number,
                text_markdown=text_markdown,
                page_image_path=page_image_path,
                table_blocks=[],
                image_blocks=image_blocks,
                layout_blocks=layout_blocks,
                page_mode=classify_page_mode(signals),
                quality_flags={
                    "has_table": has_table,
                    "has_chart": False,
                    "has_figure": has_figure,
                    "low_ocr_confidence": False,
                    "scanned_page": False,
                    "dense_layout": len(layout_blocks) >= 12,
                },
            )
        )

    if persist:
        persist_and_warm_page_assets(file_id, pages)
    return pages
