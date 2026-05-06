"""Build PageAsset records from a persisted PaddleOCR parse result.

Walks ``STORAGE_PATH/paddle_ocr/<file_id>/raw/batch_NNN/response.json`` and
materializes one PageAsset per ``layoutParsingResults`` entry. Each entry is
treated as one logical "page" — the natural unit returned by the
PP-StructureV3 layout-parsing API.

Page numbering: within a batch, the i-th entry maps to source PDF page
``batch.page_start + i`` (1-based). When the API returns a single
doc-spanning entry (e.g. for a non-PDF image) the entry inherits
``batch.page_start``.

Page mode tagging: layout block types and quality flags drive the
text / text_with_img split via :func:`page_mode.classify_page_mode`.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import page_assets_path
from ingestion.page_mode import PageModeSignals, classify_page_mode
from storage.page_store import PageAsset


# Layout block "type" tokens we treat as visual signals. PaddleOCR
# PP-StructureV3 uses ``block_label`` values like:
#   doc_title / paragraph_title / figure_title / vision_footnote
#   header / footer / header_image
#   text / number / table / image / chart / figure / formula
_TABLE_BLOCK_TYPES = {"table"}
_FIGURE_BLOCK_TYPES = {"figure", "image", "header_image", "figure_title"}
_CHART_BLOCK_TYPES = {"chart"}

# Output-image keys typically used by the API for the rendered/preprocessed
# page image. First match wins.
_PAGE_IMAGE_KEY_PREFIXES = ("preprocessed_img", "doc_preprocessor_res", "layout_det_res")


@dataclass
class PageAssetBuilder:
    """Build PageAsset objects from a paddle parse output directory."""

    output_dir: Path  # STORAGE_PATH/paddle_ocr/<file_id>/
    file_id: str

    @classmethod
    def from_parse_result(cls, parse_result) -> "PageAssetBuilder":
        return cls(output_dir=Path(parse_result.output_dir), file_id=parse_result.file_id)

    def build(self) -> List[PageAsset]:
        meta = self._load_meta()
        pages: List[PageAsset] = []
        for batch in meta["batches"]:
            pages.extend(self._build_batch(batch))
        return pages

    # ------------------------------------------------------------------ batch

    def _build_batch(self, batch: Dict[str, Any]) -> List[PageAsset]:
        batch_dir = Path(batch["batch_dir"])
        response_path = batch_dir / "response.json"
        if not response_path.is_file():
            return []
        layout_results = json.loads(response_path.read_text(encoding="utf-8")).get(
            "layoutParsingResults", []
        ) or []

        rel_prefix = batch_dir.relative_to(self.output_dir).as_posix()
        page_start: int = batch["page_start"]

        pages: List[PageAsset] = []
        for i, res in enumerate(layout_results):
            page_number = page_start + i
            page_id = self._page_id(page_number)
            text = (res.get("markdown") or {}).get("text", "") or ""
            tables = self._extract_tables(res, rel_prefix)
            images = self._extract_images(res, rel_prefix)
            page_image = self._locate_page_image(res, batch_dir, i, rel_prefix)
            layout_blocks = self._extract_layout_blocks(res)
            signals = self._signals(res, tables, images, layout_blocks)
            mode = classify_page_mode(signals)
            quality_flags = {
                "has_table": signals.has_table,
                "has_chart": signals.has_chart,
                "has_figure": signals.has_figure,
                "low_ocr_confidence": signals.low_ocr_confidence,
                "scanned_page": signals.scanned_page,
                "dense_layout": signals.dense_layout,
            }

            pages.append(
                PageAsset(
                    page_id=page_id,
                    file_id=self.file_id,
                    page_number=page_number,
                    text_markdown=self._rewrite_image_paths(text, rel_prefix),
                    page_image_path=page_image,
                    table_blocks=tables,
                    image_blocks=images,
                    layout_blocks=layout_blocks,
                    page_mode=mode,
                    quality_flags=quality_flags,
                )
            )
        return pages

    # ---------------------------------------------------------- page-level helpers

    @staticmethod
    def _page_id(page_number: int) -> str:
        return f"p_{page_number:04d}"

    @staticmethod
    def _extract_layout_blocks(res: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Pull layout blocks out of the layout-parsing response.

        PaddleOCR PP-StructureV3 returns blocks under
        ``prunedResult.parsing_res_list`` with keys ``block_label``,
        ``block_content``, ``block_bbox``, ``block_id``, ``block_order``,
        ``group_id``. A handful of top-level keys are probed as
        fallbacks for alternate deployment shapes.
        """
        pruned = res.get("prunedResult") or {}
        blocks = pruned.get("parsing_res_list")
        if blocks:
            return blocks
        return (
            res.get("layoutBlocks")
            or res.get("layout_det_res")
            or res.get("layoutDetectionResults")
            or []
        )

    @staticmethod
    def _block_label(block: Dict[str, Any]) -> str:
        """Layout-block label, normalized to lower-case."""
        return (
            block.get("block_label")
            or block.get("block_type")
            or block.get("label")
            or ""
        ).lower()

    @staticmethod
    def _signals(
        res: Dict[str, Any],
        tables: List[Dict[str, Any]],
        images: List[Dict[str, Any]],
        layout_blocks: List[Dict[str, Any]],
    ) -> PageModeSignals:
        has_table = bool(tables) or any(
            PageAssetBuilder._block_label(b) in _TABLE_BLOCK_TYPES for b in layout_blocks
        )
        has_figure = any(
            PageAssetBuilder._block_label(b) in _FIGURE_BLOCK_TYPES for b in layout_blocks
        ) or any((img.get("type") or "").lower() == "figure" for img in images)
        has_chart = any(
            PageAssetBuilder._block_label(b) in _CHART_BLOCK_TYPES for b in layout_blocks
        ) or any((img.get("type") or "").lower() == "chart" for img in images)

        # OCR confidence + scanned flags — present on some pipeline variants.
        low_ocr = bool(res.get("low_ocr_confidence", False))
        scanned = bool(res.get("scanned_page", False))
        # Dense layout heuristic: if layout has >=12 blocks the page is
        # information-dense regardless of explicit flags.
        dense = bool(res.get("dense_layout", False)) or (len(layout_blocks) >= 12)

        return PageModeSignals(
            has_table=has_table,
            has_chart=has_chart,
            has_figure=has_figure,
            low_ocr_confidence=low_ocr,
            scanned_page=scanned,
            dense_layout=dense,
        )

    @staticmethod
    def _extract_tables(res: Dict[str, Any], rel_prefix: str) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = []
        for i, t in enumerate(res.get("tables") or res.get("table_results") or []):
            tables.append(
                {
                    "table_id": f"{rel_prefix}/table_{i}",
                    "html": t.get("html") or t.get("table_html"),
                    "markdown": t.get("markdown") or t.get("table_markdown"),
                }
            )
        return tables

    @staticmethod
    def _extract_images(res: Dict[str, Any], rel_prefix: str) -> List[Dict[str, Any]]:
        images: List[Dict[str, Any]] = []
        markdown_images = (res.get("markdown") or {}).get("images") or {}
        for path in markdown_images.keys():
            images.append(
                {
                    "image_id": f"{rel_prefix}/{path}",
                    "path": f"{rel_prefix}/{path}",
                    "type": "figure",
                }
            )
        return images

    @staticmethod
    def _locate_page_image(
        res: Dict[str, Any],
        batch_dir: Path,
        i: int,
        rel_prefix: str,
    ) -> Optional[str]:
        """Return the on-disk path to the rendered page image, if present."""
        output_images_dir = batch_dir / "output_images"
        if not output_images_dir.is_dir():
            return None
        # The client saves output_images as ``<name>_<i>.jpg``; pick the
        # first prefix-match for this entry index.
        for prefix in _PAGE_IMAGE_KEY_PREFIXES:
            candidate = output_images_dir / f"{prefix}_{i}.jpg"
            if candidate.is_file():
                return f"{rel_prefix}/output_images/{candidate.name}"
        # Fallback: any JPG matching this index.
        for candidate in sorted(output_images_dir.glob(f"*_{i}.jpg")):
            return f"{rel_prefix}/output_images/{candidate.name}"
        return None

    @staticmethod
    def _rewrite_image_paths(markdown: str, rel_prefix: str) -> str:
        """Same rewrite the parser does on combined.md, applied per-page."""
        import re

        def _replace(match: "re.Match[str]") -> str:
            full, alt, target = match.group(0), match.group(1), match.group(2)
            t = target.strip()
            if t.startswith(("http://", "https://", "data:", "/")):
                return full
            return f"![{alt}]({rel_prefix}/{t})"

        return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _replace, markdown)

    # ------------------------------------------------------------------ meta

    def _load_meta(self) -> Dict[str, Any]:
        return json.loads((self.output_dir / "meta.json").read_text(encoding="utf-8"))


def build_page_assets(parse_result, persist: bool = True) -> List[PageAsset]:
    """Build PageAssets from a ParseResult and (optionally) persist them.

    Persistence performs three steps in order:

    1. Write ``STORAGE_PATH/page_assets/<file_id>.json`` (the canonical
       PageStore source).
    2. Eagerly warm the section :class:`InventoryStore` for the new
       file (heading-derived).
    3. Eagerly warm the sibling stores —
       :class:`storage.PassageStore` and :class:`storage.TableRowStore`
       — so the proof gate's read path is cache-only at agent-loop
       time.

    Steps 2-3 are deliberately part of ingest, not lazy-on-read,
    because passage/table_row stores are read often during a single
    agent run and the build cost belongs at index time.
    """
    builder = PageAssetBuilder.from_parse_result(parse_result)
    pages = builder.build()
    if persist:
        out = page_assets_path(parse_result.file_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([_asset_to_dict(p) for p in pages], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Warm the inventory + sibling stores. Construct fresh PageStore
        # / InventoryStore against the directory we just wrote to so we
        # don't depend on a global already-loaded instance.
        from config.settings import page_assets_root
        from storage import (
            InventoryStore,
            PageStore,
            build_inventory_atoms_for_file,
        )
        page_store = PageStore(page_assets_root())
        inventory = InventoryStore(page_store=page_store)
        inventory.warm_up()
        build_inventory_atoms_for_file(
            parse_result.file_id, page_store, inventory,
        )
    return pages


def _asset_to_dict(p: PageAsset) -> Dict[str, Any]:
    return {
        "page_id": p.page_id,
        "file_id": p.file_id,
        "page_number": p.page_number,
        "text_markdown": p.text_markdown,
        "page_image_path": p.page_image_path,
        "table_blocks": p.table_blocks,
        "image_blocks": p.image_blocks,
        "layout_blocks": p.layout_blocks,
        "page_mode": p.page_mode,
        "quality_flags": p.quality_flags,
    }
