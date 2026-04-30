"""Orchestrator: split → submit → persist → concatenate.

For each source file the parser produces:

    STORAGE_PATH/paddle_ocr/<file_id>/
        raw/
            batch_000/
                doc_0.md, doc_1.md, ...
                response.json
                <batch-local image paths from the API>
            batch_001/...
        combined.md     # all batches concatenated, image refs prefixed by batch dir
        meta.json       # file_id, source_path, total_pages, batches[]

`combined.md` is delimited by `PADDLE_OCR_BATCH_SEPARATOR` so a downstream
page-asset builder can recover batch boundaries.
"""
import hashlib
import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from config.settings import (
    PADDLE_OCR_BATCH_SEPARATOR,
    PADDLE_OCR_FILE_TYPE_IMAGE,
    PADDLE_OCR_FILE_TYPE_PDF,
    PADDLE_OCR_MAX_PAGES_PER_BATCH,
    paddle_ocr_root,
)

from ingestion.paddle_ocr.client import PaddleOCRClient, PaddleOCRResponse
from ingestion.paddle_ocr.splitter import PdfBatch, PdfBatchSplitter

logger = logging.getLogger(__name__)


@dataclass
class BatchOutput:
    """Persisted record of one batch's PaddleOCR run."""

    batch_index: int
    page_start: int
    page_end: int
    page_count: int
    batch_dir: str
    markdown_files: List[str] = field(default_factory=list)
    image_files: List[str] = field(default_factory=list)
    num_layout_results: int = 0


@dataclass
class ParseResult:
    """Final result of :meth:`PdfParser.parse`."""

    file_id: str
    source_path: Optional[str]
    file_type: int
    total_pages: int
    output_dir: str
    combined_markdown_path: str
    meta_path: str
    batches: List[BatchOutput] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_id": self.file_id,
            "source_path": self.source_path,
            "file_type": self.file_type,
            "total_pages": self.total_pages,
            "output_dir": self.output_dir,
            "combined_markdown_path": self.combined_markdown_path,
            "meta_path": self.meta_path,
            "batches": [asdict(b) for b in self.batches],
        }


class PdfParser:
    """Run PaddleOCR end-to-end for a single source file."""

    def __init__(
        self,
        client: Optional[PaddleOCRClient] = None,
        splitter: Optional[PdfBatchSplitter] = None,
        storage_root: Optional[Union[str, Path]] = None,
        max_pages_per_batch: int = PADDLE_OCR_MAX_PAGES_PER_BATCH,
    ):
        self.client = client or PaddleOCRClient()
        self.splitter = splitter or PdfBatchSplitter(max_pages_per_batch=max_pages_per_batch)
        self.storage_root = Path(storage_root) if storage_root is not None else paddle_ocr_root()

    # --------------------------------------------------------------- public

    def parse(
        self,
        source: Union[str, Path],
        file_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> ParseResult:
        """Parse a local PDF or image file.

        `file_id` defaults to ``<stem>_<sha256[:16]>`` so re-parsing the same
        bytes reuses the same output directory; pass ``overwrite=True`` to
        wipe an existing directory.
        """
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        file_bytes = source_path.read_bytes()
        if file_id is None:
            file_id = self._derive_file_id(source_path, file_bytes)

        is_pdf = source_path.suffix.lower() == ".pdf"
        file_type = PADDLE_OCR_FILE_TYPE_PDF if is_pdf else PADDLE_OCR_FILE_TYPE_IMAGE

        output_dir = self.storage_root / file_id
        if output_dir.exists():
            if overwrite:
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    f"Pass overwrite=True to replace it."
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_root = output_dir / "raw"
        raw_root.mkdir(parents=True, exist_ok=True)

        if is_pdf:
            batches = self.splitter.split(file_bytes)
            total_pages = sum(b.page_count for b in batches)
        else:
            # Single-image submission — one synthetic 1-page batch.
            batches = [PdfBatch(index=0, page_start=1, page_end=1, pdf_bytes=file_bytes)]
            total_pages = 1

        logger.info(
            "PdfParser.parse: file_id=%s total_pages=%d batches=%d output_dir=%s",
            file_id,
            total_pages,
            len(batches),
            output_dir,
        )

        batch_records: List[BatchOutput] = []
        for batch in batches:
            batch_dir = raw_root / f"batch_{batch.index:03d}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            if is_pdf:
                resp = self.client.parse_pdf_bytes(batch.pdf_bytes, batch_dir)
            else:
                resp = self.client.parse_image_bytes(batch.pdf_bytes, batch_dir)

            batch_records.append(self._record_batch(batch, batch_dir, resp))

        combined_md = self._concatenate_markdown(batch_records, raw_root)
        combined_path = output_dir / "combined.md"
        combined_path.write_text(combined_md, encoding="utf-8")

        meta = {
            "file_id": file_id,
            "source_path": str(source_path),
            "file_type": file_type,
            "total_pages": total_pages,
            "max_pages_per_batch": self.splitter.max_pages_per_batch,
            "batches": [asdict(b) for b in batch_records],
        }
        meta_path = output_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return ParseResult(
            file_id=file_id,
            source_path=str(source_path),
            file_type=file_type,
            total_pages=total_pages,
            output_dir=str(output_dir),
            combined_markdown_path=str(combined_path),
            meta_path=str(meta_path),
            batches=batch_records,
        )

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _derive_file_id(source_path: Path, file_bytes: bytes) -> str:
        digest = hashlib.sha256(file_bytes).hexdigest()[:16]
        stem = source_path.stem.replace(" ", "_")
        return f"{stem}_{digest}"

    @staticmethod
    def _record_batch(
        batch: PdfBatch, batch_dir: Path, resp: PaddleOCRResponse
    ) -> BatchOutput:
        return BatchOutput(
            batch_index=batch.index,
            page_start=batch.page_start,
            page_end=batch.page_end,
            page_count=batch.page_count,
            batch_dir=str(batch_dir),
            markdown_files=[str(p) for p in resp.markdown_paths],
            image_files=[str(p) for p in resp.image_paths],
            num_layout_results=len(resp.markdown_texts),
        )

    @staticmethod
    def _concatenate_markdown(batches: List[BatchOutput], raw_root: Path) -> str:
        """Stitch every doc_<i>.md from every batch into a single Markdown blob.

        Image refs in the API output are batch-local; rewrite each one to be
        relative to the file's output dir so the combined Markdown renders
        correctly when served from `<file_id>/`.
        """
        parts: List[str] = []
        for batch in batches:
            batch_dir = Path(batch.batch_dir)
            rel_prefix = batch_dir.relative_to(raw_root.parent).as_posix()
            for md_file in batch.markdown_files:
                text = Path(md_file).read_text(encoding="utf-8")
                parts.append(_rewrite_image_paths(text, rel_prefix))
        return PADDLE_OCR_BATCH_SEPARATOR.join(parts)


_IMG_LINK_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _rewrite_image_paths(markdown: str, rel_prefix: str) -> str:
    """Prefix relative image paths with `rel_prefix/`. Absolute URLs are kept."""

    def _replace(match: re.Match[str]) -> str:
        full, alt, target = match.group(0), match.group(1), match.group(2)
        target_stripped = target.strip()
        if (
            target_stripped.startswith(("http://", "https://", "data:", "/"))
        ):
            return full
        return f"![{alt}]({rel_prefix}/{target_stripped})"

    return _IMG_LINK_RE.sub(_replace, markdown)
