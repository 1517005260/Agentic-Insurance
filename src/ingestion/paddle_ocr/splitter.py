"""Slice a PDF into ≤N-page in-memory batches.

The layout-parsing API caps each request at 50 pages. The splitter operates
purely in memory: pypdf reads the source, then writes one byte blob per batch
to feed the client.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator, List, Union

from pypdf import PdfReader, PdfWriter

from config.settings import PADDLE_OCR_MAX_PAGES_PER_BATCH


@dataclass
class PdfBatch:
    """A single batch slice of a source PDF.

    page_start / page_end are 1-based and inclusive in the source PDF's
    coordinate space; page_count is derived.
    """

    index: int
    page_start: int
    page_end: int
    pdf_bytes: bytes

    @property
    def page_count(self) -> int:
        return self.page_end - self.page_start + 1


class PdfBatchSplitter:
    """Slice a PDF into ≤``max_pages_per_batch`` batches."""

    def __init__(self, max_pages_per_batch: int = PADDLE_OCR_MAX_PAGES_PER_BATCH):
        if max_pages_per_batch < 1:
            raise ValueError("max_pages_per_batch must be >= 1")
        self.max_pages_per_batch = max_pages_per_batch

    def page_count(self, pdf: Union[str, Path, bytes]) -> int:
        return len(self._open(pdf).pages)

    def split(self, pdf: Union[str, Path, bytes]) -> List[PdfBatch]:
        return list(self.iter_batches(pdf))

    def iter_batches(self, pdf: Union[str, Path, bytes]) -> Iterator[PdfBatch]:
        reader = self._open(pdf)
        total = len(reader.pages)
        if total == 0:
            return

        for batch_idx, batch_start in enumerate(range(0, total, self.max_pages_per_batch)):
            batch_end = min(batch_start + self.max_pages_per_batch, total)

            writer = PdfWriter()
            for page_idx in range(batch_start, batch_end):
                writer.add_page(reader.pages[page_idx])

            buf = BytesIO()
            writer.write(buf)
            yield PdfBatch(
                index=batch_idx,
                page_start=batch_start + 1,
                page_end=batch_end,
                pdf_bytes=buf.getvalue(),
            )

    @staticmethod
    def _open(pdf: Union[str, Path, bytes]) -> PdfReader:
        if isinstance(pdf, (bytes, bytearray)):
            return PdfReader(BytesIO(pdf))
        return PdfReader(str(pdf))
