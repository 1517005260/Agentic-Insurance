"""PDF / image → page Markdown + images via the PP-StructureV3 layout-parsing API."""

from ingestion.paddle_ocr.client import PaddleOCRClient, PaddleOCRResponse
from ingestion.paddle_ocr.parser import ParseResult, PdfParser
from ingestion.paddle_ocr.splitter import PdfBatch, PdfBatchSplitter

__all__ = [
    "PaddleOCRClient",
    "PaddleOCRResponse",
    "PdfBatch",
    "PdfBatchSplitter",
    "PdfParser",
    "ParseResult",
]
