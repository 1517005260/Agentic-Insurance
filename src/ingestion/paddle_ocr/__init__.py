"""PDF / image → page Markdown + images via the PaddleOCR layout-parsing jobs API."""

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
