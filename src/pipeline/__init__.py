"""End-to-end ingestion → index pipeline.

Single entry point :func:`parse_and_index` that runs the four-stage flow:

    PDF
      → PaddleOCR parse (split + submit + concat)
      → ingestion (page assets + page_mode tagging)
      → four index builders, run concurrently
"""

from pipeline.parse_and_index import (
    PipelineResult,
    parse_and_index,
    parse_and_index_many,
)

__all__ = ["PipelineResult", "parse_and_index", "parse_and_index_many"]
