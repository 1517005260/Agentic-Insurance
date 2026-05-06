"""End-to-end data preparation: PaddleOCR parse, page assets, retrieval indexes.

Top-level helpers in this module bridge raw PaddleOCR outputs to the
canonical PageAsset list consumed by every index builder.
"""

from ingestion.page_assets import PageAssetBuilder, build_page_assets
from ingestion.page_mode import PageModeSignals, classify_page_mode

__all__ = [
    "PageAssetBuilder",
    "build_page_assets",
    "PageModeSignals",
    "classify_page_mode",
]
