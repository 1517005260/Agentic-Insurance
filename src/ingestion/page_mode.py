"""Classify a parsed page as ``text`` or ``text_with_img``.

The decision drives downstream behavior:

* ``text``           — page is read with Markdown only.
* ``text_with_img``  — page is also routed to the VLM channel because
                       layout/structure carries information not in the OCR
                       Markdown alone (tables, charts, figures, scanned
                       regions, dense layouts, low OCR confidence).
"""

from dataclasses import dataclass


@dataclass
class PageModeSignals:
    has_table: bool = False
    has_chart: bool = False
    has_figure: bool = False
    low_ocr_confidence: bool = False
    scanned_page: bool = False
    dense_layout: bool = False
    table_structure_uncertain: bool = False

    def any_visual_signal(self) -> bool:
        return any(
            (
                self.has_table,
                self.has_chart,
                self.has_figure,
                self.low_ocr_confidence,
                self.scanned_page,
                self.dense_layout,
                self.table_structure_uncertain,
            )
        )


def classify_page_mode(signals: PageModeSignals) -> str:
    """Return ``"text_with_img"`` if any visual signal fires, else ``"text"``."""
    return "text_with_img" if signals.any_visual_signal() else "text"
