"""PDF page rendering for /files/{id}/preview.

Renders the original PDF page rather than reusing PaddleOCR's
``layout_det_res_0.jpg`` (the layout-detector's visualization with
colored bounding boxes), which looks like a CV debug screenshot rather
than a page thumbnail.

This module renders the original PDF page via ``pypdfium2`` (the
PDFium wrapper Chromium uses) and caches the result under
``<STORAGE_PATH>/preview/<file_id>/p_NNNN.jpg``. Subsequent requests
hit the cache. Re-ingest of a file invalidates the cache because
``purge_file_artifacts`` removes the ``preview/<file_id>`` directory
(see ``ingestion/index/__init__.py``).

The renderer is sync and CPU-bound (PDFium decode + libjpeg encode);
callers should run it via ``run_in_executor`` to avoid stalling the
event loop.
"""
import logging
import tempfile
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium

from config.settings import preview_page_path


logger = logging.getLogger(__name__)


# Default rasterization scale. PDFium uses 72 dpi as 1.0; 1.5 lands
# around the 100-dpi range — sharp enough for a 240px-wide thumbnail
# and the full-file scrolling preview, while keeping JPG output under
# ~150 KB / page on typical insurance PDFs.
_DEFAULT_SCALE: float = 1.5
_JPEG_QUALITY: int = 82


def render_pdf_page_to_cache(
    pdf_path: Path,
    *,
    file_id: str,
    page_number: int,
    scale: float = _DEFAULT_SCALE,
    overwrite: bool = False,
) -> Optional[Path]:
    """Render ``page_number`` (1-based) of ``pdf_path`` to the preview cache.

    Returns the cache path on success, or ``None`` when the source PDF
    cannot be opened / decoded / page index is out of range. Existing
    cache files are returned unchanged unless ``overwrite=True``.

    Idempotent and safe to call from multiple ingest threads — the
    cache write is atomic (write to ``.part`` then ``.replace``) so a
    half-written file never reaches a concurrent reader.
    """
    if page_number < 1:
        return None
    target = preview_page_path(file_id, page_number)
    if target.is_file() and not overwrite:
        return target

    try:
        doc = pdfium.PdfDocument(str(pdf_path))
    except Exception:
        logger.warning("preview render: failed to open PDF: %s", pdf_path, exc_info=True)
        return None

    try:
        n_pages = len(doc)
        if page_number > n_pages:
            return None
        try:
            page = doc[page_number - 1]
        except Exception:
            logger.warning(
                "preview render: page %d out of range (n=%d) for %s",
                page_number, n_pages, pdf_path,
            )
            return None
        try:
            pil_image = page.render(scale=scale).to_pil()
        finally:
            page.close()
    finally:
        # Always close to release the underlying file handle —
        # pypdfium2 keeps a mmap-style ref that blocks further writes
        # to the same path on Windows.
        doc.close()

    target.parent.mkdir(parents=True, exist_ok=True)
    # ``mkstemp`` gives a unique ``.part`` filename per render so two
    # concurrent requests for the same page can't clobber each other's
    # tmp file (the older shared-name approach exposed a half-written
    # JPEG when the second renamer beat the first writer to disk). The
    # final ``replace`` is atomic on Linux/macOS so readers always see
    # either the previous good cache or the new one.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".part", dir=str(target.parent)
    )
    tmp = Path(tmp_name)
    try:
        # ``optimize`` shaves another ~10 % off the JPG; cheap at
        # render time, hot-path on the read side. We close the fd
        # explicitly inside the with-block so PIL doesn't leak it.
        try:
            with open(fd, "wb") as fh:
                pil_image.save(
                    fh,
                    format="JPEG",
                    quality=_JPEG_QUALITY,
                    optimize=True,
                    progressive=False,
                )
        finally:
            # In case PIL didn't actually consume fh (e.g. raised early)
            # ensure the descriptor is closed; ``with`` above already
            # handled the happy path.
            pass
        tmp.replace(target)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        logger.warning("preview render: failed to write cache for %s", target, exc_info=True)
        return None
    return target


def page_count(pdf_path: Path) -> int:
    """Return the document's page count, or 0 on failure.

    Used by the full-file preview to know how many ``<Page>`` slots
    to lazy-render. Cheap (PDFium parses the xref but not the page
    content) so we don't bother caching.
    """
    try:
        doc = pdfium.PdfDocument(str(pdf_path))
    except Exception:
        return 0
    try:
        return len(doc)
    finally:
        doc.close()


__all__ = ["render_pdf_page_to_cache", "page_count"]
