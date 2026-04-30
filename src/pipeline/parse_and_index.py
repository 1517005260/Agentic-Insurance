"""Parse PDFs, build all four indexes per file.

Two entry points:

* :func:`parse_and_index`         — single file, end-to-end.
* :func:`parse_and_index_many`    — many files, pipelined.

Concurrency model (parse_and_index_many):

* **Parsers run concurrent**: paddle OCR is a remote service and each file
  writes to a per-file directory, so N files parse in parallel with no
  shared state.
* **Ingest is serial across files**: the four global faiss stores + the
  shared LinearRAG graphml are mutated by every ingest, so two files
  ingesting at the same time would corrupt each other. We pipeline instead:
  as soon as a parse completes, its ingest starts on the main thread while
  the other parses keep running in the pool.
* **Within one file, the four builders run concurrent** (the original
  parse_and_index() shape) — they touch different stores.
"""
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from ingestion import build_page_assets
from ingestion.index import (
    BM25IndexBuilder,
    GraphIndexBuilder,
    IndexBuilder,
    IndexBuildResult,
    TextDenseIndexBuilder,
    VisionDenseIndexBuilder,
)
from ingestion.paddle_ocr import ParseResult, PdfParser
from storage.page_store import PageAsset

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Outcome of one file through parse + ingest.

    ``ok=False`` and ``error`` carry a per-source failure (parse exploded,
    overwrite=False raised, etc.) without losing the source's place in the
    returned list — callers always see one PipelineResult per input source.
    """
    parse: Optional[ParseResult] = None
    pages: List[PageAsset] = field(default_factory=list)
    indexes: List[IndexBuildResult] = field(default_factory=list)
    total_seconds: float = 0.0
    source: Optional[str] = None
    ok: bool = True
    error: Optional[str] = None


def _default_builders() -> List[IndexBuilder]:
    """Fresh builder instances per call so each holds its own faiss handle."""
    return [
        TextDenseIndexBuilder(),
        VisionDenseIndexBuilder(),
        BM25IndexBuilder(),
        GraphIndexBuilder(),
    ]


def _ingest_one(
    parse: ParseResult,
    builders: Sequence[IndexBuilder],
) -> tuple[List[PageAsset], List[IndexBuildResult]]:
    """Page-asset build + 4-way concurrent index build for a single parsed file."""
    pages = build_page_assets(parse, persist=True)
    logger.info("page assets built: %d pages (file_id=%s)", len(pages), parse.file_id)

    results: List[IndexBuildResult] = []
    with ThreadPoolExecutor(max_workers=len(builders)) as pool:
        future_to_name = {
            pool.submit(b.build, parse.file_id, pages): b.name for b in builders
        }
        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            try:
                res = fut.result()
                logger.info(
                    "index %s done: items=%d skipped=%s (file_id=%s)",
                    name,
                    res.item_count,
                    res.skipped_reason,
                    parse.file_id,
                )
                results.append(res)
            except Exception:
                logger.exception(
                    "index %s FAILED (file_id=%s)", name, parse.file_id
                )
                results.append(
                    IndexBuildResult(
                        index_name=name,
                        file_id=parse.file_id,
                        output_dir="",
                        skipped_reason="build raised",
                    )
                )
    return pages, results


def parse_and_index(
    source: Union[str, Path],
    *,
    file_id: Optional[str] = None,
    overwrite: bool = False,
    parser: Optional[PdfParser] = None,
    builders: Optional[Sequence[IndexBuilder]] = None,
) -> PipelineResult:
    """Single-file pipeline. Backwards-compatible shape."""
    t0 = time.perf_counter()
    src_str = str(source)

    parser = parser or PdfParser()
    parse = parser.parse(source, file_id=file_id, overwrite=overwrite)
    logger.info(
        "parse done: file_id=%s pages=%d batches=%d",
        parse.file_id,
        parse.total_pages,
        len(parse.batches),
    )

    builder_list = list(builders) if builders is not None else _default_builders()
    pages, results = _ingest_one(parse, builder_list)

    return PipelineResult(
        parse=parse,
        pages=pages,
        indexes=results,
        total_seconds=time.perf_counter() - t0,
        source=src_str,
        ok=True,
        error=None,
    )


def parse_and_index_many(
    sources: Iterable[Union[str, Path]],
    *,
    overwrite: bool = False,
    parser: Optional[PdfParser] = None,
    builders: Optional[Sequence[IndexBuilder]] = None,
    parse_workers: int = 4,
) -> List[PipelineResult]:
    """Pipelined many-file pipeline.

    Parses run concurrently in a thread pool of size ``parse_workers``;
    ingest runs serially on the main thread, taking each parsed file as
    soon as it is ready (so ingest of file N runs while parses of files
    N+1, N+2 ... are still in flight).
    """
    sources = [Path(s) for s in sources]
    if not sources:
        return []

    t0 = time.perf_counter()
    parser = parser or PdfParser()
    builder_list = list(builders) if builders is not None else _default_builders()

    results: dict[Path, PipelineResult] = {}
    with ThreadPoolExecutor(max_workers=parse_workers) as parse_pool:
        future_to_src: dict[Future, Path] = {
            parse_pool.submit(parser.parse, src, None, overwrite): src
            for src in sources
        }
        for fut in as_completed(future_to_src):
            src = future_to_src[fut]
            file_t0 = time.perf_counter()
            try:
                parse = fut.result()
            except Exception as exc:
                logger.exception("parse FAILED for %s", src)
                results[src] = PipelineResult(
                    source=str(src),
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    total_seconds=time.perf_counter() - file_t0,
                )
                continue
            logger.info(
                "parse done: file_id=%s pages=%d batches=%d (%s)",
                parse.file_id,
                parse.total_pages,
                len(parse.batches),
                src.name,
            )
            try:
                pages, idx_results = _ingest_one(parse, builder_list)
            except Exception as exc:
                logger.exception("ingest FAILED for %s", src)
                results[src] = PipelineResult(
                    parse=parse,
                    source=str(src),
                    ok=False,
                    error=f"ingest: {type(exc).__name__}: {exc}",
                    total_seconds=time.perf_counter() - file_t0,
                )
                continue
            results[src] = PipelineResult(
                parse=parse,
                pages=pages,
                indexes=idx_results,
                total_seconds=time.perf_counter() - file_t0,
                source=str(src),
                ok=True,
                error=None,
            )

    n_ok = sum(1 for r in results.values() if r.ok)
    logger.info(
        "parse_and_index_many done: %d/%d files OK in %.1fs",
        n_ok,
        len(sources),
        time.perf_counter() - t0,
    )
    # Preserve input order; every source has a slot (incl. failures).
    return [results[s] for s in sources if s in results]
