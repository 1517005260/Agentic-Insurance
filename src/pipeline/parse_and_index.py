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
* **Within one file, the four builders run concurrent by default**, but
  this can blow ~3 GB of resident memory (mostly the LinearRAG graph
  builder loading en + zh transformer NER plus its embedding stores).
  Pass ``parallel_builders=False`` to fold the four builders into a
  serial loop with ``gc.collect`` between stages — required on 8 GB
  WSL2 / small VMs to avoid an OOM kill mid-ingest.
"""
import gc
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Union

from ingestion import build_page_assets
from ingestion.index.base import IndexBuilder, IndexBuildResult
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


def _default_builders(*, linear_config: Optional[Any] = None) -> List[IndexBuilder]:
    """Fresh builder instances per call so each holds its own faiss handle.

    Builder imports are local to this function — the heavy graph builder
    pulls in torch + transformers transitively, and the spawn-mode subprocess
    that runs only ``GraphIndexBuilder`` must not pay for unused builder
    modules at child boot. Localising here keeps the parent-side worker
    factory eager but the child-side worker entry strictly minimal.
    """
    from ingestion.index.bm25_tantivy import BM25IndexBuilder
    from ingestion.index.graph_linearrag import GraphIndexBuilder
    from ingestion.index.text_dense import TextDenseIndexBuilder
    from ingestion.index.vision_dense import VisionDenseIndexBuilder

    return [
        TextDenseIndexBuilder(),
        VisionDenseIndexBuilder(),
        BM25IndexBuilder(),
        GraphIndexBuilder(linear_config=linear_config),
    ]


# ``on_event`` callback signature: ``(event_name, data)``. The pipeline
# fires ``stage`` events with ``{"stage": <name>, "phase": "start|done",
# "elapsed_ms"?, "items"?, "skipped_reason"?, "error"?}`` so the API
# layer can re-encode them as SSE frames (api/runners/ingestion_runner).
# Algorithm-layer callsites pass ``None`` and the pipeline runs silently.
OnEvent = Callable[[str, Mapping[str, Any]], None]


def _safe_emit(on_event: Optional[OnEvent], event: str, data: Mapping[str, Any]) -> None:
    """Fire ``on_event`` swallowing any exception.

    Same pattern BaseAgent / ProofAgent use — the algorithm pipeline
    must never bubble a callback failure up the build chain. The
    callback is typically ``EventBus.push`` which uses
    ``call_soon_threadsafe`` to enqueue, so the worker thread fires
    cheaply even when the consumer is gone.
    """
    if on_event is None:
        return
    try:
        on_event(event, data)
    except Exception:
        logger.debug("on_event(%s) swallowed", event, exc_info=True)


def _ingest_one(
    parse: ParseResult,
    builders: Sequence[IndexBuilder],
    *,
    parallel_builders: bool = True,
    on_event: Optional[OnEvent] = None,
) -> tuple[List[PageAsset], List[IndexBuildResult]]:
    """Page-asset build + index build for a single parsed file.

    ``parallel_builders=True`` (default) fans the four builders out onto
    a thread pool — fastest path on a beefy host, but graph + dense +
    visual concurrently can resident ~3 GB and OOMs on 8 GB WSL2.

    ``parallel_builders=False`` runs the four builders serially with a
    ``gc.collect`` between stages so each builder's working set is
    released before the next one starts. Roughly 2× wall time but the
    peak RSS stays bounded by the largest single builder (LinearRAG
    graph). Use this on memory-constrained hosts and from web-app
    background ingest tasks.

    ``on_event`` (when set) receives one ``stage`` event per phase
    boundary across 5 stages: ``page_assets`` then the four
    builders by their canonical ``name`` attribute. Order is
    deterministic in serial mode (input order) and undefined in
    parallel mode (as_completed order); the consumer must key by
    ``stage`` + ``phase``, not arrival index.
    """
    _safe_emit(on_event, "stage", {"stage": "page_assets", "phase": "start"})
    t = time.perf_counter()
    try:
        pages = build_page_assets(parse, persist=True)
    except Exception as exc:
        _safe_emit(
            on_event,
            "stage",
            {
                "stage": "page_assets",
                "phase": "done",
                "elapsed_ms": int((time.perf_counter() - t) * 1000),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    _safe_emit(
        on_event,
        "stage",
        {
            "stage": "page_assets",
            "phase": "done",
            "elapsed_ms": int((time.perf_counter() - t) * 1000),
            "items": len(pages),
        },
    )
    logger.info("page assets built: %d pages (file_id=%s)", len(pages), parse.file_id)

    if parallel_builders:
        return pages, _build_indexes_parallel(
            parse.file_id, builders, pages, on_event=on_event
        )
    return pages, _build_indexes_serial(
        parse.file_id, builders, pages, on_event=on_event,
    )


def _run_one_builder(
    file_id: str,
    builder: IndexBuilder,
    pages: List[PageAsset],
    on_event: Optional[OnEvent],
) -> IndexBuildResult:
    """Run one builder under the stage emitter contract.

    Wraps ``IndexBuilder.build`` with start/done event boundaries plus
    failure conversion to ``IndexBuildResult(failed=True)``. The
    ``done`` event always fires (success, skip, or failure) so the
    consumer sees a clean per-stage timeline.
    """
    _safe_emit(on_event, "stage", {"stage": builder.name, "phase": "start"})
    t = time.perf_counter()
    try:
        res = builder.build(file_id, pages)
        elapsed_ms = int((time.perf_counter() - t) * 1000)
        logger.info(
            "index %s done: items=%d skipped=%s (file_id=%s)",
            builder.name, res.item_count, res.skipped_reason, file_id,
        )
        _safe_emit(
            on_event,
            "stage",
            {
                "stage": builder.name,
                "phase": "done",
                "elapsed_ms": elapsed_ms,
                "items": res.item_count,
                "skipped_reason": res.skipped_reason,
            },
        )
        return res
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t) * 1000)
        logger.exception("index %s FAILED (file_id=%s)", builder.name, file_id)
        _safe_emit(
            on_event,
            "stage",
            {
                "stage": builder.name,
                "phase": "done",
                "elapsed_ms": elapsed_ms,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return IndexBuildResult(
            index_name=builder.name,
            file_id=file_id,
            output_dir="",
            skipped_reason=f"build raised: {type(exc).__name__}: {exc}",
            failed=True,
        )


def _build_indexes_parallel(
    file_id: str,
    builders: Sequence[IndexBuilder],
    pages: List[PageAsset],
    *,
    on_event: Optional[OnEvent] = None,
) -> List[IndexBuildResult]:
    results: List[IndexBuildResult] = []
    with ThreadPoolExecutor(max_workers=len(builders)) as pool:
        futures = [pool.submit(_run_one_builder, file_id, b, pages, on_event) for b in builders]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def _build_indexes_serial(
    file_id: str,
    builders: Sequence[IndexBuilder],
    pages: List[PageAsset],
    *,
    on_event: Optional[OnEvent] = None,
) -> List[IndexBuildResult]:
    """Run every builder in order, ``gc.collect`` between stages.

    Used by the web ingest path on memory-constrained hosts (8 GB
    WSL): each builder's working set (NER pipeline buffers, embedding
    client batches, igraph snapshots) gets a chance to release before
    the next stage starts. Torch native allocations don't fully return
    to the OS through ``gc.collect`` alone, but the GLiNER weights stay
    pinned by the process-wide ``shared_gliner`` cache and reused
    across builders / files, so the across-file high-water mark is
    bounded by one resident FP16 copy (~0.6 GB VRAM).
    """
    results: List[IndexBuildResult] = []
    for b in builders:
        res = _run_one_builder(file_id, b, pages, on_event)
        results.append(res)
        # Drop the builder's references (NER pipeline, embedding
        # client buffers, igraph snapshots) before the next builder
        # starts so peak RSS stays bounded by the largest single
        # builder, not the sum.
        gc.collect()
    return results


def parse_only(
    source: Union[str, Path],
    *,
    file_id: Optional[str] = None,
    overwrite: bool = False,
    parser: Optional[PdfParser] = None,
    on_event: Optional[OnEvent] = None,
) -> ParseResult:
    """Run just the parse stage (paddle OCR), emit ``stage`` events.

    Carved out of :func:`parse_and_index` so the API can hold a
    semaphore around concurrent paddle calls (parses are independent
    per file, the OCR service handles its own queue) while still
    grabbing the global ``INGEST_LOCK`` only for the index-write
    stage. Pure passthrough to ``PdfParser.parse`` plus the standard
    ``stage:parse`` start/done frames.
    """
    parser = parser or PdfParser()
    _safe_emit(on_event, "stage", {"stage": "parse", "phase": "start"})
    t_parse = time.perf_counter()
    try:
        parse = parser.parse(source, file_id=file_id, overwrite=overwrite)
    except Exception as exc:
        _safe_emit(
            on_event,
            "stage",
            {
                "stage": "parse",
                "phase": "done",
                "elapsed_ms": int((time.perf_counter() - t_parse) * 1000),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    _safe_emit(
        on_event,
        "stage",
        {
            "stage": "parse",
            "phase": "done",
            "elapsed_ms": int((time.perf_counter() - t_parse) * 1000),
            "items": parse.total_pages,
        },
    )
    logger.info(
        "parse done: file_id=%s pages=%d batches=%d",
        parse.file_id,
        parse.total_pages,
        len(parse.batches),
    )
    return parse


def index_parsed(
    parse: ParseResult,
    *,
    builders: Optional[Sequence[IndexBuilder]] = None,
    parallel_builders: bool = True,
    on_event: Optional[OnEvent] = None,
    linear_config: Optional[Any] = None,
) -> PipelineResult:
    """Run page_assets + the four builders against an already-parsed file.

    Companion to :func:`parse_only` — the API runs this under the
    global ``INGEST_LOCK`` (faiss / graphml stores are not
    safe under concurrent writes). The returned ``PipelineResult.parse``
    is the input ``parse`` so callers can keep the same shape.
    """
    t0 = time.perf_counter()
    builder_list = (
        list(builders)
        if builders is not None
        else _default_builders(linear_config=linear_config)
    )
    pages, results = _ingest_one(
        parse,
        builder_list,
        parallel_builders=parallel_builders,
        on_event=on_event,
    )
    failures = [r for r in results if r.failed]
    return PipelineResult(
        parse=parse,
        pages=pages,
        indexes=results,
        total_seconds=time.perf_counter() - t0,
        source=str(parse.source_path) if hasattr(parse, "source_path") else None,
        ok=not failures,
        error=(
            "builder(s) failed: " + ", ".join(f"{r.index_name} ({r.skipped_reason})" for r in failures)
            if failures
            else None
        ),
    )


def parse_and_index(
    source: Union[str, Path],
    *,
    file_id: Optional[str] = None,
    overwrite: bool = False,
    parser: Optional[PdfParser] = None,
    builders: Optional[Sequence[IndexBuilder]] = None,
    parallel_builders: bool = True,
    on_event: Optional[OnEvent] = None,
    linear_config: Optional[Any] = None,
) -> PipelineResult:
    """Single-file ingestion pipeline.

    Convenience wrapper: ``parse_only`` then ``index_parsed``. The
    two-step variant is the API surface for the web layer (parses can
    run concurrent under a semaphore; indexes must run serially under
    the global lock).

    ``parallel_builders=False`` runs the four index builders serially
    inside one file (see :func:`_ingest_one`); use this on memory-
    constrained hosts where the default fan-out OOMs.

    ``on_event`` is forwarded down for stage-level progress streaming.
    Algorithm callsites (notebooks, scripts, tests) leave it ``None``
    and the pipeline runs silently.

    ``linear_config`` (LinearRAGConfig) is forwarded to the default
    GraphIndexBuilder when ``builders`` is None. Ignored when
    ``builders`` is set explicitly — caller owns configuration there.
    """
    t0 = time.perf_counter()
    src_str = str(source)
    parse = parse_only(
        source,
        file_id=file_id,
        overwrite=overwrite,
        parser=parser,
        on_event=on_event,
    )
    result = index_parsed(
        parse,
        builders=builders,
        parallel_builders=parallel_builders,
        on_event=on_event,
        linear_config=linear_config,
    )
    # Restore the wrapper-level total_seconds and source so existing
    # callers that printed ``result.total_seconds`` / ``result.source``
    # see the expected shape.
    return PipelineResult(
        parse=result.parse,
        pages=result.pages,
        indexes=result.indexes,
        total_seconds=time.perf_counter() - t0,
        source=src_str,
        ok=result.ok,
        error=result.error,
    )


def parse_and_index_many(
    sources: Iterable[Union[str, Path]],
    *,
    overwrite: bool = False,
    parser: Optional[PdfParser] = None,
    builders: Optional[Sequence[IndexBuilder]] = None,
    parse_workers: int = 4,
    parallel_builders: bool = True,
) -> List[PipelineResult]:
    """Pipelined many-file pipeline.

    Parses run concurrently in a thread pool of size ``parse_workers``;
    ingest runs serially on the main thread, taking each parsed file as
    soon as it is ready (so ingest of file N runs while parses of files
    N+1, N+2 ... are still in flight).

    ``parallel_builders`` controls within-file builder concurrency
    (forwarded to :func:`_ingest_one`); set to False on memory-
    constrained hosts.
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
                pages, idx_results = _ingest_one(
                    parse,
                    builder_list,
                    parallel_builders=parallel_builders,
                )
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
            failures = [r for r in idx_results if r.failed]
            results[src] = PipelineResult(
                parse=parse,
                pages=pages,
                indexes=idx_results,
                total_seconds=time.perf_counter() - file_t0,
                source=str(src),
                ok=not failures,
                error=(
                    "builder(s) failed: " + ", ".join(
                        f"{r.index_name} ({r.skipped_reason})" for r in failures
                    )
                    if failures
                    else None
                ),
            )

    # Drain any builder-side deferred state. Builders with cadence-
    # based persistence (GraphIndexBuilder + reuse_graph=True) require
    # a final flush() or the last <cadence docs' graph/store/NER state
    # never lands on disk. Builders without a flush method (text_dense,
    # vision_dense, bm25) are no-ops here since they persist per call.
    for builder in builder_list:
        flush = getattr(builder, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                logger.exception("builder.flush() FAILED for %s", type(builder).__name__)

    n_ok = sum(1 for r in results.values() if r.ok)
    logger.info(
        "parse_and_index_many done: %d/%d files OK in %.1fs",
        n_ok,
        len(sources),
        time.perf_counter() - t0,
    )
    # Preserve input order; every source has a slot (incl. failures).
    return [results[s] for s in sources if s in results]
