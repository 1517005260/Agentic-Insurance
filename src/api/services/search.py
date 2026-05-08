"""Advanced clause-search service.

Drives ``POST /search`` — a no-LLM retrieval surface that lets the
analyst pick which channels to fire, fuse them via RRF (or skip
fusion when there's only one), optionally rerank, and slice the
result snippets at page / passage / table_row granularity.

Why not reuse ``RAGPipeline.run``? The pipeline always runs all
configured channels, always preprocesses (HyDE + rewrite — two LLM
calls), and always rephrases via the answer-stage LLM. The search
surface is a lower-level "retrieval bench" that needs to be cheap
and channel-selective. We DO reuse:

* the channel objects already on ``pipeline.channels``,
* the RRF fuser,
* :func:`rerank_pages`,
* the page store + passage store + table row store,

so duplicate state is zero.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from rag.aggregate import rrf
from rag.channels.base import BaseChannel, ChannelHit
from rag.channels.bm25 import BM25Channel
from rag.channels.graph_ppr import GraphPPRChannel
from rag.channels.regex_scan import RegexChannel
from rag.channels.semantic import SemanticChannel
from rag.pipeline import RAGPipeline
from rag.preprocess import QueryContext, RegexSpec, detect_lang_local
from rag.rerank import rerank_pages
from storage.page_store import PageAsset, PageStore
from storage.passage_store import Passage


logger = logging.getLogger(__name__)


# Channel × granularity compatibility matrix. The PAGE-level rank
# is always derived from page-level channels; passage / table_row
# slicing is post-hoc on the page hits, so all four channels are
# compatible with all three granularities.
#
# We keep this matrix explicit because future channel changes may
# narrow it (e.g. a future "embedding-over-tables" channel would
# only be valid at table_row granularity). Each route call cross-
# checks here and 422s on mismatch.
_CHANNEL_GRANULARITY: Dict[str, Set[str]] = {
    "semantic": {"page", "passage", "table_row"},
    "bm25": {"page", "passage", "table_row"},
    "graph_ppr": {"page", "passage", "table_row"},
    "regex": {"page", "passage", "table_row"},
}


# Channels whose retrieval depends on prior LLM preprocessing. The
# no-LLM /search path leaves ``ctx.regexes=[]``, so RegexChannel
# would silently return zero hits — which is worse than refusing
# the channel up front (the user looks at a single-channel regex
# search, sees zero hits, and concludes the corpus has no matches).
_LLM_DEPENDENT_CHANNELS: Set[str] = {"regex"}


_CHANNEL_TYPE: Dict[str, type] = {
    "semantic": SemanticChannel,
    "bm25": BM25Channel,
    "graph_ppr": GraphPPRChannel,
    "regex": RegexChannel,
}


def _select_channels(
    pipeline: RAGPipeline, names: Sequence[str]
) -> List[BaseChannel]:
    """Pick the requested channels from the pipeline's instantiated set.

    We reuse ``pipeline.channels`` rather than constructing fresh
    objects so heavy state (faiss mmap, bm25 index, igraph) loads
    exactly once per process — same singleton sharing the chat
    surfaces use.

    Match by ``ch.name`` rather than ``type(ch)`` so a future test
    double / decorator subclass still resolves correctly. Each
    channel's ``name`` attribute is the source of truth.
    """
    by_name: Dict[str, BaseChannel] = {ch.name: ch for ch in pipeline.channels}
    chosen: List[BaseChannel] = []
    for name in names:
        ch = by_name.get(name)
        if ch is None:
            # The pipeline was built without that channel — surface
            # as an unavailable channel rather than silently dropping.
            raise ValueError(f"channel {name!r} is not configured on this pipeline")
        chosen.append(ch)
    return chosen


def validate_request(
    *, channels: Sequence[str], granularity: str
) -> Optional[str]:
    """Return a 422-worthy reason or None."""
    for c in channels:
        compat = _CHANNEL_GRANULARITY.get(c)
        if compat is None:
            return f"unknown channel {c!r}"
        if granularity not in compat:
            return (
                f"channel {c!r} not supported at granularity {granularity!r}; "
                f"compatible: {sorted(compat)}"
            )
    # The no-LLM search path can't synthesize regex specs (HyDE +
    # rewrite_regex_call would be the LLM-side source). Refuse the
    # regex channel here so the caller doesn't read a zero-hit
    # result as "no matches in the corpus" when actually the channel
    # had no patterns to scan with.
    llm_only = sorted(set(channels) & _LLM_DEPENDENT_CHANNELS)
    if llm_only:
        return (
            f"channel(s) {llm_only} require LLM-side preprocessing "
            f"(HyDE + regex synthesis) and are unavailable on the "
            f"no-LLM /search surface; use the chat /rag/stream surface "
            f"if you need regex hits"
        )
    return None


def _retrieve_subset(
    pipeline: RAGPipeline,
    ctx: QueryContext,
    channels: Sequence[BaseChannel],
) -> Tuple[Dict[str, List[ChannelHit]], Dict[str, float]]:
    """Like :meth:`RAGPipeline._retrieve_all` but a SUBSET of channels.

    Returns ``({channel_name: hits}, {channel_name: elapsed_seconds})``
    with crash isolation per channel.
    """
    timings: Dict[str, float] = {}
    hits: Dict[str, List[ChannelHit]] = {}

    def _one(channel: BaseChannel) -> Tuple[str, List[ChannelHit], float]:
        t0 = time.perf_counter()
        try:
            return channel.name, channel.retrieve(ctx), time.perf_counter() - t0
        except Exception:
            logger.exception("search: channel %s failed", channel.name)
            return channel.name, [], time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=max(2, len(channels))) as pool:
        futures = [pool.submit(_one, ch) for ch in channels]
        for fut in as_completed(futures):
            name, h, elapsed = fut.result()
            hits[name] = h
            timings[name] = elapsed
    # Preserve a stable iteration order in the returned dict
    # (Python dict preserves insertion; populate in the request's
    # channel order):
    return ({ch.name: hits.get(ch.name, []) for ch in channels}, timings)


def _apply_filters(
    page: PageAsset,
    *,
    file_ids: Optional[Sequence[str]],
    page_range: Optional[Tuple[int, int]],
    suffix_set: Optional[Set[str]],
    file_id_to_suffix: Dict[str, str],
) -> bool:
    """AND-compose every filter against a page asset."""
    if file_ids and page.file_id not in file_ids:
        return False
    if page_range is not None:
        pn = page.page_number
        if pn is None:
            return False
        lo, hi = page_range
        if pn < lo or pn > hi:
            return False
    if suffix_set is not None:
        suf = file_id_to_suffix.get(page.file_id, "")
        if suf not in suffix_set:
            return False
    return True


def _build_context(query: str, file_ids: Optional[Sequence[str]]) -> QueryContext:
    """Minimal QueryContext — no HyDE / rewrite / regex (no-LLM path).

    Channels degrade gracefully: semantic has no HyDE-augmented
    embedding, regex has no LLM-derived patterns. BM25 / graph_ppr
    are unaffected. Reasonable trade for a "fast, free" search bench.
    """
    return QueryContext(
        query=query,
        hyde="",
        rewrite=query,
        lang=detect_lang_local(query),
        regexes=[],
        file_ids=list(file_ids) if file_ids else None,
        # graph_ppr fallback is on for the search bench: no LLM has
        # extracted entities, so the gazetteer + Q2N embedding fallback
        # is the only way to seed PPR.
        enable_ppr_seed_fallback=True,
    )


def _slice_passages(
    passages: List[Passage], query: str, *, top: int = 1
) -> List[Passage]:
    """Pick the most query-relevant passages from a page's atom list.

    Cheap keyword-overlap scoring — we don't run another embedding
    pass here because the page-level rank has already been earned by
    the channels. This just tells the UI which passage to highlight.
    """
    if not passages:
        return []
    needles = [w.lower() for w in query.split() if len(w) >= 2]
    if not needles:
        return passages[:top]

    def _score(p: Passage) -> int:
        text_l = (p.text or "").lower()
        return sum(text_l.count(n) for n in needles)

    scored = sorted(passages, key=_score, reverse=True)
    # Drop zero-score passages unless none scored — better to surface
    # the first passage as a fallback than emit nothing.
    nonzero = [p for p in scored if _score(p) > 0]
    return (nonzero or passages)[:top]


def _build_snippet(
    page: PageAsset, query: str, *, max_chars: int = 240
) -> str:
    """First passage / line containing a query keyword; else first 240 chars."""
    text = page.text_markdown or ""
    if not text:
        return ""
    needles = [w.lower() for w in query.split() if len(w) >= 2]
    if needles:
        for line in text.splitlines():
            sl = line.lower().strip()
            if sl and any(n in sl for n in needles):
                return line[:max_chars] + ("…" if len(line) > max_chars else "")
    head = text.strip()
    return head[:max_chars] + ("…" if len(head) > max_chars else "")


def run_search(
    *,
    pipeline: RAGPipeline,
    query: str,
    channels: Sequence[str],
    granularity: str = "page",
    file_ids: Optional[Sequence[str]] = None,
    page_range: Optional[Tuple[int, int]] = None,
    suffixes: Optional[Sequence[str]] = None,
    rrf_k: Optional[int] = None,
    rrf_top_m: Optional[int] = None,
    top_n: Optional[int] = None,
    rerank: bool = False,
    file_id_to_suffix: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Drive the search pipeline; return a ``SearchResponse``-shaped dict.

    Caller (the route) wraps in ``run_in_threadpool`` because the
    channel calls + reranker are synchronous and blocking.
    """
    cfg = pipeline.config
    effective_k = int(rrf_k) if rrf_k is not None else cfg.rrf_k
    effective_top_n = int(top_n) if top_n is not None else cfg.rerank_top_n
    suffix_set: Optional[Set[str]] = (
        set(s.lower() for s in suffixes) if suffixes else None
    )
    file_id_set = set(file_ids) if file_ids else None
    file_id_to_suffix = file_id_to_suffix or {}

    # Filters that aren't pushed into ``QueryContext`` (page_range +
    # suffix) are applied AFTER RRF / single-channel ranking. To
    # avoid under-returning when the matching pages live below the
    # unfiltered top-M, overfetch when those filters are present.
    # file_ids IS pushed into QueryContext via _build_context, so it
    # doesn't need the overfetch budget.
    #
    # **Honest residual**: even at 10x overfetch a sparse filter
    # (e.g. ``page_range=[1,1]`` on a 500-page corpus) can still
    # miss matching candidates because the underlying retrieval
    # channels themselves cap at config-level top-Ks
    # (``semantic_channel_topk``, ``bm25_channel_topk``,
    # ``ppr_topk`` — defaults ~30 each). The fused candidate set
    # cannot exceed the union of those caps regardless of how
    # large ``rrf_top_m`` is set. For very narrow filters, prefer
    # ``file_ids`` (pushed into QueryContext, channels honor it
    # natively) or use the chat ``/rag/stream`` surface which
    # runs full LLM-augmented retrieval.
    #
    # The response carries ``post_filter_overfetched`` and
    # ``n_pre_filter`` so the caller can detect heavy pruning.
    has_post_filters = page_range is not None or suffix_set is not None
    overfetch_factor = 10 if has_post_filters else 1
    base_top_m = int(rrf_top_m) if rrf_top_m is not None else cfg.rrf_top_m
    effective_m = base_top_m * overfetch_factor

    timings_ms: Dict[str, int] = {}

    chosen_channels = _select_channels(pipeline, channels)
    ctx = _build_context(query, file_ids)

    t0 = time.perf_counter()
    channel_hits, ch_timings = _retrieve_subset(pipeline, ctx, chosen_channels)
    timings_ms["retrieve_total"] = int((time.perf_counter() - t0) * 1000)
    timings_ms.update(
        {f"retrieve.{n}": int(s * 1000) for n, s in ch_timings.items()}
    )

    # Build per-page channel-score map BEFORE any fusion — used for
    # the response's `channel_scores` annotation regardless of fusion.
    per_page_channel_scores: Dict[Tuple[str, str], Dict[str, float]] = {}
    for ch_name, hits in channel_hits.items():
        for h in hits:
            per_page_channel_scores.setdefault(h.key, {})[ch_name] = float(h.score)

    used_rrf = len(chosen_channels) >= 2
    if used_rrf:
        t0 = time.perf_counter()
        fused = rrf(list(channel_hits.values()), k=effective_k, top_m=effective_m)
        timings_ms["rrf"] = int((time.perf_counter() - t0) * 1000)
        # Enrich fused list with the per-page channel score map.
        ranked: List[Tuple[str, str, float]] = list(fused)
    else:
        # Single channel — no RRF; rank by channel score directly. Cap
        # at top_m to keep the candidate set bounded.
        only_channel = next(iter(channel_hits.values()))
        ranked = [
            (h.file_id, h.page_id, float(h.score))
            for h in only_channel[: effective_m]
        ]

    # Materialize page assets for snippet / filter / rerank.
    t0 = time.perf_counter()
    n_pre_filter = 0
    candidates: List[Tuple[Tuple[str, str], float, PageAsset]] = []
    for fid, pid, score in ranked:
        asset = pipeline.page_store.get(f"{fid}/{pid}") if pipeline.page_store else None
        if asset is None:
            continue
        n_pre_filter += 1
        if not _apply_filters(
            asset,
            file_ids=file_id_set,
            page_range=tuple(page_range) if page_range else None,
            suffix_set=suffix_set,
            file_id_to_suffix=file_id_to_suffix,
        ):
            continue
        candidates.append(((fid, pid), score, asset))
    timings_ms["filter_load"] = int((time.perf_counter() - t0) * 1000)

    # Optional rerank pass.
    used_rerank = bool(rerank) and bool(candidates)
    rerank_scores: Dict[Tuple[str, str], float] = {}
    if used_rerank:
        t0 = time.perf_counter()
        try:
            results = rerank_pages(
                query=query,
                pages=[c[2] for c in candidates],
                config=cfg,
                client=pipeline.rerank_client,
            )
            # Re-order candidates by rerank result; preserve fused
            # score in the response so the UI can show both.
            page_to_idx: Dict[str, int] = {}
            for i, (key, _, asset) in enumerate(candidates):
                page_to_idx[f"{asset.file_id}/{asset.page_id}"] = i
            new_order: List[Tuple[Tuple[str, str], float, PageAsset]] = []
            for r in results:
                key_str = f"{r.page.file_id}/{r.page.page_id}"
                idx = page_to_idx.get(key_str)
                if idx is None:
                    continue
                rerank_scores[(r.page.file_id, r.page.page_id)] = float(r.score)
                new_order.append(candidates[idx])
            # Pages the reranker dropped fall to the bottom in their
            # original order, so the UI can still see them if it cares.
            seen = {(a.file_id, a.page_id) for _, _, a in new_order}
            tail = [c for c in candidates if (c[2].file_id, c[2].page_id) not in seen]
            candidates = new_order + tail
        except Exception:
            logger.exception("search: rerank pass failed; falling back to fused order")
        timings_ms["rerank"] = int((time.perf_counter() - t0) * 1000)

    # Truncate to the response cap.
    n_total = len(candidates)
    candidates = candidates[:effective_top_n]

    # Assemble the response hits, slicing per granularity.
    t0 = time.perf_counter()
    hits_out: List[Dict[str, Any]] = []
    for (fid, pid), fused_score, asset in candidates:
        chan_scores = per_page_channel_scores.get((fid, pid), {})
        channels_hit = sorted(chan_scores.keys())
        rerank_score = rerank_scores.get((fid, pid))
        if granularity == "page":
            hits_out.append({
                "file_id": fid,
                "page_id": pid,
                "page_number": asset.page_number,
                "score": float(rerank_score if rerank_score is not None else fused_score),
                "channel_scores": chan_scores,
                "channels_hit": channels_hit,
                "snippet": _build_snippet(asset, query),
                "rerank_score": rerank_score,
            })
        elif granularity == "passage":
            picked = _passages_for_page(pipeline, fid, pid, query, top=1)
            if not picked:
                # Fall back to page-level row — better than emitting nothing.
                hits_out.append({
                    "file_id": fid, "page_id": pid,
                    "page_number": asset.page_number,
                    "score": float(rerank_score if rerank_score is not None else fused_score),
                    "channel_scores": chan_scores,
                    "channels_hit": channels_hit,
                    "snippet": _build_snippet(asset, query),
                    "rerank_score": rerank_score,
                })
                continue
            for p in picked:
                hits_out.append({
                    "file_id": fid, "page_id": pid,
                    "page_number": p.page_number or asset.page_number,
                    "passage_id": p.passage_id,
                    "score": float(rerank_score if rerank_score is not None else fused_score),
                    "channel_scores": chan_scores,
                    "channels_hit": channels_hit,
                    "snippet": (p.text or "")[:240],
                    "rerank_score": rerank_score,
                })
        elif granularity == "table_row":
            picked = _table_rows_for_page(pipeline, fid, pid, query, top=1)
            if not picked:
                # Symmetric to passage: no table-row cache (or no row
                # on this page) → fall back to a page-level row so
                # the caller still sees the candidate. Otherwise a
                # text-heavy page that ranked highly would be
                # silently dropped from a table_row search.
                hits_out.append({
                    "file_id": fid, "page_id": pid,
                    "page_number": asset.page_number,
                    "score": float(rerank_score if rerank_score is not None else fused_score),
                    "channel_scores": chan_scores,
                    "channels_hit": channels_hit,
                    "snippet": _build_snippet(asset, query),
                    "rerank_score": rerank_score,
                })
                continue
            for tr in picked:
                hits_out.append({
                    "file_id": fid, "page_id": pid,
                    "page_number": asset.page_number,
                    "table_row_id": tr.table_row_id,
                    "score": float(rerank_score if rerank_score is not None else fused_score),
                    "channel_scores": chan_scores,
                    "channels_hit": channels_hit,
                    "snippet": (tr.text or "")[:240],
                    "rerank_score": rerank_score,
                })
    timings_ms["snippet_build"] = int((time.perf_counter() - t0) * 1000)
    timings_ms["total"] = sum(
        v for k, v in timings_ms.items() if k in ("retrieve_total", "rrf", "filter_load", "rerank", "snippet_build")
    )

    return {
        "query": query,
        "granularity": granularity,
        "channels_run": list(channels),
        "filters_applied": {
            "file_ids": list(file_ids) if file_ids else None,
            "page_range": list(page_range) if page_range else None,
            "suffix": list(suffixes) if suffixes else None,
        },
        "hits": hits_out,
        "n_total": n_total,
        "n_returned": len(hits_out),
        "timings_ms": timings_ms,
        "used_rrf": used_rrf,
        "used_rerank": used_rerank,
        "rrf_k": effective_k if used_rrf else None,
        # Report the user-facing cap (base, NOT the overfetched value)
        # so the response is honest about what they asked for.
        "rrf_top_m": base_top_m if used_rrf else None,
        # Post-filter telemetry — when overfetched_factor > 1 the
        # caller can compare ``n_pre_filter`` (after RRF, before
        # post-filters) vs ``n_total`` (after post-filters) to see
        # how much the filter pruned. If the prune ratio is high,
        # widen rrf_top_m to capture more candidates.
        "post_filter_overfetched": has_post_filters,
        "n_pre_filter": n_pre_filter,
    }


# ---------- granularity expansion helpers ----------


def _passages_for_page(
    pipeline: RAGPipeline, file_id: str, page_id: str, query: str, *, top: int
) -> List[Passage]:
    """Best-effort passage slice; returns [] if the cache is missing."""
    inv = _inventory_store(pipeline)
    if inv is None:
        return []
    try:
        all_p = inv.passage_store.passages_for_page(file_id, page_id)
    except Exception:
        # PassageCacheMissing or other store-level error — silently
        # degrade rather than 500 the search.
        logger.debug("passages_for_page miss file=%s page=%s", file_id, page_id, exc_info=True)
        return []
    return _slice_passages(all_p, query, top=top)


def _table_rows_for_page(
    pipeline: RAGPipeline, file_id: str, page_id: str, query: str, *, top: int
) -> List[Any]:
    """Best-effort table_row slice; returns [] if the cache is missing."""
    inv = _inventory_store(pipeline)
    if inv is None:
        return []
    try:
        same_page = inv.table_row_store.rows_for_page(file_id, page_id)
    except Exception:
        logger.debug(
            "rows_for_page miss file=%s page=%s", file_id, page_id, exc_info=True
        )
        return []
    if not same_page:
        return []
    needles = [w.lower() for w in query.split() if len(w) >= 2]

    def _score(t: Any) -> int:
        text_l = (t.text or "").lower()
        return sum(text_l.count(n) for n in needles)

    if needles:
        same_page = sorted(same_page, key=_score, reverse=True)
        nonzero = [t for t in same_page if _score(t) > 0]
        same_page = nonzero or same_page
    return same_page[:top]


_INVENTORY_STORE_CACHE: Dict[int, Any] = {}


def _inventory_store(pipeline: RAGPipeline) -> Optional[Any]:
    """Reach the InventoryStore through the pipeline's page_store siblings.

    The pipeline doesn't own an inventory directly; the agents do.
    For the search service we accept reading from the lifespan-built
    InventoryStore by-instance — the route hands it in via the
    ``request.app.state.inventory`` if available; otherwise we
    construct a fresh one off the pipeline's page_store (idempotent
    since InventoryStore is cheap to build).
    """
    cached = _INVENTORY_STORE_CACHE.get(id(pipeline))
    if cached is not None:
        return cached
    try:
        from storage.inventory_store import InventoryStore

        inv = InventoryStore(page_store=pipeline.page_store)
        _INVENTORY_STORE_CACHE[id(pipeline)] = inv
        return inv
    except Exception:
        logger.exception("search: failed to build InventoryStore for granularity slice")
        return None
