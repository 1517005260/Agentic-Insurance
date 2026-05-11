"""Top-level RAG orchestrator: query → preprocess → 4 channels → RRF → rerank → answer."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from config import RAGConfig
from config.settings import page_assets_root
from model_client import (
    EmbeddingClient,
    LLMClient,
    RerankClient,
    VisualEmbeddingClient,
    get_cached_embedding_client,
    get_cached_rerank_client,
    get_cached_visual_embedding_client,
)
from rag.aggregate import FusedHit, rrf
from rag.answer import answer as answer_call
from rag.channels import (
    BM25Channel,
    BaseChannel,
    ChannelHit,
    GraphPPRChannel,
    RegexChannel,
    SemanticChannel,
)
from rag.preprocess import QueryContext, preprocess
from rag.rerank import RerankedPage, rerank_pages
from storage.page_store import PageAsset, PageStore
from tracer import Tracer, TraceSession


logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """Final answer plus everything needed to debug a single query."""

    query: str
    answer: str
    pages: List[RerankedPage]
    fused: List[FusedHit]
    channels: Dict[str, List[ChannelHit]]
    context: QueryContext
    timings: Dict[str, float] = field(default_factory=dict)
    channel_timings: Dict[str, float] = field(default_factory=dict)
    citations_pending: bool = True


class RAGPipeline:
    """Reusable pipeline — caches the heavy, channel-local state.

    A pipeline instance loads embedding stores, the LinearRAG graph, and the
    PageStore once and serves multiple queries. Channel objects expose
    ``retrieve(QueryContext)`` so the dispatch is trivial.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        llm: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        visual_client: Optional[VisualEmbeddingClient] = None,
        rerank_client: Optional[RerankClient] = None,
        page_store: Optional[PageStore] = None,
        channels: Optional[Sequence[BaseChannel]] = None,
        graph_channel: Optional[GraphPPRChannel] = None,
    ):
        self.config = config or RAGConfig()
        self.llm = llm or LLMClient()
        self.embedding_client = embedding_client or get_cached_embedding_client()
        self.visual_client = visual_client or get_cached_visual_embedding_client()
        self.rerank_client = rerank_client or get_cached_rerank_client()
        self.page_store = page_store or self._default_page_store()

        if channels is None:
            # ``graph_channel`` lets the web layer share ONE
            # GraphPPRChannel across the agent factories, GraphService,
            # AND the RAG pipeline so the graph is mmap'd exactly once
            # per process. The default-None path keeps the original
            # standalone behavior (experiment scripts get their own
            # channel as before).
            graph_ch = graph_channel or GraphPPRChannel(
                config=self.config,
                embedding_client=self.embedding_client,
            )
            channels = [
                SemanticChannel(
                    config=self.config,
                    embedding_client=self.embedding_client,
                    visual_client=self.visual_client,
                ),
                BM25Channel(config=self.config),
                graph_ch,
                RegexChannel(
                    config=self.config,
                    page_store=self.page_store,
                ),
            ]
        self.channels: List[BaseChannel] = list(channels)

    @staticmethod
    def _default_page_store() -> Optional[PageStore]:
        root = page_assets_root()
        if root.is_dir() and any(root.glob("*.json")):
            return PageStore(root)
        return None

    # ----------------------------------------------------------- warm-up ----

    def warm_up(self) -> Dict[str, float]:
        """Pre-load lazily-initialized heavy resources (GLiNER NER, …).

        Without this the first query absorbs the one-time cost —
        ~3-10 s for the GLiNER multi-v2.1 weights depending on HF cache
        warmth — and its timing is not comparable to subsequent queries.
        Returns a per-component warm-up timing dict for visibility.
        """
        timings: Dict[str, float] = {}
        for ch in self.channels:
            ensure = getattr(ch, "_ensure_ner", None)
            if not callable(ensure):
                continue
            t0 = time.perf_counter()
            try:
                ensure()
                timings[f"{ch.name}.ner"] = time.perf_counter() - t0
            except Exception as exc:
                logger.warning("warm_up: %s NER load failed: %s", ch.name, exc)
                timings[f"{ch.name}.ner"] = -1.0
        return timings

    # --------------------------------------------------------- public ----

    def run(
        self,
        query: str,
        file_ids: Optional[List[str]] = None,
        tracer: Optional[Tracer] = None,
        *,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
        citation_legend_provider: Optional[Callable[[Sequence], str]] = None,
        pages_block_provider: Optional[Callable[[Sequence], str]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        config_override: Optional[RAGConfig] = None,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> AnswerResult:
        """Run the full RAG pipeline.

        ``on_event`` (optional) is the streaming callback the API runner
        passes in. It fires at every stage boundary so the client can
        show "preprocessing → retrieving (4 channels) → reranking →
        answering" rather than a blank screen for the 5-30 s a query
        takes. Algorithm callers (rag_eval / scripts) leave it None.

        ``stream`` controls the answer stage: ``True`` triggers the
        ``LLMClient.chat_stream`` path, with each delta forwarded as a
        ``token`` event via ``on_event``. The function still returns a
        completed :class:`AnswerResult` once the stream drains.

        ``system_prompt`` overrides the answer-stage system prompt.
        ``citation_legend_provider`` is invoked AFTER rerank with the
        reranked page list and must return the legend string to inject
        into the user message — this is the seam the API runner uses
        to number ``[^k]`` references against the actual top-N. Both
        default None for algorithm callers.

        ``config_override`` (optional) replaces ``self.config`` for the
        duration of this call only — used by the web layer to apply
        admin-tunable RRF / rerank knobs without rebuilding the
        pipeline (which would re-load embedding stores). Passing
        ``None`` keeps the constructor config so experiment scripts
        continue to use the canonical ``RAGConfig()`` baseline.
        Channel objects keep the constructor-time config (their topks
        are static); we only override the cross-channel fusion + answer
        knobs that the run actually reads from this scope.
        """
        effective_config = config_override or self.config
        emit = _make_emitter(on_event)
        session = tracer.session(query) if tracer is not None else None
        if session is not None:
            # RAGConfig is static across runs; record once per day. The
            # per-query channel/rerank state is small but very dynamic
            # so it stays per-run via snapshot().
            session.daily(
                "config",
                {
                    "channels": [c.name for c in self.channels],
                    "rag_config": {
                        k: getattr(effective_config, k)
                        for k in vars(effective_config)
                        if not k.startswith("_")
                    },
                },
            )
        timings: Dict[str, float] = {}
        channel_timings: Dict[str, float] = {}

        emit("status", {"phase": "preprocess"})
        t0 = time.perf_counter()
        # Multi-turn: rewrite stage gets prior user queries (for
        # coreference resolution); answer stage gets full (q, a) pairs
        # below. HyDE is intentionally history-free (anchor-embedding
        # for the *current* topic, not a coreference-resolution task).
        history_user_only = (
            [q for q, _a in history] if history else None
        )
        # Pass the raw on_event (not wrapped emit) so preprocess can
        # short-circuit when no consumer is listening — same pattern
        # used by _retrieve_all below.
        ctx = preprocess(
            query,
            llm=self.llm,
            file_ids=file_ids,
            on_event=on_event,
            history_user_only=history_user_only,
        )
        # Multi-turn: when history is present, swap the standalone
        # rewrite into ctx.query so retrieval channels (semantic
        # text/vision, BM25, regex, graph_ppr) and rerank operate on
        # the coreference-resolved form. Without this, only BM25 +
        # regex (which use ctx.regexes / ctx.rewrite explicitly)
        # benefit from the rewrite — semantic text uses ctx.hyde or
        # ctx.query, semantic vision uses ctx.query, rerank uses raw
        # ``query`` — all of which would still be the ambiguous
        # "它的免赔额" form. Rerank-side query (``query`` local) is
        # likewise swapped below. Single-turn (history is None) keeps
        # the original ctx so byte-level identical with experiment
        # scripts.
        retrieval_query = query
        if history and ctx.rewrite and ctx.rewrite.strip() and ctx.rewrite.strip() != query.strip():
            from dataclasses import replace as _replace
            ctx = _replace(ctx, query=ctx.rewrite)
            retrieval_query = ctx.rewrite
        timings["preprocess"] = time.perf_counter() - t0
        if session is not None:
            session.snapshot(
                "preprocess",
                {
                    "query": ctx.query,
                    "hyde": ctx.hyde,
                    "rewrite": ctx.rewrite,
                    "lang": ctx.lang,
                    "regexes": [
                        {"pattern": r.pattern, "weight": r.weight, "rationale": r.rationale}
                        for r in ctx.regexes
                    ],
                    "file_ids": ctx.file_ids,
                    "elapsed_seconds": round(timings["preprocess"], 3),
                },
            )

        emit(
            "status",
            {
                "phase": "retrieve",
                "channels": [c.name for c in self.channels],
                "lang": ctx.lang,
                "regexes": len(ctx.regexes),
            },
        )
        t0 = time.perf_counter()
        # Pass the raw on_event through (not the wrapped emit). The
        # default path with on_event=None gets None here, and
        # _retrieve_all skips event payload construction entirely.
        # Avoids per-channel JSON-shape work on the experiment hot path.
        channel_hits = self._retrieve_all(
            ctx, channel_timings=channel_timings, emit=on_event,
        )
        timings["retrieve"] = time.perf_counter() - t0
        if session is not None:
            for name, hits in channel_hits.items():
                session.snapshot(
                    f"channels/{name}",
                    {
                        "channel": name,
                        "elapsed_seconds": round(channel_timings.get(name, 0.0), 3),
                        "hit_count": len(hits),
                        "hits": [
                            {
                                "rank": i + 1,
                                "file_id": h.file_id,
                                "page_id": h.page_id,
                                "score": h.score,
                                "evidence": h.evidence,
                            }
                            for i, h in enumerate(hits)
                        ],
                    },
                )

        t0 = time.perf_counter()
        fused = rrf(
            list(channel_hits.values()),
            k=effective_config.rrf_k,
            top_m=effective_config.rrf_top_m,
        )
        timings["rrf"] = time.perf_counter() - t0
        if session is not None:
            session.snapshot(
                "fused",
                {
                    "rrf_k": effective_config.rrf_k,
                    "top_m": effective_config.rrf_top_m,
                    "results": [
                        {"rank": i + 1, "file_id": fid, "page_id": pid, "score": score}
                        for i, (fid, pid, score) in enumerate(fused)
                    ],
                },
            )

        t0 = time.perf_counter()
        candidate_pages = self._load_pages(fused)
        if session is not None:
            session.snapshot(
                "candidates",
                {
                    "loaded": [
                        {
                            "file_id": p.file_id,
                            "page_id": p.page_id,
                            "page_number": p.page_number,
                            "text_chars": len(p.text_markdown or ""),
                        }
                        for p in candidate_pages
                    ],
                    "missing_from_fused": len(fused) - len(candidate_pages),
                },
            )

        emit("status", {"phase": "rerank", "candidates": len(candidate_pages)})
        reranked = rerank_pages(
            query=retrieval_query,  # standalone form when history is present
            pages=candidate_pages,
            config=effective_config,
            client=self.rerank_client,
        )
        timings["rerank"] = time.perf_counter() - t0
        emit(
            "reranked",
            {
                "elapsed_ms": int(timings["rerank"] * 1000),
                "pages": [
                    {
                        "rank": i + 1,
                        "file_id": r.page.file_id,
                        "page_id": r.page.page_id,
                        "page_number": r.page.page_number,
                        "score": r.score,
                    }
                    for i, r in enumerate(reranked)
                ],
            },
        )
        if session is not None:
            session.snapshot(
                "rerank",
                {
                    "top_n": effective_config.rerank_top_n,
                    "results": [
                        {
                            "rank": i + 1,
                            "file_id": r.page.file_id,
                            "page_id": r.page.page_id,
                            "page_number": r.page.page_number,
                            "score": r.score,
                        }
                        for i, r in enumerate(reranked)
                    ],
                },
            )

        legend: Optional[str] = None
        if citation_legend_provider is not None:
            try:
                legend = citation_legend_provider(reranked)
            except Exception:
                logger.exception("citation_legend_provider failed; proceeding without legend")
                legend = None

        pages_block_override: Optional[str] = None
        if pages_block_provider is not None:
            try:
                pages_block_override = pages_block_provider(reranked)
            except Exception:
                logger.exception(
                    "pages_block_provider failed; falling back to default page formatter"
                )
                pages_block_override = None

        emit("status", {"phase": "answering", "stream": stream})
        t0 = time.perf_counter()
        text = answer_call(
            query=query,
            pages=reranked,
            config=effective_config,
            llm=self.llm,
            system_prompt=system_prompt,
            citation_legend=legend,
            pages_block_override=pages_block_override,
            stream=stream,
            on_event=on_event,
            cancel_check=cancel_check,
            history=history,
        )
        timings["answer"] = time.perf_counter() - t0

        result = AnswerResult(
            query=query,
            answer=text,
            pages=reranked,
            fused=fused,
            channels=channel_hits,
            context=ctx,
            timings=timings,
            channel_timings=channel_timings,
        )
        if session is not None:
            session.finalize(
                answer=text,
                summary={
                    "timings": timings,
                    "channel_timings": channel_timings,
                    "channels_hit_counts": {n: len(h) for n, h in channel_hits.items()},
                    "fused_count": len(fused),
                    "candidate_count": len(candidate_pages),
                    "reranked_count": len(reranked),
                    "answer_chars": len(text),
                },
            )
        return result

    # --------------------------------------------------------- internals ----

    def _retrieve_all(
        self,
        ctx: QueryContext,
        channel_timings: Optional[Dict[str, float]] = None,
        emit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, List[ChannelHit]]:
        """Run all channels in parallel; isolate failures so one channel's
        crash doesn't kill the rest of the query.

        ``emit`` (when given) fires one ``retrieval`` event per channel
        as soon as it finishes — channels finish out of order, so the
        client sees progress sooner than waiting for the slowest. The
        callback is the same ``_make_emitter``-wrapped one used by
        :meth:`run`, so a None passed in is harmless.
        """

        def _one(channel: BaseChannel) -> tuple[str, List[ChannelHit], float]:
            t0 = time.perf_counter()
            try:
                hits = channel.retrieve(ctx)
            except Exception as exc:
                logger.exception("channel %s failed: %s", channel.name, exc)
                hits = []
            return channel.name, hits, time.perf_counter() - t0

        # Stage results by channel name as they finish (so ``retrieval``
        # events fire eagerly), then assemble the returned dict in the
        # original ``self.channels`` order. Downstream rrf() and trace
        # snapshots depend on this deterministic ordering.
        emit_safe = _make_emitter(emit)  # no-op if emit is None
        staged: Dict[str, List[ChannelHit]] = {}
        with ThreadPoolExecutor(max_workers=max(4, len(self.channels))) as pool:
            futures = [pool.submit(_one, ch) for ch in self.channels]
            for fut in as_completed(futures):
                name, hits, elapsed = fut.result()
                staged[name] = hits
                if channel_timings is not None:
                    channel_timings[name] = elapsed
                # Skip JSON-shape construction entirely when no
                # consumer is listening — keeps the experiment path's
                # ``timings["retrieve"]`` comparable to pre-streaming.
                if emit is not None:
                    emit_safe(
                        "retrieval",
                        {
                            "channel": name,
                            "elapsed_ms": int(elapsed * 1000),
                            "hits": [
                                {
                                    "file_id": h.file_id,
                                    "page_id": h.page_id,
                                    "score": h.score,
                                }
                                for h in hits
                            ],
                        },
                    )
        return {ch.name: staged[ch.name] for ch in self.channels}

    def _load_pages(self, fused: Sequence[FusedHit]) -> List[PageAsset]:
        if self.page_store is None:
            return []
        out: List[PageAsset] = []
        for file_id, page_id, _ in fused:
            asset = self.page_store.get(f"{file_id}/{page_id}")
            if asset is not None:
                out.append(asset)
        return out


def answer_query(
    query: str,
    file_ids: Optional[List[str]] = None,
    *,
    pipeline: Optional[RAGPipeline] = None,
    tracer: Optional[Tracer] = None,
) -> AnswerResult:
    """Convenience: run a single query against a default pipeline."""
    pipe = pipeline or RAGPipeline()
    return pipe.run(query, file_ids=file_ids, tracer=tracer)


def _make_emitter(
    on_event: Optional[Callable[[str, Dict[str, Any]], None]],
) -> Callable[[str, Dict[str, Any]], None]:
    """No-op when ``on_event`` is None; otherwise swallow callback errors.

    Mirrors :func:`agentic.agent.base._make_emitter`. We re-implement
    locally rather than importing across the algorithm-layer boundary
    so ``rag/`` has no dependency on ``agentic/``.
    """
    if on_event is None:
        def _noop(_event: str, _data: Dict[str, Any]) -> None:
            return
        return _noop

    def _emit(event: str, data: Dict[str, Any]) -> None:
        try:
            on_event(event, data)
        except Exception:
            logger.exception("on_event callback failed for %s", event)

    return _emit
