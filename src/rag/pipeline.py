"""Top-level RAG orchestrator: query → preprocess → 4 channels → RRF → rerank → answer.

See ``docs/rag_pipeline.md`` for the full design and parameter rationale.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from config import RAGConfig
from config.settings import page_assets_root
from model_client import EmbeddingClient, LLMClient, RerankClient, VisualEmbeddingClient
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
    citations_pending: bool = True  # TODO: page-level citation post-processor


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
    ):
        self.config = config or RAGConfig()
        self.llm = llm or LLMClient()
        self.embedding_client = embedding_client or EmbeddingClient()
        self.visual_client = visual_client or VisualEmbeddingClient()
        self.rerank_client = rerank_client or RerankClient()
        self.page_store = page_store or self._default_page_store()

        if channels is None:
            channels = [
                SemanticChannel(
                    config=self.config,
                    embedding_client=self.embedding_client,
                    visual_client=self.visual_client,
                ),
                BM25Channel(config=self.config),
                GraphPPRChannel(
                    config=self.config,
                    embedding_client=self.embedding_client,
                ),
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
        """Pre-load lazily-initialized heavy resources (spaCy NER, …).

        Without this the first query absorbs the one-time cost — typically
        ~10-15 s for the en + zh transformer NER models — and its timing
        is not comparable to subsequent queries. Returns a per-component
        warm-up timing dict for visibility.
        """
        timings: Dict[str, float] = {}
        for ch in self.channels:
            ensure = getattr(ch, "_ensure_spacy", None)
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
    ) -> AnswerResult:
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
                        k: getattr(self.config, k)
                        for k in vars(self.config)
                        if not k.startswith("_")
                    },
                },
            )
        timings: Dict[str, float] = {}
        channel_timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        ctx = preprocess(query, llm=self.llm, file_ids=file_ids)
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

        t0 = time.perf_counter()
        channel_hits = self._retrieve_all(ctx, channel_timings=channel_timings)
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
            k=self.config.rrf_k,
            top_m=self.config.rrf_top_m,
        )
        timings["rrf"] = time.perf_counter() - t0
        if session is not None:
            session.snapshot(
                "fused",
                {
                    "rrf_k": self.config.rrf_k,
                    "top_m": self.config.rrf_top_m,
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

        reranked = rerank_pages(
            query=query,
            pages=candidate_pages,
            config=self.config,
            client=self.rerank_client,
        )
        timings["rerank"] = time.perf_counter() - t0
        if session is not None:
            session.snapshot(
                "rerank",
                {
                    "top_n": self.config.rerank_top_n,
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

        t0 = time.perf_counter()
        text = answer_call(
            query=query,
            pages=reranked,
            config=self.config,
            llm=self.llm,
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
    ) -> Dict[str, List[ChannelHit]]:
        """Run all channels in parallel; isolate failures so one channel's
        crash doesn't kill the rest of the query."""

        def _one(channel: BaseChannel) -> tuple[str, List[ChannelHit], float]:
            t0 = time.perf_counter()
            try:
                hits = channel.retrieve(ctx)
            except Exception as exc:
                logger.exception("channel %s failed: %s", channel.name, exc)
                hits = []
            return channel.name, hits, time.perf_counter() - t0

        with ThreadPoolExecutor(max_workers=max(4, len(self.channels))) as pool:
            results = list(pool.map(_one, self.channels))

        out: Dict[str, List[ChannelHit]] = {}
        for name, hits, elapsed in results:
            out[name] = hits
            if channel_timings is not None:
                channel_timings[name] = elapsed
        return out

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
