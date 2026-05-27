"""Dense retrieval over text + vision channels with RRF fusion.

Two backends share a query embedding (multimodal model) but probe
different stores:

* text channel — sentence-level rows in ``faiss_dense_dir()`` (each
  sentence carries ``file_id`` / ``page_id`` meta). Page score = the
  maximum sentence cosine on that page within the requested ``top_k``
  pool — same aggregator the standalone-RAG semantic channel uses.
* vision channel — one row per rendered page image in
  ``faiss_visual_dir()`` (queried via the multimodal text-image shared
  embedding space).

Per-channel rankings are fused with RRF (k=60) to produce the final
page-level list. We run the two channels in parallel inside a small
ThreadPoolExecutor so wall time is dominated by whichever embedder is
slower (text is usually fastest because the query embedding for vision
is the same multimodal call).

`channels` lets the agent disable one side at will (saves a vision
embedding call when the corpus is text-only). Default is both.

The agent is told in the schema description to *try HyDE-style
reformulations* — we do not invoke HyDE inside the tool because that
would consume an extra LLM turn the agent did not authorize.
"""

import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from config.shared import shared_tiktoken_encoder

from agentic.tools.acquisition._common import (
    err,
    make_snippet,
    ok,
    parse_scope,
    Scope,
)
from agentic.tools.base import BaseTool
from config.settings import faiss_dense_dir, faiss_visual_dir
from model_client import (
    EmbeddingClient,
    VisualEmbeddingClient,
    get_cached_embedding_client,
    get_cached_visual_embedding_client,
)
from storage import EmbeddingStore
from storage.embedding_store import get_or_create_store
from storage.inventory_store import InventoryStore
from storage.page_store import PageStore

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


logger = logging.getLogger(__name__)


_RRF_K = 60
_VALID_CHANNELS = {"text", "vision"}


class SemanticSearchTool(BaseTool):
    _embedding_lock = threading.Lock()

    def __init__(
        self,
        page_store: Optional[PageStore] = None,
        text_store_dir: Optional[Path] = None,
        vision_store_dir: Optional[Path] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        visual_client: Optional[VisualEmbeddingClient] = None,
        inventory: Optional[InventoryStore] = None,
    ):
        self.page_store = page_store
        self.inventory = inventory
        self._text_store_dir = Path(text_store_dir) if text_store_dir else faiss_dense_dir()
        self._vision_store_dir = Path(vision_store_dir) if vision_store_dir else faiss_visual_dir()
        self.embedding_client = embedding_client or get_cached_embedding_client()
        self.visual_client = visual_client or get_cached_visual_embedding_client()

        self._text_store: Optional[EmbeddingStore] = None
        self._vision_store: Optional[EmbeddingStore] = None
        self._tokenizer = shared_tiktoken_encoder("gpt-4o")

    # ------------------------------------------------------------- stores

    @property
    def text_store(self) -> EmbeddingStore:
        if self._text_store is None:
            # Process-cached — same handle as the ingest builder writes to.
            self._text_store = get_or_create_store(
                self._text_store_dir, namespace="dense"
            )
        return self._text_store

    @property
    def vision_store(self) -> EmbeddingStore:
        if self._vision_store is None:
            self._vision_store = get_or_create_store(
                self._vision_store_dir, namespace="visual"
            )
        return self._vision_store

    # ------------------------------------------------------------- schema

    @property
    def name(self) -> str:
        return "semantic_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": (
                    "Dense retrieval — text + vision channels fused via "
                    "reciprocal rank. Use when keyword search misses or "
                    "the literal phrasing isn't in the corpus. A HyDE-"
                    "style draft answer often beats the bare question. "
                    "Returns up to `top_k` page hits with abbreviated "
                    "snippets — `read` before quoting. Scope filters "
                    "(`file_ids`, `page_range`, `section_ids`) intersect. "
                    "`channels` disables one side (e.g. `['text']` for a "
                    "text-only corpus)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language query; HyDE-style draft answers often improve recall.",
                        },
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional file id allow-list.",
                        },
                        "page_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "Optional [start, end] inclusive 1-based page-number filter."
                            ),
                        },
                        "section_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of section ids "
                                "(e.g. '<file_id>:sec_003') from `toc`. "
                                "A page must lie inside at least one "
                                "to qualify."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max page hits to return; default 10, max 50.",
                        },
                        "channels": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["text", "vision"]},
                            "description": (
                                "Subset of channels to run. Default both. "
                                "Order does not matter."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    # ------------------------------------------------------------- execute

    def execute(
        self,
        context: "AgentContext",
        query: str,
        file_ids: Optional[List[str]] = None,
        page_range: Optional[List[int]] = None,
        section_ids: Optional[List[str]] = None,
        top_k: int = 10,
        channels: Optional[List[str]] = None,
    ):
        if not query or not str(query).strip():
            return err(
                "invalid_argument",
                "`query` must be a non-empty string.",
                remediation="Pass `query` as a non-empty natural-language string (a HyDE-style fact-rich draft answer often improves recall).",
                valid_example={"query": "What is the maximum AFYP rebate percentage?"},
            ), {"error": "invalid_argument"}
        scope, scope_err = parse_scope(
            file_ids, page_range, section_ids, inventory=self.inventory
        )
        if scope_err is not None:
            return err(
                "invalid_argument",
                scope_err,
                remediation="Fix the scope arguments per the message: file_ids must be a list of ids from list_files; page_range must be [start, end] with 1<=start<=end; section_ids must come from toc.",
                valid_example={"file_ids": ["<file_id>"], "page_range": [1, 50], "section_ids": ["<file_id>:sec_001"]},
            ), {"error": "invalid_argument"}
        if (scope.page_range is not None or scope.section_ranges is not None) and self.page_store is None:
            # Embedding stores carry only file_id/page_id meta; without a
            # PageStore we cannot resolve page_number, so a page_range or
            # section filter would silently zero out every hit.
            return (
                err(
                    "misconfigured",
                    "page_range / section_ids filtering requires a PageStore to be wired into the tool.",
                    remediation="Drop `page_range` and `section_ids` and rely on `file_ids` only; this deployment does not have a PageStore wired into semantic_search.",
                ),
                {"error": "misconfigured"},
            )

        try:
            top_k_int = int(top_k)
        except (TypeError, ValueError):
            return err(
                "invalid_argument",
                "`top_k` must be an integer.",
                remediation="Pass `top_k` as a positive integer (default 10, max 50).",
                valid_example={"top_k": 10},
            ), {"error": "invalid_argument"}
        if top_k_int < 1:
            return err(
                "invalid_argument",
                "`top_k` must be >= 1.",
                remediation="Set `top_k` to a positive integer (default 10, max 50).",
                valid_example={"top_k": 10},
            ), {"error": "invalid_argument"}
        limit = min(top_k_int, 50)

        wanted = self._normalize_channels(channels)
        if wanted is None:
            return (
                err(
                    "invalid_argument",
                    "`channels` must be a subset of ['text', 'vision'].",
                    remediation="Set `channels` to ['text'], ['vision'], ['text','vision'], or omit it (default both).",
                    valid_example={"channels": ["text", "vision"]},
                ),
                {"error": "invalid_argument"},
            )

        # Per-channel topk: aim for ~3x the final limit so RRF has room
        # to reorder. Capped because the text store's per-sentence rows
        # explode quickly.
        per_channel_topk = max(limit * 3, 15)

        # Per-channel errors are bookkept here so the agent can tell
        # "vision returned 0 hits because it failed" apart from "vision
        # returned 0 hits because there was nothing to find". Text-side
        # failures are caught here too — without this, an embedding HTTP
        # error would surface only as a generic tool_exception envelope
        # and the agent would lose the (possibly successful) vision
        # channel's contribution.
        channel_errors: Dict[str, str] = {}
        channel_hits: Dict[str, List[Tuple[str, float]]] = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures: Dict[str, Any] = {}
            if "text" in wanted:
                if not self._text_store_available():
                    channel_errors["text"] = "store_unavailable_or_empty"
                else:
                    futures["text"] = pool.submit(self._text_channel, query, per_channel_topk, scope)
            if "vision" in wanted:
                if not self._vision_store_available():
                    channel_errors["vision"] = "store_unavailable_or_empty"
                else:
                    futures["vision"] = pool.submit(self._vision_channel, query, per_channel_topk, scope)
            for name, fut in futures.items():
                try:
                    channel_hits[name] = fut.result()
                except Exception as exc:  # network, faiss, encoder failure — isolate
                    logger.warning("semantic_search: %s channel raised: %s", name, exc)
                    channel_errors[name] = f"{type(exc).__name__}: {exc}"
                    channel_hits[name] = []

        ran_channels = [name for name in channel_hits if name not in channel_errors]
        skipped = [c for c in wanted if c not in channel_hits and c not in channel_errors]

        fused = _rrf_fuse(channel_hits.values(), k=_RRF_K)[:limit]
        results = self._materialize(fused, channel_hits)

        retrieved_tokens = (
            len(self._tokenizer.encode("\n".join(r["snippet"] for r in results)))
            if results
            else 0
        )
        log_meta = {
            "query": query,
            "scope": scope.as_dict(),
            "top_k": limit,
            "channels_run": ran_channels,
            "channels_skipped": skipped,
            "channel_errors": channel_errors,
            "hits": len(results),
        }
        context.add_retrieval_log(
            tool_name="semantic_search", tokens=retrieved_tokens, metadata=log_meta
        )

        return (
            ok(
                "PageSearchObservation",
                tool="semantic_search",
                query=query,
                scope=scope.as_dict(),
                channels_run=ran_channels,
                channels_skipped=skipped,
                channel_errors=channel_errors,
                results=results,
            ),
            {
                "retrieved_tokens": retrieved_tokens,
                "hits": len(results),
                "channel_errors": list(channel_errors.keys()),
            },
        )

    # ------------------------------------------------------------- channels

    def _normalize_channels(self, channels: Optional[Sequence[str]]) -> Optional[set]:
        if channels is None:
            return set(_VALID_CHANNELS)
        cleaned = {str(c).strip().lower() for c in channels if str(c).strip()}
        if not cleaned:
            return set(_VALID_CHANNELS)
        if not cleaned.issubset(_VALID_CHANNELS):
            return None
        return cleaned

    def _text_store_available(self) -> bool:
        try:
            return len(self.text_store) > 0
        except Exception as exc:
            logger.warning("semantic_search: text store unavailable: %s", exc)
            return False

    def _vision_store_available(self) -> bool:
        try:
            return len(self.vision_store) > 0 and self.visual_client.available()
        except Exception as exc:
            logger.warning("semantic_search: vision store unavailable: %s", exc)
            return False

    def _text_channel(
        self, query: str, top_k: int, scope: Scope
    ) -> List[Tuple[str, float]]:
        store = self.text_store
        with self._embedding_lock:
            emb = self.embedding_client.encode(query, is_query=True)
        # Sentence-level rows cluster heavily on a few pages; pull deeper
        # so per-page max-aggregation still surfaces ``top_k`` distinct
        # pages after filtering. 8x covers the worst page-density we've
        # seen; the agent can raise top_k if it still wants more.
        depth = self._cap_depth(store, top_k * (10 if (scope.file_ids or scope.page_range or scope.section_ranges) else 8))
        scored = store.topk(emb, depth)
        return self._aggregate_by_page(store, scored, top_k, scope)

    def _vision_channel(
        self, query: str, top_k: int, scope: Scope
    ) -> List[Tuple[str, float]]:
        store = self.vision_store
        try:
            with self._embedding_lock:
                emb = self.visual_client.encode_text(query)
        except Exception as exc:
            logger.warning("semantic_search: vision encode_text failed: %s", exc)
            return []
        # Vision rows are already page-level (one row per page) so a
        # smaller depth multiplier suffices.
        depth = self._cap_depth(store, top_k * (5 if (scope.file_ids or scope.page_range or scope.section_ranges) else 3))
        scored = store.topk(emb, depth)
        return self._aggregate_by_page(store, scored, top_k, scope)

    @staticmethod
    def _cap_depth(store: EmbeddingStore, depth: int) -> int:
        # faiss raises if you ask for more than ntotal; clamp.
        try:
            n = len(store)
        except Exception:
            return depth
        return min(depth, n) if n else 1

    def _aggregate_by_page(
        self,
        store: EmbeddingStore,
        scored: List[Tuple[str, float]],
        top_k: int,
        scope: Scope,
    ) -> List[Tuple[str, float]]:
        # ``text_dense`` and ``vision_dense`` only persist file_id/page_id
        # (no page_number column), so resolving page_range OR section
        # filters requires a look-up against the PageStore. We do
        # file_id-only gating up front to avoid the lookup when no
        # number-based filter is requested.
        need_page_number = scope.page_range is not None or scope.section_ranges is not None
        per_page: Dict[Tuple[str, str], float] = {}
        for hash_id, score in scored:
            row = store.get_meta_row(hash_id)
            file_id = row.get("file_id")
            page_id = row.get("page_id")
            if not file_id or not page_id:
                continue
            file_id_s = str(file_id)
            page_id_s = str(page_id)
            if scope.file_ids is not None and file_id_s not in scope.file_ids:
                continue
            if need_page_number:
                pn_int = self._lookup_page_number(file_id_s, page_id_s)
                if not scope.contains(file_id_s, pn_int):
                    continue
            key = (file_id_s, page_id_s)
            cur = per_page.get(key, float("-inf"))
            if float(score) > cur:
                per_page[key] = float(score)
        # Score-desc, then (file_id, page_id) asc — a deterministic
        # tiebreaker so boilerplate-y pages with identical similarity
        # don't shuffle across runs (faiss does not promise a stable
        # tie order).
        ranked = sorted(
            per_page.items(),
            key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
        )
        return [(f"{fid}/{pid}", sc) for (fid, pid), sc in ranked[:top_k]]

    def _lookup_page_number(self, file_id: str, page_id: str) -> Optional[int]:
        if self.page_store is None:
            return None
        page = self.page_store.get(f"{file_id}/{page_id}")
        return page.page_number if page is not None else None

    # ------------------------------------------------------------- materialize

    def _materialize(
        self,
        fused: List[Tuple[str, float]],
        channel_hits: Dict[str, List[Tuple[str, float]]],
    ) -> List[Dict[str, Any]]:
        # Per-channel score lookup so each result row carries both the RRF
        # score and the raw cos sim from each channel that contributed.
        per_channel_lookup: Dict[str, Dict[str, float]] = {
            name: dict(hits) for name, hits in channel_hits.items()
        }
        results: List[Dict[str, Any]] = []
        for global_id, rrf_score in fused:
            file_id, _, page_id = global_id.partition("/")
            if not file_id or not page_id:
                continue
            page_number = None
            snippet = ""
            if self.page_store is not None:
                page = self.page_store.get(global_id)
                if page is not None:
                    page_number = page.page_number
                    snippet = make_snippet(page.text_markdown or "")
            row: Dict[str, Any] = {
                "file_id": file_id,
                "page_id": page_id,
                "page_number": page_number,
                "score": round(rrf_score, 6),
                "snippet": snippet,
            }
            for ch_name, lookup in per_channel_lookup.items():
                row[f"score_{ch_name}"] = (
                    round(lookup[global_id], 4) if global_id in lookup else None
                )
            row["matched_channels"] = [
                ch for ch, lookup in per_channel_lookup.items() if global_id in lookup
            ]
            results.append(row)
        return results


# --------------------------------------------------------------------- RRF


def _rrf_fuse(
    channels: "Sequence[List[Tuple[str, float]]]",
    k: int = _RRF_K,
) -> List[Tuple[str, float]]:
    """Reciprocal-rank fuse one or more (global_id, score) lists."""
    fused: Dict[str, float] = defaultdict(float)
    for hits in channels:
        for rank, (gid, _score) in enumerate(hits, start=1):
            fused[gid] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
