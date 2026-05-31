"""Entity-graph retrieval over the LinearRAG-style graph.

Three modes:

* ``ppr`` — question-driven personalized PageRank over the entity ↔
  passage graph, identical to the standalone-RAG ``graph_ppr`` channel.
  We delegate to :class:`rag.channels.GraphPPRChannel` so the two paths
  cannot drift apart in subtle ways (NER → seed entities → entity-score
  BFS → passage scoring → PPR).
* ``chain`` — query-time typed-edge beam search. From union seeds
  (NER ∪ gazetteer ∪ question-embedding top-k) the search expands one
  hop at a time over entity-typed neighbors. Each candidate edge
  ``(tail, neighbor)`` is scored by the cos-similarity between the
  question embedding and the embeddings of the sentences where
  ``tail`` and ``neighbor`` co-occur — the natural-language sentence
  IS the implicit predicate, so the relation type is materialised
  per-query without ever invoking an LLM at build time. Paths are
  scored by mean log-edge-probability + min-link penalty +
  question-coverage − hub-degree-penalty; the top-K paths are
  returned together with their via-sentence snippets and a derived
  candidate-page list (entities → incident passages, weighted by
  path mass).
* ``entity_lookup`` — embedding-match a surface form to *physical*
  entity nodes, and surface each hit's *logical* (alias-cluster)
  referent. Same gradient-topk + min-sim machinery used by the
  build-time disambiguator (`gradient_topk_candidates`), so the
  semantics are stable across query / index time.

Physical vs logical entity:
  Each surface form is indexed as its own node ("AXA", "AXA Hong Kong",
  "安盛"). Alias edges between near-synonyms partition the entity layer
  into logical clusters; the cluster is the LOGICAL entity. Use
  ``entity_lookup`` to map a question term to physical hits + logical
  cluster, then feed the question into ``chain`` for multi-hop
  navigation conditioned on the question.

Pre-warming:
  GLiNER NER and igraph are heavy to load. The agent's
  ``warm_up()`` invokes :meth:`warm_up` on this tool to absorb that
  one-time cost before the user's first turn.

Lifetime assumption:
  The graph and passage stores are treated as immutable for the life
  of the agent process. ``_passage_meta_lookup`` /
  ``_passage_hash_by_meta`` / ``_cluster_index`` are cached on the
  tool instance after first use. If the corpus is re-ingested
  mid-session, construct a fresh tool instance — the caches are not
  invalidated automatically.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from agentic.tools.acquisition._common import err, ok, parse_scope
from agentic.tools.base import BaseTool
from config.settings import faiss_graph_dir
from ingestion.index.linear_rag.backfill import find_literal_matches
from ingestion.index.linear_rag.disambig import (
    compute_clusters_for_collapse,
    get_clusters,
    gradient_topk_candidates,
    load_reverse_map,
)
from ingestion.index.linear_rag.normalize import normalize_for_hash
from rag.channels.base import RawHit, aggregate_per_doc
from rag.channels.graph_ppr import GraphPPRChannel
from rag.preprocess import QueryContext
from storage.inventory_store import InventoryStore

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


logger = logging.getLogger(__name__)


_VALID_MODES = {"ppr", "chain", "entity_analysis"}
# Agent path default: 5 candidates is enough for the typical
# "see top hits → read 2-3 of them" workflow. The 20-default historical
# value was tuned for the workbench UI (which renders all 20 as a
# ranked list). For agent observations, returning 20 candidate_pages
# + per-page evidence preview balloons the message-stack token
# budget. The workbench can still pass top_k=20 explicitly.
_DEFAULT_TOP_K = 5
_MAX_TOP_K = 50
# How many chars of passage text to expose per candidate as a preview.
# Just enough for the agent to discriminate "yes this is the right
# page, read it" from "wrong page, skip". The full page comes via
# the ``read`` tool.
_PPR_PREVIEW_CHARS = 240
_DEFAULT_CHAIN_DEPTH = 2
_MAX_CHAIN_DEPTH = 3

# Beam-search caps for ``chain`` mode. Picked to keep one query under
# ~100 ms on a 50-doc KG (~20 k entities, 50 k edges) with vectorised
# sentence-sim lookup; the log-mean-exp_β aggregator below uses these
# bounds to keep per-hop work near-linear in beam width.
_CHAIN_TOP_L_NEIGHBORS = 20  # max entity neighbors per node per hop
_CHAIN_TOP_M_SENTENCES = 8   # max via-sentences considered per edge
_CHAIN_BEAM_K = 32           # beam width (paths kept between hops)
_CHAIN_MAX_SEEDS = 8         # cap on union-seed fan-out

# log-mean-exp_β + sigmoid edge-score knobs. β controls how much the
# aggregator leans toward the strongest sentence; γ down-weights pairs
# with many supporting sentences (hub-pair noise); α / τ calibrate the
# sigmoid so the typical-edge score lands in mid-range.
_CHAIN_BETA = 5.0
_CHAIN_GAMMA = 0.05
_CHAIN_ALPHA = 8.0
_CHAIN_TAU = 0.30

# Path-score weights.
_CHAIN_W_MIN = 0.5      # weak-link penalty (κ)
_CHAIN_W_COVER = 1.0    # seed coverage bonus (λ)
_CHAIN_W_HUB = 0.2      # hub-degree penalty (ρ)


class GraphExploreTool(BaseTool):
    def __init__(
        self,
        channel: Optional[GraphPPRChannel] = None,
        inventory: Optional[InventoryStore] = None,
        *,
        entity_lookup_min_sim: float = 0.6,
        entity_lookup_gradient: float = 0.5,
    ):
        # The channel owns graph + entity/passage/sentence stores + NER
        # state. We hold a reference instead of re-loading everything,
        # which both saves memory and guarantees the two retrieval paths
        # operate on the same data.
        self._channel = channel or GraphPPRChannel()
        self.inventory = inventory
        # Entity lookup tunables — admin config injects these via the
        # factory (build_graph_agent → ConfigStore). The defaults here
        # (_ENTITY_LOOKUP_MIN_SIM=0.6, gradient g=0.5) keep call sites
        # that construct the tool directly (experiment scripts) working
        # without any config plumbing.
        self._entity_lookup_min_sim = float(entity_lookup_min_sim)
        self._entity_lookup_gradient = float(entity_lookup_gradient)
        self._clusters_cache_path: Path = faiss_graph_dir() / "clusters.json"
        self._reverse_map_path: Path = faiss_graph_dir() / "reverse_map.json"
        self._cluster_index: Optional[Dict[str, Dict[str, Any]]] = None
        self._reverse_map_cache: Optional[Dict[str, str]] = None
        # Passage meta is stable for the life of the agent (graph is
        # built once at ingest). Cache the hash<->meta maps so chain
        # mode doesn't rebuild them per candidate-page render.
        self._passage_meta_by_hash: Optional[Dict[str, Tuple[str, Optional[int]]]] = None
        self._passage_hash_by_meta: Optional[Dict[Tuple[str, Optional[int]], str]] = None

    @property
    def name(self) -> str:
        return "graph_explore"

    # ---------------------------------------------------------- invalidate

    def invalidate_caches(self) -> None:
        """Drop the per-instance derived caches.

        ``_passage_meta_by_hash`` / ``_passage_hash_by_meta`` reflect a
        snapshot of the channel's passage_store at first use; after a
        reingest or delete the store has different rows and the cached
        maps return stale (file_id, page_number) pairs. Same story for
        ``_cluster_index`` — it's loaded from
        ``faiss/graph/clusters.json`` which is rewritten on ingest.

        The wrapping ``GraphPPRChannel.reload()`` rebuilds the underlying
        stores; this method is the second half of "make a stale tool
        instance look fresh again" and should be called from the
        lifespan refresh hook.
        """
        self._passage_meta_by_hash = None
        self._passage_hash_by_meta = None
        self._cluster_index = None
        self._reverse_map_cache = None

    # ----------------------------------------------------------- warm-up

    def warm_up(self) -> None:
        """Pre-load GLiNER NER and force igraph to be in memory.

        Both the PPR and chain modes need the NER model on the first
        call, which is the single biggest source of cold-start latency
        in the agent loop (~3-10 s for GLiNER multi-v2.1 depending on
        HF cache warmth). Run this before the agent's first user turn.
        """
        if self._channel.graph is None:
            return  # nothing to warm — graph not built
        try:
            self._channel._ensure_ner()
        except Exception as exc:
            logger.warning("graph_explore: NER warm-up failed: %s", exc)

    # ----------------------------------------------------------- schema

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "graph_explore",
                "description": (
                    "Navigate the entity knowledge graph. Modes:\n"
                    "- ppr: free-text query -> ranked candidate pages "
                    "with a question-conditioned snippet per page.\n"
                    "- chain: typed-edge bridge between entities; seed "
                    "via `question`, `from_page=\"file_id/p_NNNN\"`, or "
                    "`from_entities=[...]`; optionally filter with "
                    "`to_entities=[...]`.\n"
                    "- entity_analysis: resolve a `surface` to its "
                    "entity cluster + alias members + cross-doc top "
                    "pages; alternatively pass a `cluster_id` for "
                    "deep audit, or omit both to enumerate top "
                    "clusters by size or mention_weight."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": sorted(_VALID_MODES),
                        },
                        "question": {
                            "type": "string",
                            "description": "Free-text query (mode=ppr | chain).",
                        },
                        "surface": {
                            "type": "string",
                            "description": "Entity surface form (mode=entity_analysis).",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Beam depth 1-3 (mode=chain), default 2.",
                        },
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional file id allow-list (mode=ppr | chain).",
                        },
                        "page_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional [start, end] inclusive page filter (mode=ppr).",
                        },
                        "section_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional section ids '<file_id>:sec_NNN' from `toc` "
                                "(mode=ppr). ANDed with file_ids and page_range."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max candidates returned; default 5, max 50.",
                        },
                        "cluster_id": {
                            "type": "string",
                            "description": "Cluster to deep-dive (mode=entity_analysis, with cluster_id only).",
                        },
                        "from_page": {
                            "type": "string",
                            "description": "Seed chain from a page's entities (mode=chain), format 'file_id/p_NNNN'.",
                        },
                        "from_entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Explicit seed surfaces (mode=chain).",
                        },
                        "to_entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter paths to those reaching any listed surface (mode=chain).",
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": ["size", "mention_weight"],
                            "description": "Ranking metric (mode=entity_analysis, when no surface or cluster_id is given); default 'size'.",
                        },
                        "surface_filter": {
                            "type": "string",
                            "description": "Substring filter against cluster surfaces (mode=entity_analysis, when no surface or cluster_id is given).",
                        },
                    },
                    "required": ["mode"],
                },
            },
        }

    # ----------------------------------------------------------- execute

    def execute(self, context: "AgentContext", **kwargs):
        mode = str(kwargs.get("mode") or "").strip().lower()
        if mode not in _VALID_MODES:
            return (
                err(
                    "invalid_argument",
                    f"`mode` must be one of {sorted(_VALID_MODES)}.",
                    remediation="Set `mode` to 'ppr' (free-text PPR), 'chain' (typed-edge bridge), or 'entity_analysis' (surface/cluster resolution).",
                    valid_example={"mode": "chain", "question": "..."},
                ),
                {"error": "invalid_argument"},
            )
        if self._channel.graph is None:
            return (
                err(
                    "graph_unavailable",
                    "LinearRAG graph is not built; ingest the corpus first.",
                    remediation="The entity graph is not available in this corpus.",
                ),
                {"error": "graph_unavailable"},
            )

        if mode == "ppr":
            return self._run_ppr(context, **kwargs)
        if mode == "chain":
            return self._run_chain(context, **kwargs)
        return self._run_entity_analysis(context, **kwargs)

    def _run_entity_analysis(self, context: "AgentContext", **kwargs):
        """Single mode for all logical-entity (alias-cluster) operations.

        Replaces the prior ``entity_lookup`` / ``cluster_inspect`` /
        ``list_clusters`` mode trio: the dispatch is by argument shape,
        not by an explicit mode name the agent has to memorise. Three
        argument shapes:

        * ``surface="Microsoft"`` → resolve the surface to its physical
          entity hits + each hit's logical cluster (members, audit,
          top pages where any member appears). This is the most common
          path — translate a question-side surface into a cluster
          handle + immediate cross-doc page anchors.
        * ``cluster_id="c_2436"`` (no surface) → deep audit of one
          cluster (members, top passages, alias quality, co-occurring
          clusters). Used after PPR's ``clusters_touched`` flags a
          cluster the agent wants to inspect.
        * ``surface=""`` and ``cluster_id=""`` → enumerate the top
          clusters in the corpus by ``sort_by={size, mention_weight}``
          with optional ``surface_filter``. Discovery / audit path.

        Old dispatches retained as private ``_run_entity_lookup`` /
        ``_run_cluster_inspect`` / ``_run_list_clusters`` for direct
        callers; the agent sees one ``entity_analysis`` mode.
        """
        surface = (kwargs.get("surface") or "").strip()
        cluster_id = (kwargs.get("cluster_id") or "").strip()
        if cluster_id and surface:
            return err(
                "invalid_argument",
                "mode='entity_analysis' accepts EITHER `surface` OR `cluster_id`, not both.",
                remediation=(
                    "Drop one: pass `surface` to resolve a name to its "
                    "cluster, OR pass `cluster_id` to deep-audit a "
                    "known cluster. Mixing them is ambiguous."
                ),
                valid_example={"mode": "entity_analysis", "surface": "AXA"},
            ), {"error": "invalid_argument"}
        if cluster_id:
            obs, meta = self._run_cluster_inspect(context, **kwargs)
        elif surface:
            obs, meta = self._run_entity_lookup(context, **kwargs)
        else:
            obs, meta = self._run_list_clusters(context, **kwargs)
        # Rewrite the inner ``mode`` field to the public name so the
        # agent's transcript only ever sees the 3 canonical modes
        # ({ppr, chain, entity_analysis}) it was told about. The
        # private dispatchers still emit the old internal mode names
        # for debug / log clarity.
        try:
            import json as _json
            payload = _json.loads(obs) if isinstance(obs, str) else obs
            if isinstance(payload, dict) and "mode" in payload:
                payload["mode"] = "entity_analysis"
                obs = _json.dumps(payload, default=str) if isinstance(obs, str) else payload
        except Exception:
            pass
        return obs, meta

    # ----------------------------------------------------------- PPR mode

    def _run_ppr(self, context: "AgentContext", **kwargs):
        question = (kwargs.get("question") or "").strip()
        if not question:
            return err(
                "invalid_argument",
                "mode='ppr' requires `question`.",
                remediation="Add `question` (free-text natural-language query) to the call; PPR uses NER to seed the random walk.",
                valid_example={"mode": "ppr", "question": "Which sections discuss premium rebates?"},
            ), {"error": "invalid_argument"}
        scope, scope_err = parse_scope(
            kwargs.get("file_ids"),
            kwargs.get("page_range"),
            kwargs.get("section_ids"),
            inventory=self.inventory,
        )
        if scope_err is not None:
            return err(
                "invalid_argument",
                scope_err,
                remediation="Fix the scope arguments per the message: file_ids must come from list_files; page_range must be [start, end]; section_ids must come from toc.",
                valid_example={"file_ids": ["<file_id>"]},
            ), {"error": "invalid_argument"}

        try:
            top_k_int = int(kwargs.get("top_k", _DEFAULT_TOP_K))
        except (TypeError, ValueError):
            return err(
                "invalid_argument",
                "`top_k` must be an integer.",
                remediation="Pass `top_k` as a positive integer (default 20, max 50).",
                valid_example={"top_k": 20},
            ), {"error": "invalid_argument"}
        if top_k_int < 1:
            return err(
                "invalid_argument",
                "`top_k` must be >= 1.",
                remediation="Set `top_k` to a positive integer (default 20, max 50).",
                valid_example={"top_k": 20},
            ), {"error": "invalid_argument"}
        limit = min(top_k_int, _MAX_TOP_K)

        ctx = QueryContext(
            query=question,
            hyde="",
            rewrite="",
            lang="",
            regexes=[],
            file_ids=list(scope.file_ids) if scope.file_ids else None,
            # Agent path: PPR is the only signal here, so let the channel
            # rescue NER misses with the gazetteer + question-embedding
            # fallbacks. The 4-channel RAG path leaves this off.
            enable_ppr_seed_fallback=True,
        )
        try:
            # Atomic ``hits + debug snapshot`` — the channel singleton
            # is shared across all 3 agent kinds AND the RAG pipeline,
            # so reading ``self._channel.last_debug`` after the call is
            # racy: another concurrent retrieve() could overwrite it.
            # ``retrieve_with_debug`` holds the channel's RLock for both
            # the call and the snapshot.
            channel_hits, debug_snapshot = self._channel.retrieve_with_debug(ctx)
        except Exception as exc:
            logger.exception("graph_explore[ppr] failed: %s", exc)
            return err(
                "ppr_failed",
                f"PPR raised: {exc}",
                remediation="Retry with a simpler question; if the PPR channel keeps failing, switch to mode=entity_analysis (surface lookup) or mode=chain.",
            ), {"error": "ppr_failed"}

        # Apply the page_range gate post-hoc (PPR channel honors file_ids
        # but not page_range — the channel was designed for the global RAG
        # path which doesn't expose ranges). Reuse the cached meta map.
        # IMPORTANT: keep ``kept_hits`` (the ChannelHit slice that
        # passed scope) in lock-step with ``results``, because
        # _attach_previews / per-page provenance below ZIP these two
        # together to look up evidence.passage_hash_id per hit.
        # If these two drift apart (e.g. zipping unfiltered hits with
        # the filtered ``results``), scoped queries silently shift
        # preview/clusters_touched onto the wrong page.
        meta_by_hash = self._passage_meta_lookup()

        # Pass 1: scope-filter the channel hits without truncating to
        # ``limit`` yet — the doc-aware rerank below needs the full
        # eligible slate so that a doc with several lower-ranked pages
        # can still outrank a doc with one stronger page that would
        # otherwise be the only one to fit in the top-K.
        eligible: List[Tuple[Any, Optional[int]]] = []
        for hit in channel_hits:
            page_number = self._first_page_number(hit, meta_by_hash)
            if scope.contains(hit.file_id, page_number):
                eligible.append((hit, page_number))

        # Cross-doc duplicate guard: when several pages from the same
        # doc each score independently, aggregate them via
        # ``Σ s_i / sqrt(N + 1)`` (rag.channels.base.aggregate_per_doc)
        # and re-sort pages by (doc_score, original_page_score). Pages
        # from the highest-aggregated doc rise to the top; pages within
        # one doc keep their PPR order. Targets the failure mode where
        # a near-duplicate from a wrong doc outranks the gold doc's
        # several supporting pages on raw PPR score alone.
        eligible = self._doc_aware_rerank(eligible, limit)

        # Per-doc page roll-up over the FULL eligible slate (pre-truncation
        # so docs with several lower-ranked pages still surface in the
        # rollup). Powers the ``docs_summary`` field — the agent's main
        # batch-read signal: "doc X has K pages in the top-K, range
        # p_lo..p_hi" lets it issue one ``read(unit_ids=[…])`` for all
        # siblings in one shot, instead of reading only 1-2 of several
        # gold pages on the same doc.
        doc_pages_full: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for hit, page_number in eligible:
            if page_number is None:
                continue
            doc_pages_full[hit.file_id].append(
                (int(page_number), float(hit.score))
            )

        results: List[Dict[str, Any]] = []
        kept_hits: List[Any] = []
        for hit, page_number in eligible:
            results.append(
                {
                    "file_id": hit.file_id,
                    "page_id": hit.page_id,
                    "page_number": page_number,
                    "score": round(float(hit.score), 6),
                }
            )
            kept_hits.append(hit)
            if len(results) >= limit:
                break

        # Cap doc_pages_full to the doc set that actually appears in
        # ``results`` (i.e. survives the top-K cut). Including docs the
        # agent can't reach via the visible candidate list is noise.
        visible_docs = {r["file_id"] for r in results}
        doc_pages_full = {
            fid: ps for fid, ps in doc_pages_full.items() if fid in visible_docs
        }

        seeds_dbg = debug_snapshot.get("seeds", [])
        # Attach question-conditioned previews — single highest
        # cos(q_emb, sent_emb) sentence per candidate page. Decisive
        # for the agent's first read selection versus a first-N-chars
        # preview, which surfaces table headers or references
        # regardless of the question.
        results_with_preview = self._attach_previews(results, kept_hits, question=question)

        # Seed surfaces (lowercased once) — used for per-page
        # ``supported_by`` and the global ``seeds_unsupported`` diagnostic.
        # Substring check, same heuristic ``_calculate_passage_scores``
        # uses internally — cheap on top-K * |seeds| ~= 5 * 8.
        seed_surfaces: List[str] = []
        for s in (seeds_dbg if isinstance(seeds_dbg, list) else []):
            surf = (s.get("surface") or "").strip()
            if surf:
                seed_surfaces.append(surf)
        seeds_with_any_support: set = set()

        # L3 + L6: per-page logical-cluster provenance + corpus breadth
        # (pages_in_cluster) so the agent can distinguish a specific
        # anchor entity (≤ 5 pages) from a hub cluster spanning the
        # corpus (≥ 100 pages) that PPR drifted into.
        for r, hit in zip(results_with_preview, kept_hits):
            ph = None
            for ev in (getattr(hit, "evidence", None) or []):
                if isinstance(ev, dict) and ev.get("passage_hash_id"):
                    ph = ev["passage_hash_id"]
                    break
            # ``supported_by`` — which seed surfaces actually appear on
            # this passage's text. A page with high PPR score but no
            # seed support was brought in by graph diffusion through a
            # hub, not by direct match — a strong demote signal that
            # the agent today can only guess from the preview.
            page_text = ""
            if ph is not None:
                page_text = self._channel.passage_store.hash_id_to_text.get(ph, "") or ""
            page_text_lower = page_text.lower()
            supported: List[str] = []
            for surf in seed_surfaces:
                if surf.lower() in page_text_lower:
                    supported.append(surf)
                    seeds_with_any_support.add(surf)
            # Cap to top 3 (most informative; whole list bloats the obs).
            r["supported_by"] = supported[:3]

            if ph is None:
                r["clusters_touched"] = []
                continue
            try:
                clusters = self._channel.passage_top_clusters(ph, top_n=3)
                for c in clusters:
                    cid = c.get("cluster_id")
                    if cid:
                        c["pages_in_cluster"] = self._channel.cluster_passage_count(cid)
                r["clusters_touched"] = clusters
            except Exception:
                r["clusters_touched"] = []

        # ``seeds_unsupported`` — seeds that activated PPR but whose
        # surface never appears in any top-K candidate's text. Telegraphs
        # "your seed `Saint-Galmier` never landed in the surfaced pages"
        # so the agent knows to switch tactic (entity_lookup / chain)
        # instead of reformulating the same PPR query.
        seeds_unsupported = [
            s for s in seed_surfaces if s not in seeds_with_any_support
        ]

        # Per-doc rollup. Each row carries:
        #  * ``pages_in_topk`` + ``page_numbers`` + ``page_span``
        #    — the visible-from-PPR subset
        #  * ``total_pages_in_doc`` + ``pages_not_in_topk_sample`` —
        #    so the agent sees "PPR surfaced 2 of 11 pages this doc
        #    has; 9 more exist (e.g. p_3, p_4, p_8, ...)" and can
        #    batch-read the missing siblings without a second PPR
        #  * ``dominant_cluster`` — the cluster_id occurring most
        #    often across the doc's visible pages, plus its surface;
        #    distinguishes "Gignac bio" from "Florida Panthers bio"
        #    without reading p_1 of each
        per_page_clusters: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = {
            (r["file_id"], r.get("page_number")): r.get("clusters_touched") or []
            for r in results_with_preview
        }
        channel = self._channel
        docs_summary: List[Dict[str, Any]] = []
        for fid, ps in doc_pages_full.items():
            ps_sorted = sorted(ps, key=lambda t: t[0])
            page_numbers = [pn for pn, _ in ps_sorted]
            total = round(sum(s for _, s in ps), 6)
            total_pages = channel.doc_page_count(fid)
            not_in_topk = (
                [pn for pn in range(1, total_pages + 1) if pn not in page_numbers]
                if total_pages > 0 else []
            )
            # Cap the missing-pages list to keep token cost in check
            # on very long docs; the first 15 are the most informative
            # ("contiguous near the surfaced range" or "all of them").
            not_in_topk_short = not_in_topk[:15]
            cluster_counts: Dict[str, int] = defaultdict(int)
            cluster_surface: Dict[str, str] = {}
            for pn in page_numbers:
                for c in per_page_clusters.get((fid, pn), []):
                    cid = c.get("cluster_id")
                    if not cid:
                        continue
                    cluster_counts[cid] += 1
                    if cid not in cluster_surface:
                        cluster_surface[cid] = c.get("top_surface") or ""
            dominant = None
            if cluster_counts:
                cid_best, n_pages = max(cluster_counts.items(), key=lambda kv: kv[1])
                dominant = {
                    "cluster_id": cid_best,
                    "top_surface": cluster_surface.get(cid_best, ""),
                    "pages_anchored": n_pages,
                }
            row: Dict[str, Any] = {
                "file_id": fid,
                "pages_in_topk": len(page_numbers),
                "total_pages_in_doc": total_pages,
                "page_numbers": page_numbers,
                "page_span": [page_numbers[0], page_numbers[-1]] if page_numbers else [],
                "total_score": total,
            }
            if not_in_topk_short:
                row["pages_not_in_topk_sample"] = not_in_topk_short
            if dominant is not None:
                row["dominant_cluster"] = dominant
            docs_summary.append(row)
        docs_summary.sort(key=lambda d: d["total_score"], reverse=True)
        docs_summary = docs_summary[:5]  # token cap
        # L2: top logical clusters by PPR mass.  Surfaced as DIAGNOSTIC
        # CONTEXT only — LLMs navigate poorly via opaque cluster IDs
        # (exposing them as the primary navigation signal regresses
        # answer quality). The five entries here let the agent notice
        # when PPR drifts onto a hub cluster off-topic without being
        # expected to follow `cluster_id` back.
        cluster_scores = debug_snapshot.get("cluster_scores", {}) or {}
        top_logical = []
        for cid, mass in sorted(
            cluster_scores.items(), key=lambda kv: kv[1], reverse=True
        )[:5]:
            top_surf = self._channel.cluster_top_surfaces(cid, top_n=1)
            members = self._channel._cluster_cache.get(cid) if self._channel._cluster_cache else None
            top_logical.append({
                "cluster_id": cid,
                "top_surface": top_surf[0]["surface"] if top_surf
                               else self._channel.entity_store.hash_id_to_text.get(cid, cid),
                "mass": round(float(mass), 6),
                "members_n": len(members) if members else 1,
            })
        log_meta = {
            "mode": "ppr",
            "question": question,
            "scope": scope.as_dict(),
            "seeds": [s["surface"] for s in seeds_dbg] if isinstance(seeds_dbg, list) else [],
            "hits": len(results_with_preview),
        }
        context.add_retrieval_log(tool_name="graph_explore", tokens=0, metadata=log_meta)

        # We now DO surface compact cluster info (L2/L3) — the prior
        # comment said "cluster_scores workbench-only" but that was
        # the original design oversight: the agent's whole task IS
        # logical-entity navigation. ``top_logical_clusters`` is the
        # decompressed form, ``clusters_touched`` per page localizes
        # which logical entities live on each candidate.
        # ``unresolved`` — telegraphs "this PPR call did not anchor on
        # any specific entity from the question". Fires when every
        # activated seed surface was unsupported in the top-K AND the
        # leading logical cluster's mass is below a low floor. Signals
        # to the agent: do NOT answer-from-noise on this PPR slate;
        # either pivot to entity_analysis with a more specific
        # surface, or stop and ask for disambiguation. False when no
        # seeds at all (different failure mode — "no_seeds_skip").
        _top_cluster_mass = max(
            (float(c.get("mass") or 0.0) for c in top_logical), default=0.0
        )
        unresolved = (
            len(seed_surfaces) > 0
            and len(seeds_unsupported) == len(seed_surfaces)
            and _top_cluster_mass < 0.02
        )

        return (
            ok(
                "GraphExploreObservation",
                mode="ppr",
                question=question,
                scope=scope.as_dict(),
                seeds=[
                    {"surface": s.get("surface"), "sim": s.get("sim")}
                    for s in (seeds_dbg if isinstance(seeds_dbg, list) else [])
                ][:8],  # cap seed list — long NER seeds also bloat
                seeds_unsupported=seeds_unsupported,
                unresolved=unresolved,
                top_logical_clusters=top_logical,
                docs_summary=docs_summary,
                candidate_pages=results_with_preview,
            ),
            {"retrieved_tokens": 0, "hits": len(results_with_preview)},
        )

    def _attach_previews(
        self,
        results: List[Dict[str, Any]],
        channel_hits: Optional[List[Any]] = None,
        *,
        question: str = "",
    ) -> List[Dict[str, Any]]:
        """Add a question-conditioned ``preview`` field per candidate
        page — the single sentence on that page that maximises
        cos(q_emb, sent_emb). The agent uses it for fine-grained page
        selection before a ``read`` call; question-conditioned snippets
        flip "this candidate is about my topic" from a guess to a
        direct signal, unlike a first-N-chars preview.

        Channel artifact path: each page's sentences + their
        embeddings come from
        :meth:`GraphPPRChannel.passage_sentence_embs`, built once from
        the ingest-persisted ``passage_hash_id_to_sentences`` map and
        the sentence_store. The query embedding is encoded once per
        call and reused across every candidate page.

        Prefers ``channel_hits[i].evidence[0]["passage_hash_id"]`` for
        an exact passage match. Falls back to a one-time-built
        ``(file_id, page_number) → passage_hash`` map when evidence
        is unavailable (older trace formats / non-PPR modes).
        """
        from agentic.tools.acquisition._preview import query_snippet

        if not results:
            return results
        out: List[Dict[str, Any]] = []
        channel = self._channel
        page_store = getattr(channel, "passage_store", None)
        embed_client = getattr(channel, "embedding_client", None)

        # Encode the query once and reuse across every candidate page.
        q_emb = None
        if question and embed_client is not None:
            try:
                q_emb = embed_client.encode(question, is_query=True)
                if q_emb.ndim == 2:
                    q_emb = q_emb[0]
            except Exception:
                q_emb = None

        # Build the reverse meta map lazily — only if we end up needing
        # the fallback path (i.e. some hit lacks evidence).
        meta_map: Optional[Dict[Tuple[str, Optional[int]], str]] = None
        for i, hit_meta in enumerate(results):
            if page_store is None or not hasattr(page_store, "hash_id_to_text"):
                out.append({**hit_meta, "preview": ""})
                continue
            # Path 1: exact passage_hash_id from ChannelHit.evidence.
            passage_hash: Optional[str] = None
            if channel_hits and i < len(channel_hits):
                ev = getattr(channel_hits[i], "evidence", None) or []
                for e in ev:
                    h = e.get("passage_hash_id") if isinstance(e, dict) else None
                    if h:
                        passage_hash = h
                        break
            # Path 2: fallback via (file_id, page_number).  When a
            # page is split into multiple passages, a plain dict-comp
            # would keep the LAST passage's hash; use ``setdefault``
            # so the FIRST passage on each page wins — that's
            # typically the page heading / lead paragraph, which is
            # a better disambiguation preview than a mid-page span.
            if passage_hash is None:
                if meta_map is None:
                    meta_map = {}
                    try:
                        col_file_id = page_store.meta_column("file_id")
                        col_page_number = page_store.meta_column("page_number")
                        for h, fid, pn in zip(
                            page_store.hash_ids, col_file_id, col_page_number
                        ):
                            key_pn = int(pn) if pn is not None else None
                            meta_map.setdefault((str(fid), key_pn), h)
                    except Exception:
                        meta_map = {}
                key = (str(hit_meta["file_id"]),
                       int(hit_meta["page_number"]) if hit_meta.get("page_number") is not None else None)
                passage_hash = meta_map.get(key)

            page_text = ""
            cached_sentences = None
            if passage_hash is not None:
                page_text = page_store.hash_id_to_text.get(passage_hash, "") or ""
                # passage_sentence_embs is a one-time disk read + dict
                # translation; any failure (missing ner_results.json,
                # sentence_store schema drift) should degrade to the
                # slow-path snippet, not lose the whole observation.
                try:
                    cached_sentences = channel.passage_sentence_embs(passage_hash)
                except Exception:
                    cached_sentences = None
            try:
                preview = query_snippet(
                    page_text,
                    question,
                    embed_client,
                    max_chars=_PPR_PREVIEW_CHARS,
                    cached_query_emb=q_emb,
                    cached_sentences=cached_sentences,
                )
            except Exception:
                preview = (" ".join(page_text.split())[:_PPR_PREVIEW_CHARS]
                           if page_text else "")
            out.append({**hit_meta, "preview": preview})
        return out

    @staticmethod
    def _doc_aware_rerank(
        eligible: List[Tuple[Any, Optional[int]]],
        limit: int,
    ) -> List[Tuple[Any, Optional[int]]]:
        """Reorder scope-filtered PPR hits so pages from the strongest
        document (in the ``Σ s_i / sqrt(N+1)`` sense) come first.

        No-op when at most one doc is present or the eligible slate
        already fits in ``limit`` from a single doc; the reordering
        only matters when several pages from one or more docs compete
        for the top slots. Within a doc, the page order from PPR is
        preserved (secondary sort by original score).
        """
        if not eligible:
            return eligible
        distinct_docs = {h.file_id for h, _ in eligible}
        if len(distinct_docs) <= 1:
            return eligible
        raw = [RawHit(file_id=h.file_id, page_id=h.page_id, score=float(h.score))
               for h, _ in eligible]
        doc_hits = aggregate_per_doc(raw, top_k=len(distinct_docs))
        doc_score = {d.file_id: d.score for d in doc_hits}
        return sorted(
            eligible,
            key=lambda hp: (doc_score.get(hp[0].file_id, 0.0), float(hp[0].score)),
            reverse=True,
        )

    @staticmethod
    def _first_page_number(hit, meta_by_hash) -> Optional[int]:
        for ev in hit.evidence or []:
            h = ev.get("passage_hash_id")
            if h and h in meta_by_hash:
                _, pn = meta_by_hash[h]
                try:
                    return int(pn) if pn is not None else None
                except (TypeError, ValueError):
                    return None
        return None

    # ----------------------------------------------------------- chain mode

    def _run_chain(self, context: "AgentContext", **kwargs):
        """Query-time typed-edge beam search.

        Pipeline (one call):

        1. Seed entities = union(NER(question), gazetteer-literal-scan,
           question-embedding top-k against entity_store). Each
           ingredient catches a different miss mode (NER misses domain
           product names; gazetteer misses paraphrases; embedding
           misses purely structural anchors) so the union maximises
           recall before beam expansion narrows the search.
        2. For each path tail, take up to ``L`` entity-typed neighbors
           ranked by graph edge weight. For each ``(tail, neighbor)``
           pair, look up the sentences where both entities co-occur
           (the via-sentences index built lazily on
           ``GraphPPRChannel``). The cos-similarity between the
           question embedding and each via-sentence embedding gives a
           per-query, per-edge relation score — the sentence IS the
           predicate, so the typing happens at query time, not build
           time.
        3. Aggregate the top-M via-sentence sims into one edge score
           via log-mean-exp_β + frequency-penalty + sigmoid
           calibration (see §8.5.2 in the design doc for the math
           audit). Convert to log-edge-probability.
        4. Score paths by ``mean log p_e + κ·min log p_e + λ·seed-
           coverage − ρ·hub-degree-penalty``. Beam-prune to top-K
           between hops.
        5. Final output bundle: top paths with via-sentence snippets +
           candidate pages derived from entity-passage incidence
           weighted by path mass.
        """
        question = (kwargs.get("question") or "").strip()
        # Tier 3 chain extensions: `from_page` / `from_entities` /
        # `to_entities` let the agent override the seeding source and
        # filter the final paths.  Use cases:
        #   * `from_page="db_en_0937/p_0026"` — page_neighbors:
        #     expand from entities on that page (cross-doc bridging).
        #   * `from_entities=["Lincoln"]` + `to_entities=["Bishop"]`
        #     — sentence-bridge: only return paths that reach a
        #     `to` surface.  Combine with depth=2-3 to enumerate
        #     bridges between two known entities.
        from_page = (kwargs.get("from_page") or "").strip()
        from_entities = kwargs.get("from_entities") or []
        to_entities = kwargs.get("to_entities") or []
        if not question and not from_page and not from_entities:
            return err(
                "invalid_argument",
                "mode='chain' requires one of: `question`, `from_page`, or `from_entities`.",
                remediation=(
                    "Provide `question` (free-text seeding via NER + "
                    "gazetteer + embedding), `from_page=\"file_id/p_NNNN\"` "
                    "to expand from a known page, or `from_entities=[...]` "
                    "to seed from explicit surface forms.  `to_entities=[...]` "
                    "(optional) filters final paths to those reaching any "
                    "target — useful for sentence-bridge queries between "
                    "two known entities."
                ),
                valid_example={
                    "mode": "chain",
                    "question": "Who developed the chipset used in AMD 785G?",
                },
            ), {"error": "invalid_argument"}

        try:
            depth = int(kwargs.get("depth", _DEFAULT_CHAIN_DEPTH))
        except (TypeError, ValueError):
            return err(
                "invalid_argument",
                "`depth` must be an integer.",
                remediation=f"Pass `depth` as an integer in [1, {_MAX_CHAIN_DEPTH}] (default {_DEFAULT_CHAIN_DEPTH}).",
                valid_example={"depth": 2},
            ), {"error": "invalid_argument"}
        if not 1 <= depth <= _MAX_CHAIN_DEPTH:
            return err(
                "invalid_argument",
                f"`depth` must be 1..{_MAX_CHAIN_DEPTH}.",
                remediation=f"Set `depth` to an integer in [1, {_MAX_CHAIN_DEPTH}].",
                valid_example={"depth": 2},
            ), {"error": "invalid_argument"}

        try:
            top_k_int = int(kwargs.get("top_k", _DEFAULT_TOP_K))
        except (TypeError, ValueError):
            return err(
                "invalid_argument",
                "`top_k` must be an integer.",
                remediation="Pass `top_k` as a positive integer (default 20, max 50).",
                valid_example={"top_k": 20},
            ), {"error": "invalid_argument"}
        if top_k_int < 1:
            return err(
                "invalid_argument",
                "`top_k` must be >= 1.",
                remediation="Set `top_k` to a positive integer (default 20, max 50).",
                valid_example={"top_k": 20},
            ), {"error": "invalid_argument"}
        top_paths = min(top_k_int, _MAX_TOP_K)

        scope, scope_err = parse_scope(kwargs.get("file_ids"), None)
        if scope_err is not None:
            return err(
                "invalid_argument",
                scope_err,
                remediation="Fix `file_ids`: pass a list of ids returned by list_files, or omit the field for corpus-wide chain.",
                valid_example={"file_ids": ["<file_id>"]},
            ), {"error": "invalid_argument"}

        channel = self._channel
        if len(channel.entity_store) == 0 or len(channel.sentence_store) == 0:
            return err(
                "graph_unavailable",
                "Entity or sentence store is empty; index the corpus first.",
                remediation="chain mode needs both an entity layer and a sentence layer; the entity graph is not available in this corpus.",
            ), {"error": "graph_unavailable"}

        # Question embedding — used for seeding AND for the cached
        # sentence-similarity vector that drives edge scoring.  When
        # only `from_page` / `from_entities` are given (no question),
        # use a synthetic prompt so we still get a vector for
        # sentence ranking.
        embed_query = question or (
            "; ".join(from_entities) if from_entities else from_page
        )
        try:
            q_emb = channel.embedding_client.encode(embed_query, is_query=True)
        except Exception as exc:
            logger.exception("graph_explore[chain] embedding failed: %s", exc)
            return err(
                "embed_failed",
                f"Embedding the query failed: {exc}",
                remediation="Retry once; if it persists, switch to mode=ppr (free-text PPR) or mode=entity_analysis (surface lookup).",
            ), {"error": "embed_failed"}
        if q_emb.ndim == 2:
            q_emb = q_emb[0]

        # Seed selection — three input variants merged in priority order.
        seeds_info: List[Dict[str, Any]] = []
        if from_page:
            # Resolve "file_id/p_NNNN" → passage_hash → top entities.
            page_seeds = self._chain_seeds_from_page(from_page, top_n=8)
            seeds_info.extend(page_seeds)
        if from_entities:
            ent_seeds = self._chain_seeds_from_surfaces(
                from_entities, source_tag="explicit"
            )
            seeds_info.extend(ent_seeds)
        if not seeds_info and question:
            # Default behaviour: question-driven NER + gazetteer + embedding.
            seeds_info = self._chain_seeds(question, q_emb)
        # Dedup by hash_id while preserving first-seen order.
        _seen: set = set()
        seeds_info = [
            s for s in seeds_info
            if not (s["hash_id"] in _seen or _seen.add(s["hash_id"]))
        ]
        if not seeds_info:
            return ok(
                "GraphExploreObservation",
                mode="chain",
                question=question,
                scope=scope.as_dict(),
                depth=depth,
                seeds=[],
                paths=[],
                candidate_pages=[],
                note="No seeds resolved from NER, gazetteer, or question embedding.",
            ), {"retrieved_tokens": 0, "paths": 0}

        # Pre-compute sentence similarities once — vectorised faiss
        # search, O(N_sent) memory but no per-edge cost during beam.
        sent_hashes = channel.sentence_store.hash_ids
        sent_idx: Dict[str, int] = {h: i for i, h in enumerate(sent_hashes)}
        sent_sims = channel.sentence_store.all_similarities(q_emb)

        pair_via = channel.pair_via_sentences()

        # Beam state: each path is a dict with running log-probability
        # sums so we can compute the path score in O(1) per extension.
        initial = [
            {
                "nodes": [s["hash_id"]],
                "edges": [],
                "log_p_sum": 0.0,
                "log_p_min": 0.0,
                "score": 0.0,
            }
            for s in seeds_info
        ]
        beam = initial
        seed_hashes = {s["hash_id"] for s in seeds_info}
        # Collect every path (incl. single-seed) across hops so depth=2
        # callers still see depth=1 chains when the second hop pinches
        # out (no via-sentence support).
        all_paths: List[Dict[str, Any]] = list(initial)

        for _hop in range(depth):
            next_paths: List[Dict[str, Any]] = []
            for path in beam:
                tail = path["nodes"][-1]
                tail_vidx = channel._name_to_vidx.get(tail)
                if tail_vidx is None:
                    continue
                for nbr_hash, _edge_w in self._top_L_entity_neighbors(
                    tail_vidx, _CHAIN_TOP_L_NEIGHBORS
                ):
                    if nbr_hash in path["nodes"]:
                        continue  # no revisits — keeps paths simple
                    key = frozenset((tail, nbr_hash))
                    via_sents = pair_via.get(key)
                    if not via_sents:
                        continue
                    cand_sims: List[float] = []
                    cand_sids: List[str] = []
                    for sid in via_sents:
                        si = sent_idx.get(sid)
                        if si is None:
                            continue
                        cand_sims.append(float(sent_sims[si]))
                        cand_sids.append(sid)
                    if not cand_sims:
                        continue
                    cand_arr = np.asarray(cand_sims, dtype=np.float64)
                    if cand_arr.size > _CHAIN_TOP_M_SENTENCES:
                        order = np.argsort(cand_arr)[::-1][:_CHAIN_TOP_M_SENTENCES]
                        sims_top = cand_arr[order]
                        sids_top = [cand_sids[int(i)] for i in order]
                    else:
                        sims_top = cand_arr
                        sids_top = cand_sids
                    edge_score = self._edge_score(sims_top, n_support=int(cand_arr.size))
                    log_p_e = float(np.log(max(edge_score, 1e-9)))
                    new_edges = path["edges"] + [
                        {
                            "tail": tail,
                            "head": nbr_hash,
                            "edge_score": float(edge_score),
                            "via_sentences": sids_top,
                            # Per-sentence q-cos similarity, aligned to
                            # ``via_sentences`` order. Lets the agent
                            # see which snippet directly addresses the
                            # question vs is shared vocab — informs
                            # whether to quote the snippet or treat it
                            # as a weak bridge.
                            "via_sims": [float(s) for s in sims_top],
                            "n_support": int(cand_arr.size),
                        }
                    ]
                    new_path = {
                        "nodes": path["nodes"] + [nbr_hash],
                        "edges": new_edges,
                        "log_p_sum": path["log_p_sum"] + log_p_e,
                        "log_p_min": (
                            log_p_e
                            if not path["edges"]
                            else min(path["log_p_min"], log_p_e)
                        ),
                    }
                    new_path["score"] = self._path_score(new_path, seed_hashes)
                    next_paths.append(new_path)
            if not next_paths:
                break
            next_paths.sort(key=lambda p: p["score"], reverse=True)
            beam = next_paths[:_CHAIN_BEAM_K]
            all_paths.extend(beam)

        # De-dupe by node tuple, keep the highest-scoring instance per
        # path topology. Paths with no edges (the seed-only initial
        # state) are dropped here — they convey nothing the caller
        # didn't already pass in via ``seeds``.
        by_signature: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for p in all_paths:
            if not p["edges"]:
                continue
            sig = tuple(p["nodes"])
            cur = by_signature.get(sig)
            if cur is None or p["score"] > cur["score"]:
                by_signature[sig] = p
        final = sorted(by_signature.values(), key=lambda p: p["score"], reverse=True)[
            :top_paths
        ]

        # Tier-3: `to_entities` filter — only keep paths reaching at
        # least one CLUSTER MEMBER of any resolved target.  Cluster-
        # aware match (not exact-hash) because the user gives a
        # natural surface like "Bishop" — the resolved hash is one
        # representative; any sibling in its alias-cluster counts as
        # a valid bridge target.  This dramatically widens chain's
        # sentence-bridge usefulness without inflating false hits
        # (alias siblings are by design semantically equivalent).
        if to_entities:
            target_hashes: set = set()
            _, m2c = channel._load_clusters_cached()
            for s in self._chain_seeds_from_surfaces(to_entities, source_tag="target"):
                target_hashes.add(s["hash_id"])
                # Expand to cluster siblings.
                cid = m2c.get(s["hash_id"])
                if cid is not None:
                    members = channel._cluster_cache.get(cid) if channel._cluster_cache else None
                    if members:
                        target_hashes.update(members)
            if target_hashes:
                final = [p for p in final if any(n in target_hashes for n in p["nodes"])]
                # Also: if from seeds + to_entities directly co-occur
                # in shared sentences (depth-1 trivially via
                # pair_via_sentences), return those as "direct bridges"
                # even though no beam expansion happened.  Allows
                # sentence-bridge to surface 1-hop evidence the beam
                # would otherwise drop (paths with no edges are filtered).
                # Source-set covers all three input modes — from_entities,
                # from_page (page-derived seeds), or question seeds.
                if not final:
                    pair_via = channel.pair_via_sentences()
                    src_hashes = {s["hash_id"] for s in seeds_info}
                    bridge_pairs = []
                    for src_h in src_hashes:
                        for tgt_h in target_hashes:
                            key = frozenset((src_h, tgt_h))
                            sents = pair_via.get(key)
                            if sents:
                                bridge_pairs.append((src_h, tgt_h, sents))
                    if bridge_pairs:
                        # Synthesize 1-hop "paths" from direct co-occurrence.
                        bridge_pairs.sort(key=lambda t: len(t[2]), reverse=True)
                        synthetic = []
                        for src_h, tgt_h, sents in bridge_pairs[:top_paths]:
                            synthetic.append({
                                "nodes": [src_h, tgt_h],
                                "edges": [{
                                    "tail": src_h,
                                    "head": tgt_h,
                                    "edge_score": 1.0,
                                    "via_sentences": list(sents)[:_CHAIN_TOP_M_SENTENCES],
                                    "n_support": len(sents),
                                }],
                                "score": float(len(sents)),
                                "log_p_sum": 0.0,
                                "log_p_min": 0.0,
                            })
                        final = synthetic

        paths_out = self._render_chain_paths(final)
        candidate_pages = self._chain_candidate_pages(final, scope, limit=_MAX_TOP_K)

        log_meta = {
            "mode": "chain",
            "question": question,
            "scope": scope.as_dict(),
            "depth": depth,
            "seeds": [s["surface"] for s in seeds_info],
            "paths": len(paths_out),
            "candidate_pages": len(candidate_pages),
        }
        context.add_retrieval_log(tool_name="graph_explore", tokens=0, metadata=log_meta)

        return (
            ok(
                "GraphExploreObservation",
                mode="chain",
                question=question,
                scope=scope.as_dict(),
                depth=depth,
                seeds=[
                    {
                        "hash_id": s["hash_id"],
                        "surface": s["surface"],
                        "sim": round(s["sim"], 4),
                        "source": s["source"],
                    }
                    for s in seeds_info
                ],
                paths=paths_out,
                candidate_pages=candidate_pages,
            ),
            {
                "retrieved_tokens": 0,
                "paths": len(paths_out),
                "pages": len(candidate_pages),
            },
        )

    # ----------------------------------------------------- chain helpers

    def _chain_seeds(
        self, question: str, q_emb: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Union NER ∪ gazetteer ∪ question-embedding top-k.

        Returns at most ``_CHAIN_MAX_SEEDS`` ranked by similarity
        (literal gazetteer hits pinned at sim=1.0). Hidden vertices
        (collapse-absorbed) are skipped so we don't seed into a
        logically-gone node.

        Each ingredient is run unconditionally so seeds are a true
        union — different question shapes use different ingredients
        ("Who developed X?" → NER+embedding; "FATCA reporting" →
        gazetteer; "what's connected here?" → embedding only).
        """
        channel = self._channel
        best: Dict[str, Tuple[float, str, str]] = {}  # hash → (sim, surface, source)

        # 1) NER-driven seeds — embedding-match each tagged surface.
        try:
            ner = channel._ensure_ner()
            raw_surfaces = ner.question_ner(question)
        except Exception as exc:
            logger.warning("graph_explore[chain] question_ner failed: %s", exc)
            raw_surfaces = []
        canonical: List[str] = []
        seen_canon: set = set()
        for raw in raw_surfaces:
            c = normalize_for_hash(
                raw,
                fold_traditional=channel.linear_config.fold_traditional,
                han_fragment_max_chars=channel.linear_config.junk_max_han_chars,
            )
            if c and c not in seen_canon:
                seen_canon.add(c)
                canonical.append(c)
        if canonical:
            embs = channel.embedding_client.encode(canonical)
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)
            for vec in embs:
                top1 = channel.entity_store.topk(vec, 1)
                if not top1:
                    continue
                hid, sc = top1[0]
                if hid not in channel._name_to_vidx:
                    continue
                if channel._is_hidden(channel._name_to_vidx[hid]):
                    continue
                surface = channel.entity_store.hash_id_to_text.get(hid, "")
                cur = best.get(hid)
                if cur is None or float(sc) > cur[0]:
                    best[hid] = (float(sc), surface, "ner")

        # 2) Gazetteer literal scan — Aho-Corasick over question text.
        try:
            gaz = channel._ensure_gazetteer()
        except Exception as exc:
            logger.warning("graph_explore[chain] gazetteer build failed: %s", exc)
            gaz = None
        if gaz is not None:
            counts = find_literal_matches(question, gaz)
            for hid in counts.keys():
                if hid not in channel._name_to_vidx:
                    continue
                if channel._is_hidden(channel._name_to_vidx[hid]):
                    continue
                surface = channel.entity_store.hash_id_to_text.get(hid, "")
                cur = best.get(hid)
                # Literal matches are unambiguous → pin to 1.0; source
                # is upgraded to "gazetteer" if it beat the prior score.
                if cur is None or 1.0 > cur[0]:
                    best[hid] = (1.0, surface, "gazetteer")

        # 3) Whole-question embedding top-k.
        top = channel.entity_store.topk(q_emb, 5)
        floor = 0.50
        for hid, sc in top:
            if sc < floor:
                break
            if hid not in channel._name_to_vidx:
                continue
            if channel._is_hidden(channel._name_to_vidx[hid]):
                continue
            surface = channel.entity_store.hash_id_to_text.get(hid, "")
            cur = best.get(hid)
            if cur is None or float(sc) > cur[0]:
                best[hid] = (float(sc), surface, "question_embedding")

        seeds = [
            {"hash_id": hid, "sim": sim, "surface": surf, "source": src}
            for hid, (sim, surf, src) in best.items()
        ]
        seeds.sort(key=lambda s: s["sim"], reverse=True)
        return seeds[:_CHAIN_MAX_SEEDS]

    def _chain_seeds_from_page(
        self,
        page_ref: str,
        top_n: int = 8,
    ) -> List[Dict[str, Any]]:
        """Page-neighbors seeding: take the top entities mentioned on
        ``page_ref`` (= ``"file_id/p_NNNN"``) as chain seeds.  Used
        when the agent already knows the relevant page and wants to
        bridge OUT (cross-doc entity that also appears in another
        passage etc).
        """
        channel = self._channel
        if "/" not in page_ref:
            return []
        file_id, page_id = page_ref.split("/", 1)
        # Resolve to passage_hash via the (file_id, page_number) map.
        try:
            pn = int(page_id.replace("p_", "").lstrip("0") or "0")
        except ValueError:
            return []
        meta_to_hash = self._passage_meta_to_hash()
        passage_hash = meta_to_hash.get((file_id, pn))
        if passage_hash is None:
            return []
        channel._build_entity_passage_indexes()
        ents = (channel._passage_entities or {}).get(passage_hash, [])
        text = channel.entity_store.hash_id_to_text
        out: List[Dict[str, Any]] = []
        for ent_hash, w in ents[:top_n]:
            if ent_hash not in channel._name_to_vidx:
                continue
            out.append({
                "hash_id": ent_hash,
                "sim": float(w),
                "surface": text.get(ent_hash, ""),
                "source": "page_neighbors",
            })
        return out

    def _chain_seeds_from_surfaces(
        self,
        surfaces: List[str],
        *,
        source_tag: str,
    ) -> List[Dict[str, Any]]:
        """Resolve a list of free-text surfaces to entity hashes via
        embedding-match (same gradient-topk machinery as
        entity_lookup).  Used for explicit-entity seeding and for
        `to_entities` target resolution (sentence-bridge filter).
        """
        if not surfaces:
            return []
        channel = self._channel
        text = channel.entity_store.hash_id_to_text
        out: List[Dict[str, Any]] = []
        for surf in surfaces:
            surf = (surf or "").strip()
            if not surf:
                continue
            try:
                vec = channel.embedding_client.encode(surf)
                if vec.ndim == 2:
                    vec = vec[0]
                top = channel.entity_store.topk(vec, 1)
            except Exception:
                continue
            if not top:
                continue
            hid, score = top[0]
            if hid not in channel._name_to_vidx:
                continue
            out.append({
                "hash_id": hid,
                "sim": float(score),
                "surface": text.get(hid, surf),
                "source": source_tag,
            })
        return out

    def _top_L_entity_neighbors(
        self, vidx: int, L: int
    ) -> List[Tuple[str, float]]:
        """Up to ``L`` entity-typed neighbors of ``vidx``, by edge weight desc.

        Passage / sentence neighbors are skipped — chain navigates the
        entity layer; pages are surfaced as a derived view in the
        final candidate_pages list, not as beam-search nodes.
        """
        graph = self._channel.graph
        has_weight = "weight" in graph.es.attributes()
        out: List[Tuple[str, float]] = []
        for e in graph.incident(vidx, mode="all"):
            edge = graph.es[e]
            w = float(edge["weight"]) if has_weight else 1.0
            tgt = edge.target if edge.source == vidx else edge.source
            v = graph.vs[tgt]
            vtype = v.attributes().get("vertex_type") or self._guess_vertex_type(v["name"])
            if vtype != "entity":
                continue
            if self._channel._is_hidden(tgt):
                continue
            out.append((v["name"], w))
        out.sort(key=lambda item: item[1], reverse=True)
        return out[:L]

    @staticmethod
    def _edge_score(sims: np.ndarray, n_support: int) -> float:
        """log-mean-exp_β aggregate + frequency penalty + sigmoid.

        ``log-mean-exp_β`` interpolates between mean (β→0) and max
        (β→∞) so a single strong sentence can carry the edge but the
        score still tracks aggregate evidence — sidesteps the "max
        picks the noisiest hit" failure mode of naive top-1 pooling.

        The frequency penalty ``γ·log(1 + n_support)`` down-weights
        edges whose endpoints co-occur in many sentences — these are
        almost always hub-pair noise (e.g. an entity that mentions
        "figure" on every page) rather than a strong relation.

        Sigmoid calibration maps the raw aggregate into [0, 1] so the
        downstream log-probability path-scoring is well-conditioned.
        """
        # Compute log_mean_exp_β in float64 with the standard max-shift
        # trick so overflow can't bite when β·sim is large.
        z = _CHAIN_BETA * sims
        z_max = float(z.max())
        lme = z_max + float(np.log(np.mean(np.exp(z - z_max))))
        A = (lme / _CHAIN_BETA) - _CHAIN_GAMMA * float(np.log1p(n_support))
        # Sigmoid keeps the answer in (0, 1) so log() downstream is safe.
        return float(1.0 / (1.0 + np.exp(-_CHAIN_ALPHA * (A - _CHAIN_TAU))))

    def _path_score(
        self, path: Dict[str, Any], seed_hashes: set
    ) -> float:
        """``mean log p_e + κ·min log p_e + λ·coverage − ρ·hub-penalty``.

        - mean log p_e: average edge confidence (length-normalised).
        - min log p_e: weak-link penalty — a path with one near-zero
          edge is worse than two mediocre ones, even if the sum is
          similar.
        - coverage: fraction of distinct seeds the path touches; an
          ungrounded path (seed → random walk into a different topic)
          scores lower than one that stays in the question's
          referenced entities.
        - hub_penalty: mean log-degree of nodes in path; routes
          through high-degree hubs (which dilute mass everywhere)
          score lower than routes through specific entities.
        """
        n_edges = len(path["edges"])
        if n_edges == 0:
            return 0.0
        mean_log = path["log_p_sum"] / n_edges
        min_log = path["log_p_min"]
        seed_count = sum(1 for n in path["nodes"] if n in seed_hashes)
        coverage = seed_count / max(len(seed_hashes), 1)
        hub_pen = self._hub_penalty(path["nodes"])
        return (
            mean_log
            + _CHAIN_W_MIN * min_log
            + _CHAIN_W_COVER * coverage
            - _CHAIN_W_HUB * hub_pen
        )

    def _hub_penalty(self, node_hashes: List[str]) -> float:
        graph = self._channel.graph
        name_to_vidx = self._channel._name_to_vidx
        pen = 0.0
        n = 0
        for h in node_hashes:
            vidx = name_to_vidx.get(h)
            if vidx is None:
                continue
            pen += float(np.log1p(graph.degree(vidx)))
            n += 1
        return pen / max(n, 1)

    def _render_chain_paths(
        self, paths: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Serialise paths for the agent: nodes, edges, via-sentence snippets.

        Each via-sentence is truncated to 220 chars (single-line
        snippet) — the agent uses these as candidate answer spans, so
        a useful snippet is enough; verbatim full-text is the
        read_page tool's job.
        """
        channel = self._channel
        ent_text = channel.entity_store.hash_id_to_text
        sent_text = channel.sentence_store.hash_id_to_text
        # L4: annotate each chain node with its logical cluster so the
        # agent can see "this hop stays inside Nagano cluster, the
        # next hop bridges to Olympics cluster".  Singleton entities
        # surface their own hash as cluster_id (per the
        # member_to_cluster `.get(name, name)` fallback).
        _, m2c = channel._load_clusters_cached()
        out: List[Dict[str, Any]] = []
        for p in paths:
            nodes_out = []
            for h in p["nodes"]:
                cid = m2c.get(h, h)
                # Try to get a canonical/representative surface
                # (sometimes ≠ the literal node's surface).
                rep_surfs = channel.cluster_top_surfaces(cid, top_n=1)
                nodes_out.append({
                    "hash_id": h,
                    "surface": ent_text.get(h, ""),
                    "cluster_id": cid,
                    "cluster_top_surface": (rep_surfs[0]["surface"]
                                            if rep_surfs else ent_text.get(h, "")),
                })
            edges_out: List[Dict[str, Any]] = []
            for e in p["edges"]:
                via_snippets: List[Dict[str, Any]] = []
                # ``via_sims`` is aligned to ``via_sentences`` order
                # when set by the beam path; the synthetic 1-hop
                # bridge path omits it (sims weren't computed) and we
                # fall back to None per snippet there.
                via_sims_aligned = e.get("via_sims") or [None] * len(e["via_sentences"])
                for sid, q_sim in zip(e["via_sentences"], via_sims_aligned):
                    txt = sent_text.get(sid, "")
                    if not txt:
                        continue
                    snippet = txt if len(txt) <= 220 else txt[:217] + "..."
                    entry = {"sentence_id": sid, "text": snippet}
                    if q_sim is not None:
                        entry["q_sim"] = round(float(q_sim), 4)
                    via_snippets.append(entry)
                edges_out.append(
                    {
                        "from": e["tail"],
                        "to": e["head"],
                        "edge_score": round(float(e["edge_score"]), 4),
                        "n_support": int(e["n_support"]),
                        "via_sentences": via_snippets,
                    }
                )
            out.append(
                {
                    "nodes": nodes_out,
                    "edges": edges_out,
                    "score": round(float(p["score"]), 4),
                    "hops": len(p["edges"]),
                }
            )
        return out

    def _chain_candidate_pages(
        self,
        paths: List[Dict[str, Any]],
        scope,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Aggregate passage mass from entities visited in top paths.

        Path weight is ``1 / (1 + rank)`` so the score-sign issue (path
        scores are mean-log-prob and therefore negative) doesn't
        collapse all candidates to zero; rank is well-defined for any
        comparable score. For each entity in each path, walk the
        entity-passage edges and accumulate ``rank_weight · edge_weight
        ÷ (1 + position·0.5)`` into the incident passages. Position
        decay biases attribution toward passages anchored by the path's
        endpoints (where the answer span typically lives) rather than
        the seed entity (already known to the agent).

        Returns passages in score-desc order, filtered by ``scope``.
        """
        graph = self._channel.graph
        name_to_vidx = self._channel._name_to_vidx
        passage_meta = self._passage_meta_lookup()
        has_weight = "weight" in graph.es.attributes()
        page_score: Dict[str, float] = defaultdict(float)
        for rank, p in enumerate(paths):
            base = 1.0 / (1.0 + rank)
            for pos, node in enumerate(p["nodes"]):
                vidx = name_to_vidx.get(node)
                if vidx is None:
                    continue
                v = graph.vs[vidx]
                vtype = v.attributes().get("vertex_type") or self._guess_vertex_type(
                    v["name"]
                )
                if vtype != "entity":
                    continue
                decay = 1.0 + 0.5 * pos
                for e in graph.incident(vidx, mode="all"):
                    edge = graph.es[e]
                    w = float(edge["weight"]) if has_weight else 1.0
                    tgt = edge.target if edge.source == vidx else edge.source
                    tv = graph.vs[tgt]
                    tvtype = tv.attributes().get("vertex_type") or self._guess_vertex_type(
                        tv["name"]
                    )
                    if tvtype != "passage":
                        continue
                    page_score[tv["name"]] += base * w / decay
        out: List[Dict[str, Any]] = []
        for passage_hash, score in sorted(
            page_score.items(), key=lambda kv: kv[1], reverse=True
        ):
            meta = passage_meta.get(passage_hash)
            if meta is None:
                continue
            file_id, page_n = meta
            pn_int = _coerce_int(page_n)
            if not scope.contains(file_id, pn_int):
                continue
            out.append(
                {
                    "file_id": file_id,
                    "page_id": f"p_{pn_int:04d}" if pn_int is not None else None,
                    "page_number": pn_int,
                    "score": round(float(score), 6),
                }
            )
            if len(out) >= limit:
                break
        return out

    # ----------------------------------------------------------- entity_lookup

    def _run_entity_lookup(self, context: "AgentContext", **kwargs):
        surface = (kwargs.get("surface") or "").strip()
        if not surface:
            return err(
                "invalid_argument",
                "mode='entity_lookup' requires `surface`.",
                remediation="Pass `surface` as the entity name to look up (e.g. 'AXA' or 'AXA Hong Kong').",
                valid_example={"mode": "entity_lookup", "surface": "AXA"},
            ), {"error": "invalid_argument"}
        try:
            top_k_int = int(kwargs.get("top_k", 5))
        except (TypeError, ValueError):
            return err(
                "invalid_argument",
                "`top_k` must be an integer.",
                remediation="Pass `top_k` as a positive integer (default 5, max 10).",
                valid_example={"top_k": 5},
            ), {"error": "invalid_argument"}
        if top_k_int < 1:
            return err(
                "invalid_argument",
                "`top_k` must be >= 1.",
                remediation="Set `top_k` to a positive integer (default 5, max 10).",
                valid_example={"top_k": 5},
            ), {"error": "invalid_argument"}
        limit = min(top_k_int, 10)

        store = self._channel.entity_store
        if len(store) == 0:
            return (
                err(
                    "graph_unavailable",
                    "Entity store is empty; index the corpus first.",
                    remediation="The entity layer is not built for this corpus; entity_analysis cannot run.",
                ),
                {"error": "graph_unavailable"},
            )

        try:
            emb = self._channel.embedding_client.encode(surface)
        except Exception as exc:
            return (
                err(
                    "embed_failed",
                    f"Embedding the surface failed: {exc}",
                    remediation="Retry once; if the failure repeats, try a different surface spelling for entity_analysis.",
                ),
                {"error": "embed_failed"},
            )
        emb_arr = np.asarray(emb)
        if emb_arr.ndim == 2:
            emb_arr = emb_arr[0]
        # The disambiguator's 0.85 threshold is tuned for *adding* alias
        # edges (high precision, low recall); query-time lookup wants
        # something looser, but 0.4 surfaces too much noise (e.g. a
        # surface like "Premium Refund" matched a sim=0.4 product name).
        # 0.6 trades those off; admin config
        # `graph_explore.entity_lookup_min_sim` overrides it per corpus.
        cands = gradient_topk_candidates(
            emb_arr,
            store,
            k=limit,
            g=self._entity_lookup_gradient,
            min_sim=self._entity_lookup_min_sim,
        )
        if not cands:
            return (
                ok(
                    "GraphExploreObservation",
                    mode="entity_lookup",
                    surface=surface,
                    physical=[],
                    note=(
                        f"No entity above min similarity {self._entity_lookup_min_sim} "
                        f"— try a different surface form."
                    ),
                ),
                {"retrieved_tokens": 0, "physical_hits": 0},
            )

        cluster_idx = self._cluster_index_lookup()
        reverse_map = self._reverse_map_lookup()
        physical: List[Dict[str, Any]] = []
        seen_cluster_ids: set = set()
        for cand in cands:
            row = store.get_meta_row(cand.hash_id)
            phys_text = (row.get("text") or "").strip()
            cluster_info = cluster_idx.get(cand.hash_id)
            entry = {
                "hash_id": cand.hash_id,
                "surface": phys_text,
                "similarity": round(float(cand.score), 4),
            }
            if cluster_info is not None:
                cid = cluster_info.get("cluster_id")
                # L1: replace hash-id members with readable
                # surfaces ranked by mention weight (top-8).  Add
                # cluster_size so the agent knows whether a tiny
                # cluster (likely correct) or a giant percolation
                # blob (likely over-merged).
                top_surfs = self._channel.cluster_top_surfaces(cid, top_n=8)
                full_size = len(cluster_info.get("members") or [])
                # Inline the top pages this cluster anchors so the agent
                # gets immediate cross-doc page references in a single
                # entity_lookup call (instead of needing a follow-up
                # ``cluster_inspect`` round trip). Capped at 5 to keep
                # token cost ~150 / cluster; skipped for hub clusters
                # (>50 members) where the page list would just be the
                # corpus's most-mentioned docs and dilute the signal.
                if full_size <= 50:
                    try:
                        top_pages = self._channel.cluster_top_passages(cid, top_n=5)
                    except Exception:
                        top_pages = []
                else:
                    top_pages = []
                entry["logical_cluster"] = {
                    "cluster_id": cid,
                    "canonical": cluster_info.get("canonical"),
                    "cluster_size": full_size,
                    "top_members": top_surfs,
                    "top_pages": top_pages,
                    # Heuristic flag: clusters > 50 members likely
                    # include over-merges; agent should treat them
                    # with caution.  Concrete audit available via
                    # ``mode=cluster_inspect``.
                    "audit_recommended": full_size > 50,
                }
                seen_cluster_ids.add(cid)
            # Collapse-mode citation bridge: walk the reverse_map chain
            # to the live canonical so a hidden intermediate hop is
            # never surfaced. Overlay mode never populates reverse_map
            # so this is a zero-cost path there.
            if cand.hash_id in reverse_map:
                from ingestion.index.linear_rag.disambig import follow_reverse_map

                entry["collapsed_to"] = follow_reverse_map(cand.hash_id, reverse_map)
            physical.append(entry)

        log_meta = {"mode": "entity_lookup", "surface": surface, "physical_hits": len(physical)}
        context.add_retrieval_log(tool_name="graph_explore", tokens=0, metadata=log_meta)

        return (
            ok(
                "GraphExploreObservation",
                mode="entity_lookup",
                surface=surface,
                physical=physical,
            ),
            {"retrieved_tokens": 0, "physical_hits": len(physical)},
        )

    # ----------------------------------------------------------- shared helpers

    def _passage_meta_lookup(self) -> Dict[str, Tuple[str, Optional[int]]]:
        if self._passage_meta_by_hash is None:
            store = self._channel.passage_store
            col_file_id = store.meta_column("file_id")
            col_page_n = store.meta_column("page_number")
            self._passage_meta_by_hash = {
                h: (f, p) for h, f, p in zip(store.hash_ids, col_file_id, col_page_n)
            }
        return self._passage_meta_by_hash

    def _passage_meta_to_hash(self) -> Dict[Tuple[str, Optional[int]], str]:
        if self._passage_hash_by_meta is None:
            self._passage_hash_by_meta = {
                (file_id, _coerce_int(page_n)): h
                for h, (file_id, page_n) in self._passage_meta_lookup().items()
            }
        return self._passage_hash_by_meta

    @staticmethod
    def _guess_vertex_type(hash_id: str) -> str:
        # When `vertex_type` is missing, derive it from the hash-id
        # namespace prefix (`entity-…`, `passage-…`, `sentence-…`).
        if hash_id.startswith("entity-"):
            return "entity"
        if hash_id.startswith("passage-"):
            return "passage"
        if hash_id.startswith("sentence-"):
            return "sentence"
        return "unknown"

    # ----------------------------------------------------------- new modes

    def _run_cluster_inspect(self, context: "AgentContext", **kwargs):
        """``mode='cluster_inspect'`` — audit a logical cluster.

        Returns:
          - all member surfaces (ranked by mention weight)
          - top-K passages where the cluster appears (cross-doc)
          - co-occurring clusters (via pair_via_sentences)
          - alias edge quality audit (cos/reranker/accepted_by)

        Decisive use case: the agent suspects an over-merged cluster
        (e.g. Lincoln ↔ John Ashby) and wants to check trustworthiness
        before answering.  Also lets the agent enumerate cross-doc
        evidence pages for a single logical entity in one call.
        """
        cluster_id = (kwargs.get("cluster_id") or "").strip()
        if not cluster_id:
            return err(
                "invalid_argument",
                "`cluster_id` is required for mode='cluster_inspect'.",
                valid_example={"mode": "cluster_inspect", "cluster_id": "c_2436"},
            ), {"error": "invalid_argument"}
        top_passages = int(kwargs.get("top_passages") or 10)
        top_members = int(kwargs.get("top_members") or 20)
        top_cooccur = int(kwargs.get("top_cooccur") or 8)
        ch = self._channel
        # Validate the id exists (singleton fallback: treat unknown id
        # as the entity hash itself).
        clusters, _ = ch._load_clusters_cached()
        if cluster_id not in clusters:
            return err(
                "unknown_cluster",
                f"cluster_id {cluster_id!r} not found in clusters.json.",
                remediation=(
                    "Use entity_lookup(surface=…) to find a cluster_id, "
                    "then re-call cluster_inspect on it."
                ),
            ), {"error": "unknown_cluster"}
        members = clusters[cluster_id]
        canonical = ch.entity_store.hash_id_to_text.get(members[0], "") if members else ""
        try:
            member_surfs = ch.cluster_top_surfaces(cluster_id, top_n=top_members)
            top_pgs = ch.cluster_top_passages(cluster_id, top_n=top_passages)
            cooccur = ch.cluster_cooccurrences(cluster_id, top_n=top_cooccur)
            audit = ch.cluster_alias_audit(cluster_id)
        except Exception as exc:
            logger.exception("cluster_inspect failed: %s", exc)
            return err(
                "internal_error", f"cluster_inspect raised: {exc}",
            ), {"error": "internal_error"}

        log_meta = {
            "mode": "cluster_inspect",
            "cluster_id": cluster_id,
            "cluster_size": len(members),
        }
        context.add_retrieval_log(tool_name="graph_explore", tokens=0, metadata=log_meta)

        return (
            ok(
                "GraphExploreObservation",
                mode="cluster_inspect",
                cluster_id=cluster_id,
                canonical=canonical,
                cluster_size=len(members),
                member_surfaces=member_surfs,
                top_passages=top_pgs,
                cooccur_clusters=cooccur,
                alias_audit=audit,
            ),
            {"retrieved_tokens": 0, "members": len(member_surfs)},
        )

    def _run_list_clusters(self, context: "AgentContext", **kwargs):
        """``mode='list_clusters'`` — enumerate clusters by size /
        mention_weight, with optional surface filter.  Use cases:
          - audit which clusters dominate the corpus
          - find a cluster by partial surface match (when
            entity_lookup's embedding match misses)
        """
        top_k = int(kwargs.get("top_k") or 20)
        sort_by = (kwargs.get("sort_by") or "size").strip()
        min_size = int(kwargs.get("min_size") or 2)
        surface_filter = kwargs.get("surface_filter") or None
        if sort_by not in {"size", "mention_weight"}:
            return err(
                "invalid_argument",
                f"`sort_by` must be one of ['size', 'mention_weight'].",
                valid_example={"mode": "list_clusters", "sort_by": "size"},
            ), {"error": "invalid_argument"}
        try:
            rows = self._channel.list_top_clusters(
                top_n=top_k, sort_by=sort_by,
                min_size=min_size, surface_filter=surface_filter,
            )
        except Exception as exc:
            logger.exception("list_clusters failed: %s", exc)
            return err(
                "internal_error", f"list_clusters raised: {exc}",
            ), {"error": "internal_error"}

        log_meta = {
            "mode": "list_clusters", "sort_by": sort_by,
            "n_returned": len(rows),
        }
        context.add_retrieval_log(tool_name="graph_explore", tokens=0, metadata=log_meta)
        return (
            ok(
                "GraphExploreObservation",
                mode="list_clusters",
                sort_by=sort_by,
                surface_filter=surface_filter,
                clusters=rows,
            ),
            {"retrieved_tokens": 0, "clusters": len(rows)},
        )

    # ----------------------------------------------------------- helpers

    def _reverse_map_lookup(self) -> Dict[str, str]:
        """Lazy-load the on-disk reverse_map (collapse mode); empty in overlay."""
        if self._reverse_map_cache is None:
            self._reverse_map_cache = load_reverse_map(self._reverse_map_path)
        return self._reverse_map_cache

    def _cluster_index_lookup(self) -> Dict[str, Dict[str, Any]]:
        """Map ``entity_hash → {cluster_id, canonical, members}``.

        Built lazily because the cache file may not exist on a corpus
        that was indexed before clusters were enabled. Falls through to
        reverse_map-derived synthetic clusters when the graph was
        ingested under a collapse handler (no alias subgraph exists in
        that case so :func:`get_clusters` would return empty).
        """
        if self._cluster_index is not None:
            return self._cluster_index
        idx: Dict[str, Dict[str, Any]] = {}
        reverse_map = self._reverse_map_lookup()
        try:
            if reverse_map:
                clusters = compute_clusters_for_collapse(
                    self._channel.graph, reverse_map
                )
            else:
                _lc = getattr(self._channel, "linear_config", None)
                clusters = get_clusters(
                    self._channel.graph,
                    self._clusters_cache_path,
                    algorithm=getattr(
                        _lc, "cluster_algorithm", "connected_components"
                    ),
                    leiden_resolution=getattr(
                        _lc, "cluster_leiden_resolution", 0.05
                    ),
                    leiden_weighted=getattr(_lc, "cluster_leiden_weighted", True),
                )
        except Exception as exc:
            logger.warning("graph_explore: cluster cache load failed: %s", exc)
            clusters = []
        for c in clusters:
            for member in c.get("members", []):
                idx[member] = {
                    "cluster_id": c.get("id"),
                    "canonical": c.get("canonical"),
                    "members": c.get("members", []),
                }
        self._cluster_index = idx
        return idx


def _coerce_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
