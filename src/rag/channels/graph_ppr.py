"""Graph PPR channel — Personalized PageRank retrieval over the
LinearRAG entity-passage graph.

Pipeline (one PPR run per query):

1. NER on ``query + " " + rewrite`` → raw question surfaces.
2. ``normalize_for_hash`` each surface so it lives in the same canonical
   space as the stored entities.
3. For each canonical question entity, find the single best-matching stored
   entity by cosine similarity (LinearRAG.get_seed_entities).
4. BFS entity-score expansion: for every active entity, take its mention
   sentences, dot with the question embedding, follow the top-1 sentence to
   the next entities; cut by ``iteration_threshold`` and ``max_iterations``.
5. Passage scores: dense passage retrieval (cos sim against the question)
   normalized min-max, plus an entity-occurrence log bonus per entity, all
   weighted by ``passage_ratio`` and damped by ``passage_node_weight``.
6. ``node_weights = entity_weights + passage_weights``; PPR via igraph
   ``personalized_pagerank`` with ``damping``, ``prpack``, ``directed=False``,
   ``weights='weight'``.
7. Map passage vertex names → ``(file_id, page_id)`` by reading the
   passage store's ``file_id`` / ``page_number`` meta columns.

Logical-entity (alias-cluster) aggregation is **not** done here: the
build-time alias edges already participate in PPR propagation, so passage
scores include cross-alias mass without an extra aggregation step.
"""

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import igraph as ig
import numpy as np

from config import LinearRAGConfig, RAGConfig
from config.settings import (
    faiss_graph_dir,
    faiss_graph_entity_dir,
    faiss_graph_passage_dir,
    faiss_graph_sentence_dir,
)
from ingestion.index.linear_rag.backfill import (
    build_gazetteer_automaton,
    find_literal_matches,
)
from ingestion.index.linear_rag.disambig import (
    aggregate_by_cluster,
    compute_clusters_for_collapse,
    get_clusters,
    load_reverse_map,
)
from ingestion.index.linear_rag.ner import GLiNERAdapter
from ingestion.index.linear_rag.normalize import normalize_for_hash
from model_client import EmbeddingClient, get_cached_embedding_client
from rag.channels.base import BaseChannel, ChannelHit
from rag.preprocess import QueryContext
from storage import EmbeddingStore
from storage.embedding_store import get_or_create_store


logger = logging.getLogger(__name__)


class GraphPPRChannel(BaseChannel):
    name = "graph_ppr"

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        linear_config: Optional[LinearRAGConfig] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.config = config or RAGConfig()
        self.embedding_client = embedding_client or get_cached_embedding_client()
        self.linear_config = linear_config or LinearRAGConfig(
            embedding_client=self.embedding_client
        )
        # Runtime NER follows ingest. If this KG persisted the GLiNER knobs it
        # was built with (``ner_config.json``, written by LinearRAG flush), they
        # override the runtime config so query-time entity extraction uses the
        # SAME labels as ingest. Each dataset (insurance / Double-Bench /
        # multi-hop) therefore auto-uses its own labels with no runtime wiring.
        # Absent file → config unchanged.
        self.linear_config = self._apply_persisted_ner_config(self.linear_config)

        # Stores are pulled from the process-wide cache (storage.embedding_store
        # ::get_or_create_store) so the lifespan-built channel and any
        # per-ingest LinearRAG share the same in-memory faiss / parquet
        # handle. Avoids ~1.5 GB duplicate RSS on 8 GB hosts.
        self.passage_store = get_or_create_store(faiss_graph_passage_dir(), namespace="passage")
        self.entity_store = get_or_create_store(faiss_graph_entity_dir(), namespace="entity")
        self.sentence_store = get_or_create_store(faiss_graph_sentence_dir(), namespace="sentence")

        self._graphml_path = faiss_graph_dir() / "LinearRAG.graphml"
        self._ner_path = faiss_graph_dir() / "ner_results.json"

        self.graph: Optional[ig.Graph] = None
        if self._graphml_path.is_file():
            self.graph = ig.Graph.Read_GraphML(str(self._graphml_path))

        self._name_to_vidx: Dict[str, int] = {}
        self._passage_vidx: List[int] = []
        if self.graph is not None:
            self._name_to_vidx = {v["name"]: v.index for v in self.graph.vs}
            if "vertex_type" in self.graph.vs.attributes():
                self._passage_vidx = [
                    v.index for v in self.graph.vs if v["vertex_type"] == "passage"
                ]
            else:
                # Without an explicit `vertex_type` attribute, treat any
                # vertex whose name is in the passage store as a passage.
                passage_hashes = set(self.passage_store.hash_ids)
                self._passage_vidx = [
                    v.index for v in self.graph.vs if v["name"] in passage_hashes
                ]

        # Lazy NER + lazy mention-graph init — not all queries reach PPR.
        self._ner: Optional[GLiNERAdapter] = None
        self._entity_to_sents: Optional[Dict[str, List[str]]] = None
        self._sent_to_entities: Optional[Dict[str, List[str]]] = None
        # Reverse index ``frozenset({entity_hash_a, entity_hash_b}) →
        # [sentence_hash_id, ...]`` materialised on first call to
        # :meth:`pair_via_sentences`. Powers query-time typed-edge scoring
        # in graph_explore's ``chain`` mode: the sentence(s) where two
        # entities co-occur ARE the implicit predicate, so cos(q_emb,
        # sent_emb) gives a per-query, per-edge dynamic relation score
        # without ever invoking an LLM at build time.
        self._pair_via_sentences: Optional[Dict[frozenset, List[str]]] = None

        # ``passage_hash → [(sentence_text, sentence_embedding), ...]``
        # for the question-conditioned preview snippet that PPR returns
        # on each candidate page. Built once from the ingest-persisted
        # ``passage_hash_id_to_sentences`` map in ``ner_results.json``;
        # each sentence_text is resolved to its hash via the same hash
        # function the sentence_store used at ingest, then its
        # embedding is pulled from the store. None until first access;
        # cleared by :meth:`reload`.
        self._passage_sentence_embs: Optional[
            Dict[str, List[Tuple[str, np.ndarray]]]
        ] = None

        # ``cluster_id → number of passages that have at least one
        # entity_passage edge to any cluster member``. Built once from
        # ``_passage_entities`` + ``member_to_cluster``; powers the
        # ``pages_in_cluster`` hub-vs-specific signal that the PPR
        # observation attaches to each candidate page's
        # ``clusters_touched`` entries. Cleared by :meth:`reload`.
        self._cluster_passage_count: Optional[Dict[str, int]] = None

        # ``entity_hash → [(passage_hash, edge_weight), ...]`` — inverse
        # of ``_passage_entities``. Built once via one pass over the
        # entity_passage edge list; powers ``ReadTool.neighbour_pages``
        # ("which other pages share this page's anchor entities") in
        # O(1) per-entity lookup. Cleared by :meth:`reload`.
        self._entity_to_passages_cache: Optional[
            Dict[str, List[Tuple[str, float]]]
        ] = None

        # ``(file_id, page_number) → passage_hash`` and its inverse,
        # both derived from ``passage_store`` meta columns. Built once
        # per channel; the meta is stable for the lifetime of the
        # store (any reload swaps the underlying store and resets
        # these caches in :meth:`reload`). Lookups are O(1) for every
        # tool that needs the mapping (read annotations, PPR preview
        # passage_hash resolution, chain page-seeding).
        self._page_meta_to_hash: Optional[
            Dict[Tuple[str, Optional[int]], str]
        ] = None
        self._page_hash_to_meta: Optional[
            Dict[str, Tuple[str, Optional[int]]]
        ] = None

        # ``file_id → total page count in the corpus``. Lets the agent
        # tell "PPR surfaced 2 of 11 pages this doc has" vs "PPR
        # surfaced 2 of 2 — already complete." Cleared by reload().
        self._doc_page_count: Optional[Dict[str, int]] = None

        # ``entity_hash → [(other_entity_hash, co_occurrence_count), ...]
        # sorted by count desc``. Derived from
        # :meth:`pair_via_sentences`. Powers the per-entity bridge
        # hint on ``read.entities``: which other entities share at
        # least one sentence with this one — the implicit multi-hop
        # relation predicate in our no-relation-edge tri-graph design.
        # Cleared by reload().
        self._entity_co_occur_top: Optional[
            Dict[str, List[Tuple[str, int]]]
        ] = None

        # Lazy gazetteer Aho-Corasick automaton over entity surfaces, used
        # as a query-side seed fallback when NER misses domain product
        # names (e.g. "Heritage Protector Option"). Mirrors the
        # HippoRAG-2 query-to-node seeding idea.
        self._gazetteer_automaton = None

        # Lazy quotient graph for ``ppr_on_logical=True`` — alias CCs
        # contracted into super-nodes. Built once on first query that
        # needs it, then cached. Refreshed by :meth:`reload`.
        self._quotient_graph: Optional[ig.Graph] = None
        self._quotient_member_to_super: Dict[str, int] = {}
        self._quotient_super_passage_vidx: List[int] = []
        self._quotient_name_to_super: Optional[Dict[str, int]] = None

        # Shared cluster cache for ``ppr_seed_cluster_spread`` and
        # ``ppr_on_logical``. Loading clusters.json once per channel
        # saves ~tens of ms per query × 1950q.
        self._cluster_cache: Optional[Dict[str, List[str]]] = None
        self._member_to_cluster_cache: Optional[Dict[str, str]] = None

        # Pre-built indexes for agent-facing helpers (lazy on first use):
        #   - ``_entity_passage_degree[entity_hash]`` → cumulative
        #     entity_passage edge weight from that entity. Surrogate
        #     for "how often does this entity appear in the corpus".
        #   - ``_passage_entities[passage_hash]`` → list of
        #     (entity_hash, edge_weight), sorted desc. Used to surface
        #     "which entities/clusters dominate this page" per
        #     candidate page returned by PPR (provenance).
        # Building both once costs O(E_entity_passage); in-memory dicts
        # then serve agent helpers in O(1).
        self._entity_passage_degree: Optional[Dict[str, float]] = None
        self._passage_entities: Optional[Dict[str, List[Tuple[str, float]]]] = None

        # Debug snapshot — populated each retrieve() call. Inspect with
        # ``channel.last_debug`` after a query for tracing.
        self.last_debug: Dict[str, object] = {}

        # Serializes any retrieve / retrieve_subgraph call that mutates
        # ``last_debug`` so two concurrent agents (or the RAG pipeline +
        # an agent) sharing the same channel can't interleave a tool's
        # post-call ``self.last_debug`` read with another query's write.
        # PPR is faiss + igraph + GLiNER heavy; serializing has no
        # measurable throughput cost on a single-process dev box.
        import threading

        self._call_lock = threading.RLock()

    # ----------------------------------------------------------- public API

    def reload(self, linear_config: Optional[LinearRAGConfig] = None) -> None:
        """Re-mmap faiss + re-read the GraphML so a fresh ingest is visible.

        The whole steady-state ``__init__`` is rebuilt:

        * three :class:`EmbeddingStore` handles are reconstructed —
          faiss / parquet are not auto-reloading caches, so without this
          a channel built before any ingest stays at ``len() == 0``
          forever;
        * the igraph ``Graph`` is re-read from disk if the GraphML now
          exists (the original ``__init__`` left ``self.graph = None``
          on first boot when the file did not yet exist);
        * vertex-name / passage-vertex indexes are recomputed against
          the new graph;
        * lazy NER / mention-map / gazetteer caches are dropped so the
          next query rebuilds from the current entity surfaces.

        ``linear_config`` (optional): swap in a freshly-materialised
        ``LinearRAGConfig`` so admin-rotated NER knobs (``gliner_labels``
        / ``gliner_threshold`` / ``gliner_model_id`` / literal-backfill
        params) take effect on the query path without a process
        restart. Without this, the channel kept the lifespan-time
        snapshot and admin changes only landed at next ingest.

        The shared GLiNER weights stay pinned by ``shared_gliner``'s
        process-level cache; we drop the local adapter so the next
        ``_ensure_ner()`` rebuilds it against the new linear_config.

        Held under ``_call_lock`` for the swap so a concurrent retrieve
        cannot read a half-rebuilt state.
        """
        with self._call_lock:
            if linear_config is not None:
                self.linear_config = linear_config
            # Same shared-cache lookup as ``__init__``. The cache
            # returns the *same* object as before, so reassignment is
            # idempotent. To force the cached store to pick up
            # out-of-band on-disk changes (admin restored a backup,
            # operator hand-edited faiss/), explicitly call
            # ``reload_from_disk()`` on each store — that drops the
            # in-memory state under the per-store lock and re-runs
            # ``_load`` against the current artifacts. This restores
            # the original ``/admin/refresh-indexes`` contract that was
            # silently broken when the cache started returning the same
            # object on every ``reload()`` call.
            self.passage_store = get_or_create_store(
                faiss_graph_passage_dir(), namespace="passage"
            )
            self.passage_store.reload_from_disk()
            self.entity_store = get_or_create_store(
                faiss_graph_entity_dir(), namespace="entity"
            )
            self.entity_store.reload_from_disk()
            self.sentence_store = get_or_create_store(
                faiss_graph_sentence_dir(), namespace="sentence"
            )
            self.sentence_store.reload_from_disk()

            if self._graphml_path.is_file():
                self.graph = ig.Graph.Read_GraphML(str(self._graphml_path))
            else:
                self.graph = None

            self._name_to_vidx = {}
            self._passage_vidx = []
            if self.graph is not None:
                self._name_to_vidx = {v["name"]: v.index for v in self.graph.vs}
                if "vertex_type" in self.graph.vs.attributes():
                    self._passage_vidx = [
                        v.index for v in self.graph.vs if v["vertex_type"] == "passage"
                    ]
                else:
                    passage_hashes = set(self.passage_store.hash_ids)
                    self._passage_vidx = [
                        v.index for v in self.graph.vs if v["name"] in passage_hashes
                    ]

            self._entity_to_sents = None
            self._sent_to_entities = None
            self._pair_via_sentences = None
            self._passage_sentence_embs = None
            self._cluster_passage_count = None
            self._entity_to_passages_cache = None
            self._page_meta_to_hash = None
            self._page_hash_to_meta = None
            self._doc_page_count = None
            self._entity_co_occur_top = None
            self._gazetteer_automaton = None
            self._quotient_graph = None
            self._quotient_member_to_super = {}
            self._quotient_super_passage_vidx = []
            self._quotient_name_to_super = None
            self._cluster_cache = None
            self._member_to_cluster_cache = None
            self._entity_passage_degree = None
            self._passage_entities = None
            # Hub-suppression caches (keyed by edge count) must drop on reload:
            # a swapped-in graph can share an edge count with the old one, so
            # ecount alone would otherwise hand back a stale df / damped-weight
            # vector for the wrong graph.
            self._entity_df_cache = None
            self._hub_damped_cache = None
            # Drop the cached NER adapter too. The shared GLiNER model
            # itself stays pinned by ``shared_gliner``'s lru_cache, but
            # this adapter holds a frozen reference to the old
            # labels / threshold from before the admin rotated config;
            # forcing a re-construct on the next ``_ensure_ner()`` call
            # picks up the current ``self.linear_config`` values.
            self._ner = None

    def retrieve(self, ctx: QueryContext) -> List[ChannelHit]:
        # Hold ``_call_lock`` for the whole body so a downstream caller
        # that reads ``self.last_debug`` after our return (the agent's
        # graph tools do) cannot observe another concurrent
        # query's debug payload. See :meth:`retrieve_with_debug` for the
        # atomic (hits, debug) variant — agents should prefer it; the
        # 4-channel RAG pipeline ignores last_debug, so plain
        # ``retrieve()`` is fine for that path.
        with self._call_lock:
            self.last_debug = {}
            if self.graph is None or len(self.passage_store) == 0:
                self.last_debug["mode"] = "no_graph"
                return []

            question = (ctx.query + " " + (ctx.rewrite or "")).strip()
            seeds = self._seed_entities(question, enable_fallback=ctx.enable_ppr_seed_fallback)

            self.last_debug["seeds"] = [
                {
                    "hash_id": hid,
                    "surface": self.entity_store.hash_id_to_text.get(hid, ""),
                    "sim": round(sim, 4),
                }
                for hid, _, sim in seeds
            ]

            # No seeds → return empty. LinearRAG (single-path) falls back to
            # plain dense retrieval here, but in our 4-channel RRF setup that
            # would just duplicate the semantic channel's ranking and bias the
            # fusion. Letting the graph channel sit out keeps RRF's
            # independence assumption intact.
            if not seeds:
                self.last_debug["mode"] = "no_seeds_skip"
                return []

            self.last_debug["mode"] = "ppr"
            question_emb = self.embedding_client.encode(question, is_query=True)
            entity_weights, actived = self._calculate_entity_scores(question_emb, seeds)
            passage_weights = self._calculate_passage_scores(
                question, question_emb, actived
            )
            node_weights = entity_weights + passage_weights
            self.last_debug["actived_entities"] = len(actived)

            ranked, scores_arr = self._run_ppr_full(node_weights)
            # Cluster_scores is a post-hoc view of the PPR mass — never
            # used to re-rank passages, only surfaced for debug / agent
            # tooling. Empty dict when PPR returned zero mass.
            self.last_debug["cluster_scores"] = self._cluster_scores(scores_arr)
            return self._materialize_hits(ranked, ctx.file_ids)

    def retrieve_with_debug(
        self, ctx: QueryContext
    ) -> Tuple[List[ChannelHit], Dict[str, Any]]:
        """Atomic ``retrieve()`` + snapshot of ``last_debug``.

        Use this from any consumer that reads ``self.last_debug`` after
        the call (currently: the agent's ``graph_explore`` tool's PPR
        mode). Holds the same RLock as :meth:`retrieve` for the call
        AND the snapshot so a concurrent query can't overwrite the
        debug payload between them.
        """
        with self._call_lock:
            hits = self.retrieve(ctx)
            return hits, dict(self.last_debug)

    def _is_hidden(self, vidx: int) -> bool:
        """Return True if vertex ``vidx`` was absorbed into a canonical.

        Hidden vertices are kept in the graph so the reverse_map and
        ``follow_reverse_map`` chains remain resolvable, but they must
        not be seeds for PPR (their incident alias / entity_passage
        edges have been redirected onto the canonical and the residual
        edges would route mass back to a logically-gone node).
        """
        if self.graph is None:
            return False
        if "hidden" not in self.graph.vs.attributes():
            return False
        return bool(self.graph.vs[vidx]["hidden"])

    # ----------------------------------------------------------- seeding

    def _apply_persisted_ner_config(self, cfg):
        """Override ``cfg``'s GLiNER knobs with those ingest persisted in
        ``faiss_graph_dir()/ner_config.json`` so query-time NER == ingest NER.
        Returns ``cfg`` unchanged if the file is absent/unreadable."""
        import json as _json
        from dataclasses import replace as _replace

        path = faiss_graph_dir() / "ner_config.json"
        if not path.is_file():
            return cfg
        try:
            nc = _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cfg
        fields = {
            "gliner_model_id", "gliner_labels", "gliner_noise_labels",
            "gliner_threshold", "gliner_batch_size", "ner_max_span_chars",
            "gliner_calibration_enabled", "gliner_temperature",
            "gliner_label_thresholds", "gliner_stopword_languages",
            "gliner_stopword_confidence_floor",
        }
        over = {k: v for k, v in nc.items() if k in fields}
        return _replace(cfg, **over) if over else cfg

    def _ensure_ner(self) -> GLiNERAdapter:
        if self._ner is None:
            self._ner = GLiNERAdapter(
                model_id=self.linear_config.gliner_model_id,
                labels=self.linear_config.gliner_labels,
                threshold=self.linear_config.gliner_threshold,
                batch_size=self.linear_config.gliner_batch_size,
                max_span_chars=self.linear_config.ner_max_span_chars,
                noise_labels=self.linear_config.gliner_noise_labels,
                calibration_enabled=self.linear_config.gliner_calibration_enabled,
                temperature=self.linear_config.gliner_temperature,
                label_thresholds=self.linear_config.gliner_label_thresholds,
            )
        return self._ner

    def _seed_entities(
        self,
        question: str,
        *,
        enable_fallback: bool = False,
    ) -> List[Tuple[str, int, float]]:
        """Return ``[(hash_id, vertex_idx, sim), ...]`` — at most one per
        question entity, deduped by hash_id (best score wins).

        When ``enable_fallback`` is True, also runs two query-side
        recall boosters in order if the NER path produced no seeds:

        1. **Gazetteer literal scan.** Aho-Corasick word-boundary match
           of the question text against every known entity surface.
           Catches "Heritage Protector Option" or "FATCA" that
           contextual NER fails to tag in question form.
        2. **Whole-question embedding ➜ entity-store top-K.** Mirrors
           HippoRAG-2's Query-to-Node fallback (+12.5 Recall@5 vs NER
           seeding in their paper); the cost is one extra embedding
           call per query.

        Only enabled by the graph_explore agent tool (see
        :class:`QueryContext.enable_ppr_seed_fallback`). The 4-channel
        RAG pipeline keeps this off so the graph channel doesn't shadow
        the dense semantic channel under RRF fusion.
        """
        # GLiNER is a hard dependency by project policy ("zero fallback,
        # NER as critical as the embedding API"). If the model fails to
        # load — usually CUDA missing or HF cache corrupt — surface the
        # error rather than silently downgrading PPR to dense-only mode:
        # a silent downgrade hides the misconfiguration while quietly
        # degrading graph-channel answer quality.
        ner = self._ensure_ner()

        canonical_seen: List[str] = []
        seen_set: set = set()
        raw_surfaces = ner.question_ner(question)
        for raw in raw_surfaces:
            can = normalize_for_hash(
                raw,
                fold_traditional=self.linear_config.fold_traditional,
                han_fragment_max_chars=self.linear_config.junk_max_han_chars,
            )
            if can and can not in seen_set:
                seen_set.add(can)
                canonical_seen.append(can)

        best: Dict[str, Tuple[int, float]] = {}

        if canonical_seen and len(self.entity_store) > 0:
            embs = self.embedding_client.encode(canonical_seen)
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)
            for vec in embs:
                top1 = self.entity_store.topk(vec, 1)
                if not top1:
                    continue
                hid, score = top1[0]
                if hid not in self._name_to_vidx:
                    continue
                # Skip vertices the collapse handler absorbed into a
                # canonical — seeding a hidden surface would re-introduce
                # the very provenance ambiguity collapse mode was meant
                # to eliminate, and PPR mass would flow through edges
                # that no longer logically exist.
                if self._is_hidden(self._name_to_vidx[hid]):
                    continue
                if hid not in best or score > best[hid][1]:
                    best[hid] = (self._name_to_vidx[hid], float(score))

        # Fallback path — fires when caller asked AND the NER signal
        # is weak. Treating only zero seeds as "weak" misses the
        # partial-NER case: one or two low-confidence seeds also benefit
        # from the gazetteer + question-embedding rescue. The gate
        # ``< 2 seeds OR max(sim) < 0.7`` catches those without
        # over-triggering on questions with strong full-NER coverage.
        if enable_fallback and len(self.entity_store) > 0:
            max_sim = max((s for _, s in best.values()), default=0.0)
            if len(best) < 2 or max_sim < 0.7:
                supplement = self._fallback_seeds(question)
                for hid, (vidx, sim) in supplement.items():
                    if hid not in best or sim > best[hid][1]:
                        best[hid] = (vidx, sim)

        seeds = [(hid, vidx, sim) for hid, (vidx, sim) in best.items()]

        # Option A: spread reset mass to alias-cluster members. The
        # logical entity is the alias-connected component, termed a
        # "cluster". Seeding only the matched
        # physical surface gives PPR a sparse start; spreading to
        # all cluster members better approximates "I want to walk
        # from THIS LOGICAL ENTITY" without changing the underlying
        # physical graph (which keeps full provenance + repair
        # properties — alias edges remain deletable).
        # Controlled by ``RAGConfig.ppr_seed_cluster_spread``;
        # default True since this is the design intent of the
        # three-layer (physical / cluster / logical) abstraction.
        #
        # **Skip when ``ppr_on_logical=True``**: in logical PPR each
        # cluster is already a single super-node, so projecting
        # already-spread seeds would inflate reset mass by cluster
        # size. Only the direct NER seeds are projected onto
        # super-nodes in the logical path.
        if getattr(self.config, "ppr_seed_cluster_spread", True) and not getattr(
            self.config, "ppr_on_logical", False
        ):
            seeds = self._spread_seeds_to_clusters(seeds)

        return seeds

    # Cap on cluster members spread as additional seeds, per cluster.
    # A handful of giant connected components (a single hub cluster can
    # absorb >50% of all entities on real corpora) would otherwise dump
    # tens of thousands of low-info seeds into PPR personalization and
    # explode _calculate_passage_scores (which substring-counts every
    # active entity against every passage). 50 covers the
    # alias-precision-validated cases; bigger clusters mostly carry
    # noise from over-merging and should not propagate full mass.
    _CLUSTER_SPREAD_MAX_MEMBERS = 50

    # ----------------------------------------------------------- agent helpers

    def _build_entity_passage_indexes(self) -> None:
        """Build ``_entity_passage_degree`` + ``_passage_entities`` in one
        graph-edge scan.  Used by agent-facing helpers (cluster top
        surface ranking, page-level provenance, cluster_inspect).

        Cost: single O(E) walk over the entity_passage edges.
        Dictionaries then serve agent calls in O(1).
        """
        if self._entity_passage_degree is not None:
            return
        if self.graph is None:
            self._entity_passage_degree = {}
            self._passage_entities = {}
            return
        if "edge_type" not in self.graph.es.attributes():
            self._entity_passage_degree = {}
            self._passage_entities = {}
            return
        deg: Dict[str, float] = {}
        pe: Dict[str, List[Tuple[str, float]]] = {}
        is_entity = "vertex_type" in self.graph.vs.attributes()
        for e in self.graph.es:
            if e["edge_type"] != "entity_passage":
                continue
            u_name = self.graph.vs[e.source]["name"]
            v_name = self.graph.vs[e.target]["name"]
            # Identify which endpoint is entity vs passage.
            if is_entity:
                u_type = self.graph.vs[e.source]["vertex_type"]
                v_type = self.graph.vs[e.target]["vertex_type"]
                if u_type == "entity" and v_type == "passage":
                    ent_name, pass_name = u_name, v_name
                elif u_type == "passage" and v_type == "entity":
                    ent_name, pass_name = v_name, u_name
                else:
                    continue  # malformed; skip
            else:
                # Fallback for graphs built without ``vertex_type``:
                # infer from the ``entity-`` / ``passage-`` name prefix.
                if u_name.startswith("entity-") and v_name.startswith("passage-"):
                    ent_name, pass_name = u_name, v_name
                elif v_name.startswith("entity-") and u_name.startswith("passage-"):
                    ent_name, pass_name = v_name, u_name
                else:
                    continue
            w = float(e.attributes().get("weight") or 0.0)
            deg[ent_name] = deg.get(ent_name, 0.0) + w
            pe.setdefault(pass_name, []).append((ent_name, w))
        # Sort each passage's entity list desc by weight.
        for k in pe:
            pe[k].sort(key=lambda t: t[1], reverse=True)
        self._entity_passage_degree = deg
        self._passage_entities = pe

    def cluster_top_surfaces(
        self, cluster_id: str, top_n: int = 8
    ) -> List[Dict[str, Any]]:
        """Return ``[{surface, hash_id, mention_weight}, ...]`` for a
        cluster's members, ranked by aggregate entity_passage weight.
        Used by entity_lookup / PPR / cluster_inspect to make the
        logical entity human-readable for the agent.
        """
        self._build_entity_passage_indexes()
        clusters, _ = self._load_clusters_cached()
        members = clusters.get(cluster_id)
        if members is None:
            # Singleton: cluster_id IS the hash. Wrap as a one-element list.
            members = [cluster_id]
        text = self.entity_store.hash_id_to_text
        deg = self._entity_passage_degree or {}
        scored = []
        for hid in members:
            if hid not in self._name_to_vidx:
                continue
            scored.append({
                "surface": text.get(hid, ""),
                "hash_id": hid,
                "mention_weight": round(float(deg.get(hid, 0.0)), 4),
            })
        scored.sort(key=lambda d: d["mention_weight"], reverse=True)
        return scored[:top_n]

    def passage_top_clusters(
        self, passage_hash: str, top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """Return the top-N logical clusters touching this passage,
        ranked by entity_passage edge weight.  Each entry:
        ``{cluster_id, top_surface, weight}`` (cluster_id = entity_hash
        for singletons).  Drives PPR per-page provenance.
        """
        self._build_entity_passage_indexes()
        _, m2c = self._load_clusters_cached()
        text = self.entity_store.hash_id_to_text
        ents = (self._passage_entities or {}).get(passage_hash, [])
        # Aggregate weight per cluster.
        per_cluster: Dict[str, float] = {}
        top_surface_per_cluster: Dict[str, Tuple[str, float]] = {}
        for ent_hash, w in ents:
            cid = m2c.get(ent_hash, ent_hash)
            per_cluster[cid] = per_cluster.get(cid, 0.0) + w
            # Track the entity with the highest single-edge weight per cluster
            # — its surface is most representative of the cluster's
            # appearance on this page.
            prev = top_surface_per_cluster.get(cid)
            if prev is None or w > prev[1]:
                top_surface_per_cluster[cid] = (text.get(ent_hash, ""), w)
        ranked = sorted(per_cluster.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [
            {
                "cluster_id": cid,
                "top_surface": top_surface_per_cluster[cid][0],
                "weight": round(float(w), 4),
            }
            for cid, w in ranked
        ]

    def page_meta_to_hash(self) -> Dict[Tuple[str, Optional[int]], str]:
        """``(file_id, page_number) → passage_hash``, lazily built once.

        ReadTool annotations + PPR preview resolution both need this
        reverse map; building it per call (15 k iterations × N pages)
        was a per-read hot loop. Cached on the channel since
        passage_store meta is immutable until ``reload()``.
        """
        if self._page_meta_to_hash is None:
            with self._call_lock:
                if self._page_meta_to_hash is None:
                    col_fid = self.passage_store.meta_column("file_id")
                    col_pn = self.passage_store.meta_column("page_number")
                    m: Dict[Tuple[str, Optional[int]], str] = {}
                    for h, fid, pn in zip(self.passage_store.hash_ids, col_fid, col_pn):
                        key_pn = int(pn) if pn is not None else None
                        m.setdefault((str(fid), key_pn), h)
                    self._page_meta_to_hash = m
        return self._page_meta_to_hash

    def page_hash_to_meta(self) -> Dict[str, Tuple[str, Optional[int]]]:
        """Inverse of :meth:`page_meta_to_hash`. Same lazy contract."""
        if self._page_hash_to_meta is None:
            with self._call_lock:
                if self._page_hash_to_meta is None:
                    col_fid = self.passage_store.meta_column("file_id")
                    col_pn = self.passage_store.meta_column("page_number")
                    self._page_hash_to_meta = {
                        h: (str(fid), int(pn) if pn is not None else None)
                        for h, fid, pn in zip(
                            self.passage_store.hash_ids, col_fid, col_pn
                        )
                    }
        return self._page_hash_to_meta

    def entity_to_passages(self) -> Dict[str, List[Tuple[str, float]]]:
        """``entity_hash → [(passage_hash, edge_weight), ...]`` — inverse
        of ``_passage_entities``. One pass over the entity_passage edge
        list; cached on the channel since edges are immutable until
        ``reload()``. Powers ``ReadTool.neighbour_pages``.
        """
        if self._entity_to_passages_cache is None:
            with self._call_lock:
                if self._entity_to_passages_cache is None:
                    self._build_entity_passage_indexes()
                    inv: Dict[str, List[Tuple[str, float]]] = {}
                    for ph_other, ent_list in (self._passage_entities or {}).items():
                        for ent_h, w_e in ent_list:
                            inv.setdefault(ent_h, []).append(
                                (ph_other, float(w_e))
                            )
                    self._entity_to_passages_cache = inv
        return self._entity_to_passages_cache

    def doc_page_count(self, file_id: str) -> int:
        """Total pages in ``file_id`` across the corpus. Lets PPR
        ``docs_summary`` rows convey "this doc has N pages and you
        only saw K of them" — the partial-coverage signal that the
        agent needs to decide whether to read sibling pages.
        """
        if self._doc_page_count is None:
            with self._call_lock:
                if self._doc_page_count is None:
                    counts: Dict[str, int] = defaultdict(int)
                    col_fid = self.passage_store.meta_column("file_id")
                    for fid in col_fid:
                        counts[str(fid)] += 1
                    self._doc_page_count = dict(counts)
        return self._doc_page_count.get(file_id, 0)

    def entity_top_co_occurring(
        self, entity_hash: str, top_n: int = 2
    ) -> List[Tuple[str, int]]:
        """Top entities that share at least one sentence with this one,
        ranked by sentence-co-occurrence count.

        In our tri-graph (entity ↔ passage ↔ sentence) there are NO
        relation edges between entities — the predicate is the
        natural-language sentence in which two entities co-occur.
        This method projects ``pair_via_sentences`` into a per-entity
        view: the agent reading a page can immediately see which
        other entities the page's anchors are "documented near", and
        decide whether to ``chain`` from them or jump to their pages
        via entity_analysis. The multi-hop bridge surface, without
        the agent having to call chain mode just to discover the
        partner list exists.
        """
        if self._entity_co_occur_top is None:
            with self._call_lock:
                if self._entity_co_occur_top is None:
                    pair_idx = self.pair_via_sentences()
                    co_occur: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
                    for pair, sents in pair_idx.items():
                        pair_list = list(pair)
                        if len(pair_list) < 2:
                            continue
                        a, b = pair_list[0], pair_list[1]
                        n = len(sents)
                        co_occur[a].append((b, n))
                        co_occur[b].append((a, n))
                    for k in co_occur:
                        co_occur[k].sort(key=lambda t: t[1], reverse=True)
                    self._entity_co_occur_top = dict(co_occur)
        return self._entity_co_occur_top.get(entity_hash, [])[:top_n]

    def cooccurrence_neighbors(
        self,
        tail_hash: str,
        sent_sims: np.ndarray,
        sent_idx: Dict[str, int],
        *,
        top_s: int = 32,
        top_l: int = 20,
        max_via: int = 8,
    ) -> List[Dict[str, Any]]:
        """Sentence-first co-occurrence neighbors of one tail entity.

        The chain beam's *only* source of entity→entity hops. Walking the
        graph's entity-incident edges yields alias edges (variants), not
        relations; this instead projects the tail's mention sentences —
        ranked by query relevance — onto the entities they co-mention. The
        natural-language sentence IS the predicate, so ranking sentences by
        ``cos(query, sentence)`` types the hop at query time.

        Sentence-first, not co-occurrence-count-first: a bridge entity that
        shares a single sentence with the tail is reachable as long as that
        sentence ranks high for the query. A count prefilter would drop
        exactly those rare bridges, which are the multi-hop answer path.

        Returns up to ``top_l`` neighbors, each folded to its alias cluster
        (surface variants → one logical hop; a partner in the tail's own
        cluster is dropped — an alias is not a relation). Each entry::

            {hash_id, cluster_id, max_cos, mean_cos, support, via_sids}

        ranked by ``(max_cos, mean_cos, support)`` desc. ``support`` (number
        of distinct co-occurrence sentences in the top-S window) is evidence
        metadata only, never a score penalty — repeated parent/spouse/office
        sentences are the *good* bridge in multi-hop, not hub noise.

        Cost per tail-cluster: O(Σ |sents(member)| over the cluster members)
        to gather + argpartition the top-S (not a full sort) + O(Σ entities in
        the top-S sentences). A giant alias cluster gives a one-time
        O(cluster-incidence) spike on first expansion; ``top_s`` bounds the
        downstream work. ``sent_sims`` is computed once per query by the caller
        and shared across every tail in the beam.
        """
        ent_to_sents, sent_to_ents = self._mention_maps()
        clusters, member_to_cluster = self._load_clusters_cached()
        tail_cluster = member_to_cluster.get(tail_hash, tail_hash)
        # Tail-side alias folding: a hop may start from any surface variant of
        # the tail's logical entity, so pool every cluster member's mention
        # sentences. This also makes the result a function of the cluster, so
        # the beam can memoize neighbors per cluster (not per surface).
        members = clusters.get(tail_cluster, [tail_hash])
        sents: List[str] = []
        seen_s: set = set()
        for m in members:
            for s in ent_to_sents.get(m, ()):
                if s not in seen_s:
                    seen_s.add(s)
                    sents.append(s)
        if not sents:
            return []

        # Rank the tail's mention sentences by query cosine; keep top-S.
        pairs = [(s, float(sent_sims[sent_idx[s]])) for s in sents if s in sent_idx]
        if not pairs:
            return []
        if len(pairs) > top_s:
            sims_arr = np.fromiter(
                (p[1] for p in pairs), dtype=np.float64, count=len(pairs)
            )
            keep = np.argpartition(sims_arr, -top_s)[-top_s:]
            pairs = [pairs[int(i)] for i in keep]

        # Aggregate co-mentioned entities → cluster-folded neighbors. ``via``
        # is a {sentence → sim} map so the same sentence mentioning two
        # cluster members counts once toward support.
        agg: Dict[str, Dict[str, Any]] = {}
        for sent_hash, sim in pairs:
            for e2 in sent_to_ents.get(sent_hash, ()):
                if e2 == tail_hash:
                    continue
                vidx = self._name_to_vidx.get(e2)
                if vidx is None or self._is_hidden(vidx):
                    continue
                c2 = member_to_cluster.get(e2, e2)
                if c2 == tail_cluster:
                    continue  # alias variant of the tail, not a relational hop
                slot = agg.get(c2)
                if slot is None:
                    agg[c2] = {"rep": e2, "rep_sim": sim, "via": {sent_hash: sim}}
                    continue
                if sim > slot["rep_sim"]:
                    slot["rep"], slot["rep_sim"] = e2, sim
                prev = slot["via"].get(sent_hash)
                if prev is None or sim > prev:
                    slot["via"][sent_hash] = sim

        neighbors: List[Dict[str, Any]] = []
        for c2, slot in agg.items():
            via_items = sorted(slot["via"].items(), key=lambda t: t[1], reverse=True)
            sims = [s for _, s in via_items]
            support = len(sims)
            neighbors.append({
                "hash_id": slot["rep"],
                "cluster_id": c2,
                "max_cos": sims[0],
                "mean_cos": sum(sims) / support,
                "support": support,
                "via_sids": [h for h, _ in via_items[:max_via]],
            })
        neighbors.sort(
            key=lambda n: (n["max_cos"], n["mean_cos"], n["support"]), reverse=True
        )
        return neighbors[:top_l]

    def cluster_passage_count(self, cluster_id: str) -> int:
        """Number of passages that have at least one entity_passage edge to
        any member of ``cluster_id``. Cheap O(1) lookup after a one-time
        O(E_entity_passage) build in :meth:`_build_cluster_passage_count`.

        Used by the PPR observation to annotate each ``clusters_touched``
        entry with corpus breadth — small (≤5) means a tight specific
        entity worth following; large (≥100) means a hub cluster (e.g.
        a generic organization name that mentions every page) the agent
        should down-trust.
        """
        if self._cluster_passage_count is None:
            with self._call_lock:
                if self._cluster_passage_count is None:
                    self._build_cluster_passage_count()
        return self._cluster_passage_count.get(cluster_id, 0)

    def _build_cluster_passage_count(self) -> None:
        self._build_entity_passage_indexes()
        _, m2c = self._load_clusters_cached()
        cluster_pages: Dict[str, set] = defaultdict(set)
        pe = self._passage_entities or {}
        for pass_hash, ent_list in pe.items():
            seen_clusters: set = set()
            for ent_hash, _w in ent_list:
                cid = m2c.get(ent_hash, ent_hash)
                if cid in seen_clusters:
                    continue
                seen_clusters.add(cid)
                cluster_pages[cid].add(pass_hash)
        self._cluster_passage_count = {c: len(s) for c, s in cluster_pages.items()}

    def cluster_top_passages(
        self, cluster_id: str, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Return the top passages whose entity_passage edges sum
        highest from any member of the cluster.  Each entry:
        ``{passage_hash, file_id, page_id, page_number, weight,
        member_surfaces}``.
        """
        self._build_entity_passage_indexes()
        clusters, _ = self._load_clusters_cached()
        members = set(clusters.get(cluster_id) or [cluster_id])
        deg_per_passage: Dict[str, float] = {}
        members_per_passage: Dict[str, set] = {}
        text = self.entity_store.hash_id_to_text
        pe = self._passage_entities or {}
        # Reverse-iterate: for each passage in pe, check overlap with members.
        for pass_hash, ent_list in pe.items():
            page_total = 0.0
            seen_members: set = set()
            for ent_hash, w in ent_list:
                if ent_hash in members:
                    page_total += w
                    seen_members.add(ent_hash)
            if page_total > 0:
                deg_per_passage[pass_hash] = page_total
                members_per_passage[pass_hash] = seen_members
        ranked = sorted(deg_per_passage.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        # Pull (file_id, page_number) via passage_store meta.
        col_fid = self.passage_store.meta_column("file_id")
        col_pn = self.passage_store.meta_column("page_number")
        meta_map = {
            h: (str(fid), int(pn) if pn is not None else None)
            for h, fid, pn in zip(self.passage_store.hash_ids, col_fid, col_pn)
        }
        out: List[Dict[str, Any]] = []
        for pass_hash, w in ranked:
            fid, pn = meta_map.get(pass_hash, ("", None))
            out.append({
                "passage_hash": pass_hash,
                "file_id": fid,
                "page_id": f"p_{int(pn):04d}" if pn is not None else "",
                "page_number": pn,
                "weight": round(float(w), 4),
                "member_surfaces": [text.get(m, "")
                                    for m in list(members_per_passage[pass_hash])[:3]],
            })
        return out

    def cluster_cooccurrences(
        self, cluster_id: str, top_n: int = 8
    ) -> List[Dict[str, Any]]:
        """For ``cluster_id``, find the top-N OTHER clusters whose
        members co-occur in shared sentences (via the pair_via_sentences
        index).  Each entry:
        ``{cluster_id, top_surface, shared_sentences_n}``.

        Captures "what else is this entity usually talked about with"
        — a structural neighbor that bridges multi-hop questions.
        """
        clusters, m2c = self._load_clusters_cached()
        members = set(clusters.get(cluster_id) or [cluster_id])
        if not members:
            return []
        pair_via = self.pair_via_sentences()
        # For each member, look at its pair_via_sentences entries.
        cooccur: Dict[str, int] = {}
        for pair, sents in pair_via.items():
            a, b = list(pair)
            if a in members and b not in members:
                other = m2c.get(b, b)
                if other == cluster_id:
                    continue
                cooccur[other] = cooccur.get(other, 0) + len(sents)
            elif b in members and a not in members:
                other = m2c.get(a, a)
                if other == cluster_id:
                    continue
                cooccur[other] = cooccur.get(other, 0) + len(sents)
        ranked = sorted(cooccur.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        text = self.entity_store.hash_id_to_text
        out: List[Dict[str, Any]] = []
        for cid, sn in ranked:
            top_surf_list = self.cluster_top_surfaces(cid, top_n=1)
            top_surf = top_surf_list[0]["surface"] if top_surf_list else text.get(cid, "")
            out.append({
                "cluster_id": cid,
                "top_surface": top_surf,
                "shared_sentences_n": int(sn),
            })
        return out

    def cluster_alias_audit(self, cluster_id: str) -> Dict[str, Any]:
        """Return aggregate alias-edge quality for a cluster:
        ``{n_alias_edges, cos_min, cos_avg, reranker_avg, accepted_by}``.
        Lets the agent decide whether to TRUST a cluster's
        cross-surface aggregation (e.g. avoid a low-cos / sketchy
        merge for the final answer).
        """
        clusters, _ = self._load_clusters_cached()
        members = set(clusters.get(cluster_id) or [cluster_id])
        if self.graph is None or len(members) <= 1:
            return {"n_alias_edges": 0}
        cos_list: List[float] = []
        rerank_list: List[float] = []
        accepted_by: Dict[str, int] = {}
        # Walk alias edges; keep those with both endpoints inside the cluster.
        member_vidx = {self._name_to_vidx[m] for m in members
                       if m in self._name_to_vidx}
        if not member_vidx:
            return {"n_alias_edges": 0}
        for e in self.graph.es.select(edge_type="alias"):
            if e.source not in member_vidx or e.target not in member_vidx:
                continue
            attrs = e.attributes()
            fj = attrs.get("features_json")
            if fj:
                try:
                    import json as _json
                    feats = _json.loads(fj)
                    cs = feats.get("cos_sim")
                    rs = feats.get("reranker_score")
                    ab = feats.get("accepted_by")
                    if cs is not None: cos_list.append(float(cs))
                    if rs is not None: rerank_list.append(float(rs))
                    if ab: accepted_by[ab] = accepted_by.get(ab, 0) + 1
                except Exception:
                    pass
        result: Dict[str, Any] = {"n_alias_edges": len(cos_list)}
        if cos_list:
            result["cos_min"] = round(min(cos_list), 4)
            result["cos_avg"] = round(sum(cos_list) / len(cos_list), 4)
        if rerank_list:
            result["reranker_avg"] = round(sum(rerank_list) / len(rerank_list), 4)
        if accepted_by:
            result["accepted_by"] = accepted_by
        return result

    def list_top_clusters(
        self,
        top_n: int = 20,
        sort_by: str = "size",
        min_size: int = 2,
        surface_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List top clusters by ``size`` / ``mention_weight`` /
        ``alias_cos_avg``. Used by the new list_clusters mode for
        agent + workbench audit.
        """
        self._build_entity_passage_indexes()
        clusters, _ = self._load_clusters_cached()
        text = self.entity_store.hash_id_to_text
        deg = self._entity_passage_degree or {}
        rows: List[Dict[str, Any]] = []
        sf = surface_filter.lower() if surface_filter else None
        for cid, members in clusters.items():
            if len(members) < min_size:
                continue
            top_surfs = self.cluster_top_surfaces(cid, top_n=3)
            if sf and not any(sf in s.get("surface", "").lower() for s in top_surfs):
                continue
            mention_weight = sum(float(deg.get(m, 0.0)) for m in members)
            rows.append({
                "cluster_id": cid,
                "size": len(members),
                "top_surfaces": [s["surface"] for s in top_surfs],
                "mention_weight": round(mention_weight, 2),
            })
        key_map = {
            "size": lambda r: r["size"],
            "mention_weight": lambda r: r["mention_weight"],
        }
        sort_key = key_map.get(sort_by, key_map["size"])
        rows.sort(key=sort_key, reverse=True)
        return rows[:top_n]

    def _load_clusters_cached(self) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Load + invert clusters once per channel; cache for subsequent
        queries.  Returns ``(clusters, member_to_cluster)`` as
        ``{cluster_id: [member_hash, ...]}`` and
        ``{member_hash: cluster_id}``.  Empty dicts on any failure
        (logged, not raised).

        Normalizes ``get_clusters`` / ``compute_clusters_for_collapse``
        output (``List[{"id":, "members":, ...}]``) to a plain
        ``Dict[cluster_id, members]`` for callers.
        """
        if self._cluster_cache is not None and self._member_to_cluster_cache is not None:
            return self._cluster_cache, self._member_to_cluster_cache
        if self.graph is None:
            self._cluster_cache = {}
            self._member_to_cluster_cache = {}
            return self._cluster_cache, self._member_to_cluster_cache
        try:
            handler = getattr(self.linear_config, "acceptance_handler", "overlay")
            if handler in ("collapse_basic", "collapse_provenance"):
                reverse_path = faiss_graph_dir() / "reverse_map.json"
                reverse_map = load_reverse_map(reverse_path)
                raw_rows = compute_clusters_for_collapse(self.graph, reverse_map)
            else:
                raw_rows = get_clusters(
                    self.graph,
                    faiss_graph_dir() / "clusters.json",
                    algorithm=getattr(self.linear_config,
                                      "cluster_algorithm",
                                      "connected_components"),
                    leiden_resolution=getattr(self.linear_config,
                                              "cluster_leiden_resolution", 0.05),
                    leiden_weighted=getattr(self.linear_config,
                                            "cluster_leiden_weighted", True),
                )
        except Exception as exc:
            logger.warning("graph_ppr: cluster load failed (%s)", exc)
            raw_rows = []
        # Normalize: source format is ``List[Dict]`` with ``id`` and
        # ``members`` keys (also ``canonical``).  Singletons missing
        # from the rows are implicitly under their own hash via the
        # ``aggregate_by_cluster`` contract; we'll handle "no row =
        # singleton" at the caller side via ``.get(name, name)``.
        clusters: Dict[str, List[str]] = {}
        if isinstance(raw_rows, list):
            for i, row in enumerate(raw_rows):
                if not isinstance(row, dict):
                    continue
                members = list(row.get("members") or [])
                if not members:
                    continue
                cid = str(row.get("id") or row.get("canonical") or f"c_{i:06d}")
                clusters[cid] = members
        elif isinstance(raw_rows, dict):
            # Accept ``{cluster_id: members}`` as an alternative shape.
            for cid, members in raw_rows.items():
                clusters[str(cid)] = list(members)
        member_to_cluster: Dict[str, str] = {}
        for cluster_id, members in clusters.items():
            for m in members:
                member_to_cluster[m] = cluster_id
        self._cluster_cache = clusters
        self._member_to_cluster_cache = member_to_cluster
        return clusters, member_to_cluster

    def _spread_seeds_to_clusters(
        self,
        seeds: List[Tuple[str, int, float]],
    ) -> List[Tuple[str, int, float]]:
        """Augment ``seeds`` with their alias-cluster siblings.

        For each seed entity, find the cluster it belongs to and add
        up to ``_CLUSTER_SPREAD_MAX_MEMBERS`` other physical members
        (that are actual graph vertices) as additional seeds with the
        same sim score scaled by ``1 / sqrt(|cluster|)`` so a huge
        cluster doesn't dominate the PPR personalization vector. Hub
        clusters formed by alias-percolation can absorb a large fraction
        of all entities; ``_CLUSTER_SPREAD_MAX_MEMBERS`` caps the spread
        from any single cluster.

        Returns the union (original seeds preferred when a hash_id
        appears in both, since the original score came from direct
        NER-match and is more reliable than cluster-spread).
        """
        if not seeds or self.graph is None:
            return seeds
        clusters, member_to_cluster = self._load_clusters_cached()
        if not clusters:
            return seeds

        import math as _m
        existing = {hid for hid, _, _ in seeds}
        augmented: List[Tuple[str, int, float]] = list(seeds)
        # Per-call trace counters — surfaced on ``last_debug`` for the
        # agent / workbench / postmortem to inspect spread behavior.
        total_added = 0
        total_truncated = 0  # cluster members skipped due to cap
        max_cluster_seen = 0
        for hid, _vidx, sim in seeds:
            cluster_id = member_to_cluster.get(hid)
            if cluster_id is None:
                continue
            members = clusters.get(cluster_id, [hid])
            if len(members) <= 1:
                continue
            max_cluster_seen = max(max_cluster_seen, len(members))
            damp = 1.0 / _m.sqrt(float(len(members)))
            shared_sim = float(sim) * damp
            added = 0
            considered = 0
            for member_hid in members:
                considered += 1
                if added >= self._CLUSTER_SPREAD_MAX_MEMBERS:
                    total_truncated += max(
                        0, len(members) - considered + 1
                    )
                    break
                if member_hid == hid or member_hid in existing:
                    continue
                if member_hid not in self._name_to_vidx:
                    continue
                m_vidx = self._name_to_vidx[member_hid]
                if self._is_hidden(m_vidx):
                    continue
                augmented.append((member_hid, m_vidx, shared_sim))
                existing.add(member_hid)
                added += 1
                total_added += 1
        self.last_debug["seed_spread_added"] = total_added
        self.last_debug["seed_spread_truncated"] = total_truncated
        self.last_debug["seed_spread_max_cluster"] = max_cluster_seen
        return augmented

    # Cap on literal-gazetteer-fallback seeds. The fallback fires when
    # the question has weak NER signal (< 2 seeds OR max_sim < 0.7) and
    # a long question can otherwise inject dozens of common-noun matches
    # as PPR seeds, drowning out the few entities that actually matter.
    # 8 covers the typical multi-hop "X of Y that Z" + supporting noun
    # phrase set without flooding.
    _FALLBACK_LITERAL_MAX_SEEDS = 8

    def _fallback_seeds(self, question: str) -> Dict[str, Tuple[int, float]]:
        """Two-stage no-NER seed recovery; see :meth:`_seed_entities`."""
        # 1) Cheap word-boundary substring scan.
        gaz = self._ensure_gazetteer()
        if gaz is not None:
            counts = find_literal_matches(question, gaz)
            # Cap to top-N hits ranked by match count so a long question
            # cannot inject dozens of common-noun gazetteer hits as PPR
            # seeds. Distinct hits in the same question are themselves
            # equally weighted (score 1.0) — the cap is a guard against
            # gazetteer flood, not a within-question relevance signal.
            ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            literal_hits: Dict[str, Tuple[int, float]] = {}
            for hid, _count in ranked[: self._FALLBACK_LITERAL_MAX_SEEDS]:
                if hid in self._name_to_vidx:
                    literal_hits[hid] = (self._name_to_vidx[hid], 1.0)
            if literal_hits:
                self.last_debug["fallback"] = "gazetteer_literal"
                return literal_hits

        # 2) Whole-question embedding ➜ entity-store top-K.
        try:
            q_emb = self.embedding_client.encode(question, is_query=True)
        except Exception as exc:
            logger.warning("graph_ppr: question embedding failed (%s); no fallback seeds", exc)
            return {}
        if q_emb.ndim == 2:
            q_emb = q_emb[0]

        top_k = max(3, self.config.ppr_topk // 2 if hasattr(self.config, "ppr_topk") else 3)
        top = self.entity_store.topk(q_emb, top_k)
        # Loose threshold — fallback only fires when NER found nothing,
        # so we'd rather have low-confidence seeds than zero.
        floor = 0.45
        embed_hits: Dict[str, Tuple[int, float]] = {}
        for hid, score in top:
            if score < floor:
                break  # topk is sorted desc
            if hid in self._name_to_vidx:
                embed_hits[hid] = (self._name_to_vidx[hid], float(score))
        if embed_hits:
            self.last_debug["fallback"] = "question_embedding"
        return embed_hits

    def _ensure_gazetteer(self):
        """Lazily build the Aho-Corasick automaton over entity surfaces."""
        if self._gazetteer_automaton is not None:
            return self._gazetteer_automaton
        if len(self.entity_store) == 0:
            return None
        surfaces = self.entity_store.hash_id_to_text
        # Mirror the ingest-time backfill defaults — same admin knob
        # tunes both ingest backfill and query-time PPR gazetteer:
        # short / single-word surfaces get filtered to avoid spurious
        # "us" / "irs" hits. Sourced from ``self.linear_config`` which
        # ConfigStore.materialize_linear_rag_config() injects per
        # request from admin overrides.
        automaton, n_kept = build_gazetteer_automaton(
            surfaces,
            min_surface_chars=self.linear_config.literal_backfill_min_chars,
            multi_word_only=self.linear_config.literal_backfill_multi_word_only,
        )
        if n_kept == 0:
            return None
        self._gazetteer_automaton = automaton
        return self._gazetteer_automaton

    # ----------------------------------------------------------- entity BFS

    def _calculate_entity_scores(
        self,
        question_emb: np.ndarray,
        seeds: List[Tuple[str, int, float]],
    ) -> Tuple[np.ndarray, Dict[str, Tuple[int, float, int]]]:
        cfg = self.config
        n_vertices = self.graph.vcount()
        entity_weights = np.zeros(n_vertices, dtype=np.float64)

        # HippoRAG node-specificity: scale each query seed's reset mass by
        # s_i = 1/|P_i| (inverse passage frequency) so a phrase appearing in
        # many passages (a hub) seeds little mass. ``actived`` keeps the raw
        # similarity so the downstream semantic-expansion BFS is unchanged;
        # only the reset vector is damped.
        df = self._entity_passage_df() if getattr(cfg, "ppr_node_specificity", False) else None
        actived: Dict[str, Tuple[int, float, int]] = {}
        for hid, vidx, sim in seeds:
            actived[hid] = (vidx, sim, 1)
            w = sim
            if df is not None:
                w = sim / max(1, df.get(vidx, 0))
            entity_weights[vidx] = w

        if not seeds:
            return entity_weights, actived

        ent_to_sents, sent_to_ents = self._mention_maps()

        # Pre-compute sentence similarity once so the BFS can index in O(1).
        sentence_hash_ids = self.sentence_store.hash_ids
        sentence_idx_lookup = {h: i for i, h in enumerate(sentence_hash_ids)}
        if not sentence_hash_ids:
            return entity_weights, actived
        # faiss search instead of reconstruct + np.dot — avoids a per-
        # request (N_sent, D) temporary; identical IP arithmetic.
        sentence_similarities = self.sentence_store.all_similarities(question_emb)

        used_sentences: set = set()
        current = dict(actived)
        iteration = 1
        while current and iteration < cfg.ppr_max_iterations:
            new_layer: Dict[str, Tuple[int, float, int]] = {}
            for entity_hash_id, (_, entity_score, _tier) in current.items():
                if entity_score < cfg.ppr_iteration_threshold:
                    continue
                cand_sentences = [
                    sid for sid in ent_to_sents.get(entity_hash_id, [])
                    if sid not in used_sentences
                ]
                if not cand_sentences:
                    continue
                sims = np.array(
                    [sentence_similarities[sentence_idx_lookup[sid]] for sid in cand_sentences],
                    dtype=np.float64,
                )
                top_idx = np.argsort(sims)[::-1][: cfg.ppr_top_k_sentence]
                for ti in top_idx:
                    sent_hash_id = cand_sentences[int(ti)]
                    sent_score = float(sims[int(ti)])
                    used_sentences.add(sent_hash_id)
                    for next_entity_hash_id in sent_to_ents.get(sent_hash_id, []):
                        next_score = entity_score * sent_score
                        if next_score < cfg.ppr_iteration_threshold:
                            continue
                        if next_entity_hash_id not in self._name_to_vidx:
                            continue
                        next_vidx = self._name_to_vidx[next_entity_hash_id]
                        entity_weights[next_vidx] += next_score
                        new_layer[next_entity_hash_id] = (
                            next_vidx,
                            next_score,
                            iteration + 1,
                        )
            actived.update(new_layer)
            current = new_layer
            iteration += 1
        return entity_weights, actived

    def _mention_maps(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        if self._entity_to_sents is not None and self._sent_to_entities is not None:
            return self._entity_to_sents, self._sent_to_entities

        ent_to_sents: Dict[str, List[str]] = defaultdict(list)
        sent_to_ents: Dict[str, List[str]] = defaultdict(list)
        if not self._ner_path.is_file():
            self._entity_to_sents = ent_to_sents
            self._sent_to_entities = sent_to_ents
            return ent_to_sents, sent_to_ents

        cache = json.loads(self._ner_path.read_text(encoding="utf-8"))
        sentence_to_entities = cache.get("sentence_to_entities", {}) or {}
        sentence_text_to_hash = self.sentence_store.text_to_hash_id
        entity_text_to_hash = self.entity_store.text_to_hash_id
        for sent_text, ent_surfaces in sentence_to_entities.items():
            sent_hash = sentence_text_to_hash.get(sent_text)
            if sent_hash is None:
                continue
            for ent_text in ent_surfaces:
                ent_hash = entity_text_to_hash.get(ent_text)
                if ent_hash is None:
                    continue
                if sent_hash not in ent_to_sents[ent_hash]:
                    ent_to_sents[ent_hash].append(sent_hash)
                if ent_hash not in sent_to_ents[sent_hash]:
                    sent_to_ents[sent_hash].append(ent_hash)
        self._entity_to_sents = ent_to_sents
        self._sent_to_entities = sent_to_ents
        return ent_to_sents, sent_to_ents

    def pair_via_sentences(self) -> Dict[frozenset, List[str]]:
        """Reverse index ``{entity_a_hash, entity_b_hash} → [sent_hash, ...]``.

        Drives query-time typed-edge scoring in graph_explore's
        ``chain`` mode (no LLM at build time — the natural-language
        sentence in which two entities co-occur IS the implicit
        predicate, scored per-query by cos(q_emb, sent_emb)). Built
        lazily from :meth:`_mention_maps` on first access; takes ~O(Σ
        |entities_per_sentence|²) which for typical NER (≤5 entities /
        sentence) is dominated by Σ |entities_per_sentence|.

        Storage: ``frozenset({a, b})`` for the key so direction-free
        lookup is O(1) and the index is symmetric (the underlying graph
        is undirected). Singleton sentences (one or zero entities) are
        skipped — they contribute no edges. Duplicate hashes within a
        single sentence are de-duped before pairing so reflexive
        ``(a, a)`` entries never appear.
        """
        if self._pair_via_sentences is not None:
            return self._pair_via_sentences
        _, sent_to_ents = self._mention_maps()
        pair_idx: Dict[frozenset, List[str]] = {}
        for sent_hash, ent_list in sent_to_ents.items():
            uniq = list(dict.fromkeys(ent_list))  # preserve order, dedup
            if len(uniq) < 2:
                continue
            for i in range(len(uniq)):
                a = uniq[i]
                for j in range(i + 1, len(uniq)):
                    b = uniq[j]
                    key = frozenset((a, b))
                    bucket = pair_idx.get(key)
                    if bucket is None:
                        pair_idx[key] = [sent_hash]
                    elif sent_hash not in bucket:
                        bucket.append(sent_hash)
        self._pair_via_sentences = pair_idx
        return pair_idx

    def passage_sentence_embs(
        self, passage_hash: str
    ) -> List[Tuple[str, np.ndarray]]:
        """Return ``[(sentence_text, sentence_embedding), ...]`` for a
        page, in the original sentence order on the page.

        Powers the question-conditioned preview snippet that
        graph_explore's PPR observation surfaces on each candidate
        page: agent picks the single highest-cos(q_emb, sent_emb)
        sentence to discriminate "this page answers my question" from
        "this page mentions the topic but isn't the answer".

        The underlying ``passage_hash → [sentence_text]`` map is
        recorded at ingest time inside ``ner_results.json``
        (``passage_hash_id_to_sentences``); we only translate each
        text to its embedding here via the sentence_store's hash
        function. Built once on first access (resolves ~225 k
        sentence texts to hashes in one pass over the persisted map),
        then served from memory.

        Returns an empty list when the page has no recorded
        sentences (empty passage) or when the sentence text was
        post-filtered out of the sentence_store (rare; would mean
        ingest and serve schemas diverged).
        """
        if self._passage_sentence_embs is None:
            with self._call_lock:
                if self._passage_sentence_embs is None:
                    self._build_passage_sentence_embs()
        return self._passage_sentence_embs.get(passage_hash, [])

    def _build_passage_sentence_embs(self) -> None:
        """Load passage→sentence-text map from ``ner_results.json`` and
        resolve each sentence text to its (text, embedding) pair via the
        sentence_store. The map is a build-time artifact; this method
        does not run sentence splitting or NER — pure dict translation.

        Pre-materialises the full sentence embedding matrix ONCE via
        ``sentence_store.embeddings`` (a faiss
        ``reconstruct_n(0, ntotal)`` call) and indexes into it row-wise.
        Naively calling ``ss.embeddings[idx]`` inside the loop would
        reconstruct the full matrix on every access (the property has
        no internal cache), turning a ~225 k-sentence build from
        seconds into hours.
        """
        if not self._ner_path.is_file():
            self._passage_sentence_embs = {}
            return
        cache = json.loads(self._ner_path.read_text(encoding="utf-8"))
        passage_to_sent_text: Dict[str, List[str]] = (
            cache.get("passage_hash_id_to_sentences") or {}
        )
        ss = self.sentence_store
        # The sentence_store's hash function namespaces texts under
        # ``"sentence"`` (same hash that ingest used to dedup adds).
        hash_for = ss.hash_for
        hash_id_to_text = ss.hash_id_to_text
        idx_map = ss._hash_id_to_idx
        all_embs = ss.embeddings  # one faiss reconstruct_n, cached locally below
        out: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        for passage_hash, sent_texts in passage_to_sent_text.items():
            pairs: List[Tuple[str, np.ndarray]] = []
            for s_text in sent_texts:
                sid = hash_for(s_text)
                idx = idx_map.get(sid)
                if idx is None:
                    continue
                # Use the canonical text stored under this hash so the
                # snippet we surface matches what the index encoded
                # (post-strip / post-normalise differences would
                # otherwise leak through).
                canonical = hash_id_to_text.get(sid, s_text)
                pairs.append((canonical, all_embs[idx]))
            out[passage_hash] = pairs
        self._passage_sentence_embs = out

    # ----------------------------------------------------------- passage scoring

    def _calculate_passage_scores(
        self,
        question: str,
        question_emb: np.ndarray,
        actived: Dict[str, Tuple[int, float, int]],
    ) -> np.ndarray:
        cfg = self.config
        n_vertices = self.graph.vcount()
        passage_weights = np.zeros(n_vertices, dtype=np.float64)

        if len(self.passage_store) == 0:
            return passage_weights
        # faiss search instead of reconstruct + np.dot — saves a per-
        # request (N_passage, D) temporary on large corpora.
        sims = self.passage_store.all_similarities(question_emb)

        # Min-max normalize so the dpr term lives on [0, 1] before being
        # combined with the log entity bonus. When all values are equal
        # we return ones (NOT zeros), so a single-passage corpus or
        # all-tied DPR doesn't zero out passage weights.
        if sims.size > 0:
            lo, hi = float(sims.min()), float(sims.max())
            if hi > lo:
                norm_dpr = (sims - lo) / (hi - lo)
            else:
                norm_dpr = np.ones_like(sims)
        else:
            norm_dpr = sims

        passage_hash_ids = self.passage_store.hash_ids
        hash_id_to_text = self.passage_store.hash_id_to_text
        entity_hash_to_text = self.entity_store.hash_id_to_text

        for i, passage_hash_id in enumerate(passage_hash_ids):
            if passage_hash_id not in self._name_to_vidx:
                continue
            passage_text_lower = hash_id_to_text[passage_hash_id].lower()
            total_entity_bonus = 0.0
            for entity_hash_id, (_, entity_score, tier) in actived.items():
                surface = entity_hash_to_text.get(entity_hash_id, "")
                if not surface:
                    continue
                occurrences = passage_text_lower.count(surface.lower())
                if occurrences <= 0:
                    continue
                denom = tier if tier >= 1 else 1
                total_entity_bonus += entity_score * math.log(1 + occurrences) / denom

            passage_score = (
                cfg.ppr_passage_ratio * float(norm_dpr[i])
                + math.log(1 + total_entity_bonus)
            )
            passage_weights[self._name_to_vidx[passage_hash_id]] = (
                passage_score * cfg.ppr_passage_node_weight
            )
        return passage_weights

    # ------------------------------------------------- hub suppression

    def _entity_passage_df(self) -> Dict[int, int]:
        """Per-entity passage frequency (count of incident entity_passage
        edges), keyed by vertex index and cached by edge count.

        Feeds HippoRAG node-specificity (seed side, ``_calculate_entity_scores``)
        and SPRIG df-damping (transition side). One O(E) pass; recomputed only
        when the graph's edge count changes (store refresh)."""
        ec = self.graph.ecount()
        cache = getattr(self, "_entity_df_cache", None)
        if cache is not None and cache[0] == ec:
            return cache[1]
        df: Dict[int, int] = {}
        if (
            "edge_type" in self.graph.es.attributes()
            and "vertex_type" in self.graph.vs.attributes()
        ):
            vtype = self.graph.vs["vertex_type"]
            for e in self.graph.es:
                if e["edge_type"] != "entity_passage":
                    continue
                s, t = e.source, e.target
                if vtype[s] == "entity":
                    df[s] = df.get(s, 0) + 1
                if vtype[t] == "entity":
                    df[t] = df.get(t, 0) + 1
        self._entity_df_cache = (ec, df)
        return df

    def _hub_damped_weights(self, p: float) -> List[float]:
        """Per-edge PPR weights with entity_passage edges scaled by
        df(entity)^(-p) (SPRIG). Non-entity_passage edges keep their raw
        weight. Cached by (edge_count, p)."""
        ec = self.graph.ecount()
        cache = getattr(self, "_hub_damped_cache", None)
        if cache is not None and cache[0] == ec and cache[1] == p:
            return cache[2]
        df = self._entity_passage_df()
        base = self.graph.es["weight"]
        has_et = "edge_type" in self.graph.es.attributes()
        vtype = (
            self.graph.vs["vertex_type"]
            if "vertex_type" in self.graph.vs.attributes() else None
        )
        out: List[float] = []
        for i, e in enumerate(self.graph.es):
            w = float(base[i])
            if has_et and vtype is not None and e["edge_type"] == "entity_passage":
                ent = (
                    e.source if vtype[e.source] == "entity"
                    else (e.target if vtype[e.target] == "entity" else None)
                )
                if ent is not None:
                    d = df.get(ent, 0)
                    if d > 1:
                        w *= d ** (-p)
            out.append(w)
        self._hub_damped_cache = (ec, p, out)
        return out

    # ----------------------------------------------------------- PPR

    def _run_ppr(self, node_weights: np.ndarray) -> List[Tuple[str, float]]:
        ranked, _scores_arr = self._run_ppr_full(node_weights)
        return ranked

    def _run_ppr_full(
        self, node_weights: np.ndarray
    ) -> Tuple[List[Tuple[str, float]], Optional[np.ndarray]]:
        """Run PPR and return both the ranked passage list and the full
        per-vertex score array.

        Dispatches between physical and logical PPR per
        ``RAGConfig.ppr_on_logical``. The two paths take the same
        physical ``node_weights`` vector (already populated by
        ``_calculate_entity_scores`` + ``_calculate_passage_scores``);
        the logical path projects the seed mass through the cluster
        contraction before running PPR on the quotient graph and
        then back-projects scores to physical passages.

        The score array enables post-hoc views (cluster_scores, surface
        attribution) without re-running PPR; it's used internally by
        :meth:`retrieve_subgraph` and the agent's graph_explore tool.
        Returns ``([], None)`` when ``reset.sum() <= 0`` (no seed mass
        to propagate).
        """
        cfg = self.config
        if getattr(cfg, "ppr_on_logical", False):
            return self._run_ppr_full_logical(node_weights)
        reset = np.where(np.isnan(node_weights) | (node_weights < 0), 0.0, node_weights)
        if reset.sum() <= 0:
            return [], None
        # SPRIG transition-side hub damping: scale entity_passage edge weights
        # by df(entity)^(-p) so PPR mass flows less freely into high-degree
        # entities. p=0 → use the raw "weight" attribute (zero-copy).
        weights_param = "weight" if "weight" in self.graph.es.attributes() else None
        p = float(getattr(cfg, "ppr_hub_damping_p", 0.0) or 0.0)
        if p > 0 and weights_param is not None:
            weights_param = self._hub_damped_weights(p)
        scores = self.graph.personalized_pagerank(
            vertices=range(self.graph.vcount()),
            damping=cfg.ppr_damping,
            directed=False,
            weights=weights_param,
            reset=reset.tolist(),
            implementation="prpack",
        )
        scores_arr = np.asarray(scores, dtype=np.float64)
        if not self._passage_vidx:
            return [], scores_arr
        passage_scores = scores_arr[self._passage_vidx]
        order = np.argsort(passage_scores)[::-1]
        out: List[Tuple[str, float]] = []
        for j in order[: max(self.config.ppr_topk * 4, self.config.ppr_topk)]:
            vidx = self._passage_vidx[int(j)]
            name = self.graph.vs[vidx]["name"]
            out.append((str(name), float(passage_scores[int(j)])))
        return out, scores_arr

    # ----------------------------------------------------------- logical PPR

    def _ensure_quotient_graph(self) -> Optional[ig.Graph]:
        """Build the alias-cluster-contracted quotient graph once.

        Construction:
          * For each entity vertex, look up its alias-cluster id.
            Members of a cluster collapse to one super-node.
            Singletons become super-nodes named after their own hash.
          * Passage vertices are preserved 1:1 (they don't collapse).
          * Edges:
            - alias edges: dropped (they're WITHIN super-nodes now).
            - entity-passage: source maps to its super-node;
              multi-edges merged by sum of weights.
            - adjacent-passage / sentence-passage: preserved as-is.
          * The quotient is constructed in O(V + E) once; subsequent
            retrieves use the cached graph.

        Used only when ``RAGConfig.ppr_on_logical=True``. The physical
        graph is the source of truth; this is a retrieval-time
        projection that the user can toggle without touching storage.
        """
        if self._quotient_graph is not None:
            return self._quotient_graph
        if self.graph is None:
            return None
        import time as _time
        _t0 = _time.time()
        clusters, member_to_cluster = self._load_clusters_cached()
        if not clusters:
            logger.warning("graph_ppr: quotient cluster load returned empty")
            return None

        # Each super-node has a synthetic name distinct from any
        # physical id. We use the cluster_id directly (callers know
        # they're looking at logical ids).
        is_entity = ("vertex_type" in self.graph.vs.attributes())
        super_names: List[str] = []
        super_name_to_idx: Dict[str, int] = {}

        # First pass: entity super-nodes. For each entity, find its
        # cluster_id and ensure we have a super-name for it.
        for v in self.graph.vs:
            if is_entity and v["vertex_type"] != "entity":
                continue
            name = v["name"]
            cluster_id = member_to_cluster.get(name, name)
            if cluster_id not in super_name_to_idx:
                super_name_to_idx[cluster_id] = len(super_names)
                super_names.append(cluster_id)

        n_entity_super = len(super_names)
        # Second pass: passage / sentence vertices kept 1:1.
        passage_super_vidx: List[int] = []
        for v in self.graph.vs:
            if is_entity and v["vertex_type"] == "entity":
                continue
            name = v["name"]
            if name not in super_name_to_idx:
                super_name_to_idx[name] = len(super_names)
                super_names.append(name)
            if (not is_entity) or v["vertex_type"] == "passage":
                passage_super_vidx.append(super_name_to_idx[name])

        # Build the quotient graph.
        q = ig.Graph(directed=False)
        q.add_vertices(len(super_names))
        q.vs["name"] = super_names
        # vertex_type flag — useful for downstream debugging.
        if is_entity:
            types = ["entity"] * n_entity_super + ["passage"] * (len(super_names) - n_entity_super)
            q.vs["vertex_type"] = types

        # Edge aggregation: (super_u, super_v) → summed weight. We drop
        # alias edges (they're within-cluster by definition) and
        # ignore reflexive after-collapse edges. **Critical**: map each
        # physical endpoint through ``member_to_cluster`` before the
        # super-name lookup — entity vertices were registered in
        # ``super_name_to_idx`` UNDER THEIR CLUSTER_ID (e.g. ``c_0000``),
        # not their physical hash, so a direct ``super_name_to_idx[
        # physical_hash]`` would silently miss every clustered member's
        # incident edges.
        edge_weight: Dict[Tuple[int, int], float] = {}
        # The builder writes the edge kind under ``edge_type``; check
        # that attribute (not the generic ``type``) so alias edges are
        # actually filtered out of the quotient.
        has_edge_type = "edge_type" in self.graph.es.attributes()
        has_weight = "weight" in self.graph.es.attributes()
        # SPRIG transition-side hub damping, applied to entity_passage edges
        # before they aggregate into super-edges — same df(entity)^(-p) rule as
        # the physical path so the two views are consistent. p=0 disables.
        p_hub = float(getattr(self.config, "ppr_hub_damping_p", 0.0) or 0.0)
        df_map = self._entity_passage_df() if p_hub > 0 else None
        vtype_q = (
            self.graph.vs["vertex_type"]
            if "vertex_type" in self.graph.vs.attributes() else None
        )
        for e in self.graph.es:
            if has_edge_type and e["edge_type"] == "alias":
                continue
            u_name = self.graph.vs[e.source]["name"]
            v_name = self.graph.vs[e.target]["name"]
            u_key = member_to_cluster.get(u_name, u_name)
            v_key = member_to_cluster.get(v_name, v_name)
            su = super_name_to_idx.get(u_key)
            sv = super_name_to_idx.get(v_key)
            if su is None or sv is None or su == sv:
                continue
            key = (su, sv) if su < sv else (sv, su)
            w = float(e["weight"]) if has_weight else 1.0
            if (
                df_map is not None and has_edge_type
                and e["edge_type"] == "entity_passage" and vtype_q is not None
            ):
                ent = (
                    e.source if vtype_q[e.source] == "entity"
                    else (e.target if vtype_q[e.target] == "entity" else None)
                )
                if ent is not None:
                    d = df_map.get(ent, 0)
                    if d > 1:
                        w *= d ** (-p_hub)
            edge_weight[key] = edge_weight.get(key, 0.0) + w

        if edge_weight:
            edges = list(edge_weight.keys())
            weights = list(edge_weight.values())
            q.add_edges(edges)
            q.es["weight"] = weights

        self._quotient_graph = q
        self._quotient_member_to_super = {
            m: super_name_to_idx[member_to_cluster.get(m, m)]
            for m in self._name_to_vidx
            if member_to_cluster.get(m, m) in super_name_to_idx
        }
        self._quotient_super_passage_vidx = passage_super_vidx
        build_ms = int((_time.time() - _t0) * 1000)
        self.last_debug["quotient_build_ms"] = build_ms
        self.last_debug["quotient_vcount"] = q.vcount()
        self.last_debug["quotient_ecount"] = q.ecount()
        logger.info(
            "graph_ppr: built quotient graph V=%d E=%d (physical V=%d E=%d) in %d ms",
            q.vcount(), q.ecount(), self.graph.vcount(), self.graph.ecount(), build_ms,
        )
        return self._quotient_graph

    def _run_ppr_full_logical(
        self, node_weights: np.ndarray
    ) -> Tuple[List[Tuple[str, float]], Optional[np.ndarray]]:
        """Run PPR on the alias-cluster-contracted quotient graph.

        Project physical seeds → super-nodes (summing reset mass per
        super-node), run igraph PPR on the quotient, then materialize
        passages by reading the super-passage scores. Returns
        ``(ranked_physical_hashes, physical_scores_arr)`` so the
        caller's downstream code (workbench, cluster_scores debug)
        sees the same shape it always did.

        Back-projection of mass to physical passages is direct
        because passages are not collapsed (they keep their physical
        identity in the quotient).
        """
        cfg = self.config
        q = self._ensure_quotient_graph()
        if q is None or q.vcount() == 0:
            logger.warning("graph_ppr: quotient graph unavailable; fallback to physical PPR")
            # Strip the flag temporarily to avoid infinite recursion.
            saved = getattr(cfg, "ppr_on_logical", False)
            try:
                cfg.ppr_on_logical = False
                return self._run_ppr_full(node_weights)
            finally:
                cfg.ppr_on_logical = saved

        # Project physical reset mass → super-node reset mass. Use the
        # pre-built ``_quotient_member_to_super`` for entities; for
        # passage / sentence vertices (preserved 1:1 in the quotient)
        # we kept their original names so a name → super lookup also
        # works there.
        q_reset = np.zeros(q.vcount(), dtype=np.float64)
        is_entity = ("vertex_type" in self.graph.vs.attributes())
        # Build a quick name→super index that covers BOTH entities
        # (via cluster mapping) and passage/sentence (1:1). The
        # quotient graph's ``name`` attribute is the source of truth.
        if not hasattr(self, "_quotient_name_to_super") or self._quotient_name_to_super is None:
            self._quotient_name_to_super = {q.vs[i]["name"]: i for i in range(q.vcount())}
        for vidx in range(self.graph.vcount()):
            w = float(node_weights[vidx])
            if w <= 0 or (np.isnan(w) if isinstance(w, float) else False):
                continue
            name = self.graph.vs[vidx]["name"]
            super_idx = self._quotient_member_to_super.get(name)
            if super_idx is None:
                # Passage / sentence vertex (preserved 1:1).
                super_idx = self._quotient_name_to_super.get(name)
                if super_idx is None:
                    continue
            q_reset[super_idx] += w

        if q_reset.sum() <= 0:
            return [], None
        scores = q.personalized_pagerank(
            vertices=range(q.vcount()),
            damping=cfg.ppr_damping,
            directed=False,
            weights="weight" if "weight" in q.es.attributes() else None,
            reset=q_reset.tolist(),
            implementation="prpack",
        )
        scores_arr_q = np.asarray(scores, dtype=np.float64)

        # Back-project to a physical-shape score array so downstream
        # code (cluster_scores debug, retrieve_subgraph) doesn't need
        # to know we ran on the quotient.
        physical_scores = np.zeros(self.graph.vcount(), dtype=np.float64)
        for vidx in range(self.graph.vcount()):
            name = self.graph.vs[vidx]["name"]
            super_idx = self._quotient_member_to_super.get(name)
            if super_idx is None:
                continue
            physical_scores[vidx] = scores_arr_q[super_idx]

        if not self._passage_vidx:
            return [], physical_scores
        passage_scores = physical_scores[self._passage_vidx]
        order = np.argsort(passage_scores)[::-1]
        out: List[Tuple[str, float]] = []
        for j in order[: max(self.config.ppr_topk * 4, self.config.ppr_topk)]:
            vidx = self._passage_vidx[int(j)]
            name = self.graph.vs[vidx]["name"]
            out.append((str(name), float(passage_scores[int(j)])))
        self.last_debug["ppr_on_logical"] = True
        return out, physical_scores

    def _cluster_scores(
        self, scores_arr: Optional[np.ndarray], op: str = "sum"
    ) -> Dict[str, float]:
        """Project the full PPR score array onto logical-cluster scores.

        Reads cluster membership from the on-disk ``clusters.json`` (or
        synthesises it from ``reverse_map.json`` in collapse mode).
        Returns ``{cluster_id_or_hash: aggregated_score}``; singleton
        entities pass through under their hash_id as documented in
        :func:`aggregate_by_cluster`.

        Important caveat: evidence landing — i.e. the passages we
        surface as answers — uses physical-node scores. The
        cluster_scores map is a *post-hoc view* for the workbench /
        agent debug, not a re-ranker.
        """
        if scores_arr is None or self.graph is None:
            return {}
        # Build hash_id → score map over entity vertices only — passages
        # are not clusters and would pollute the singleton pass-through.
        if "vertex_type" not in self.graph.vs.attributes():
            return {}
        scores_by_hash: Dict[str, float] = {}
        for v in self.graph.vs:
            if v["vertex_type"] != "entity":
                continue
            scores_by_hash[v["name"]] = float(scores_arr[v.index])
        # Cluster source — gated by the configured acceptance handler so
        # a stale on-disk ``reverse_map.json`` from a previous collapse run
        # can't silently flip overlay's cluster semantics into a
        # canonical-collapse projection.
        handler = getattr(self.linear_config, "acceptance_handler", "overlay")
        if handler in ("collapse_basic", "collapse_provenance"):
            reverse_path = faiss_graph_dir() / "reverse_map.json"
            reverse_map = load_reverse_map(reverse_path)
            clusters = compute_clusters_for_collapse(self.graph, reverse_map)
        else:
            clusters = get_clusters(
                self.graph,
                faiss_graph_dir() / "clusters.json",
                algorithm=getattr(
                    self.linear_config, "cluster_algorithm", "connected_components"
                ),
                leiden_resolution=getattr(
                    self.linear_config, "cluster_leiden_resolution", 0.05
                ),
                leiden_weighted=getattr(
                    self.linear_config, "cluster_leiden_weighted", True
                ),
            )
        return aggregate_by_cluster(scores_by_hash, clusters, op=op)

    # -------------------------------------------------- public — subgraph

    def retrieve_subgraph(
        self,
        question: str,
        file_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run PPR end-to-end and return seeds + actived entities + ranked
        passages — the data the web visualizer needs.

        Independent of :meth:`retrieve` — the 4-channel RAG path keeps
        its existing contract untouched. We share the same internal
        helpers (``_seed_entities`` etc) so the algorithm is identical
        modulo two intentional differences: (a) ``enable_fallback`` is
        always True (graph workbench wants gazetteer + Q2N seeds, just
        like the agent path), and (b) we return the structured
        intermediate state instead of just the final ``ChannelHit`` list.

        Empty graph or zero seeds returns an empty payload with
        ``mode`` set so the caller can show "no graph match" instead of
        a blank canvas.
        """
        # Same lock as :meth:`retrieve` — prevents the gazetteer /
        # NER state from interleaving with a concurrent agent call
        # on the shared channel singleton AND keeps a concurrent
        # ``reload()`` from swapping the entity_store / graph out
        # mid-payload-assembly. The lock must stay held through
        # ``_materialize_hits`` and the subsequent
        # ``self.entity_store.hash_id_to_text`` read: releasing it
        # between them lets a concurrent refresh pair vertex indices
        # from the old graph with surface text from the new entity
        # store.
        with self._call_lock:
            if self.graph is None or len(self.passage_store) == 0:
                return {"mode": "no_graph", "seeds": [], "actived_entities": {}, "passages": [], "cluster_scores": {}}

            seeds = self._seed_entities(question, enable_fallback=True)
            if not seeds:
                return {"mode": "no_seeds", "seeds": [], "actived_entities": {}, "passages": [], "cluster_scores": {}}

            question_emb = self.embedding_client.encode(question, is_query=True)
            entity_weights, actived = self._calculate_entity_scores(question_emb, seeds)
            passage_weights = self._calculate_passage_scores(
                question, question_emb, actived
            )
            node_weights = entity_weights + passage_weights
            ranked, scores_arr = self._run_ppr_full(node_weights)
            passages = self._materialize_hits(ranked, file_ids)
            cluster_scores = self._cluster_scores(scores_arr)

            # Final payload assembly stays inside the lock for the
            # consistency reason above.
            ent_text = self.entity_store.hash_id_to_text
            return {
                "mode": "ppr",
                "cluster_scores": cluster_scores,
                "seeds": [
                    {
                        "hash_id": hid,
                        "vertex_idx": vidx,
                        "surface": ent_text.get(hid, ""),
                        "sim": float(sim),
                    }
                    for hid, vidx, sim in seeds
                ],
                "actived_entities": {
                    hid: {
                        "vertex_idx": vidx,
                        "surface": ent_text.get(hid, ""),
                        "score": float(score),
                        "iteration_tier": int(tier),
                    }
                    for hid, (vidx, score, tier) in actived.items()
                },
                "passages": passages,  # List[ChannelHit] — has file_id / page_id / score
            }

    # ----------------------------------------------------------- materialize

    def _materialize_hits(
        self,
        ranked: List[Tuple[str, float]],
        file_ids: Optional[List[str]],
    ) -> List[ChannelHit]:
        cfg = self.config
        file_id_filter = set(file_ids) if file_ids else None
        # Read the (file_id, page_number) meta columns once so each lookup
        # is O(1) and we don't probe the parquet table per hit.
        hash_ids = self.passage_store.hash_ids
        col_file_id = self.passage_store.meta_column("file_id")
        col_page_number = self.passage_store.meta_column("page_number")
        meta_lookup: Dict[str, Tuple[Optional[str], Optional[int]]] = {
            h: (fid, pn) for h, fid, pn in zip(hash_ids, col_file_id, col_page_number)
        }

        out: List[ChannelHit] = []
        for passage_hash_id, score in ranked:
            file_id, page_n = meta_lookup.get(passage_hash_id, (None, None))
            if not file_id or page_n is None:
                continue
            if file_id_filter and file_id not in file_id_filter:
                continue
            try:
                page_id = f"p_{int(page_n):04d}"
            except (TypeError, ValueError):
                continue
            out.append(
                ChannelHit(
                    file_id=str(file_id),
                    page_id=page_id,
                    score=score,
                    evidence=[{"passage_hash_id": passage_hash_id}],
                )
            )
            if len(out) >= cfg.ppr_topk:
                break
        return out
