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

        # Lazy gazetteer Aho-Corasick automaton over entity surfaces, used
        # as a query-side seed fallback when NER misses domain product
        # names (e.g. "Heritage Protector Option"). Mirrors the
        # HippoRAG-2 query-to-node seeding idea.
        self._gazetteer_automaton = None

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
            self._gazetteer_automaton = None
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
        # GraphExploreTool does) cannot observe another concurrent
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
           Catches "Heritage Protector Option" or "FATCA" that spaCy's
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

        # Fallback path — only when caller asked AND NER produced nothing.
        if enable_fallback and not best and len(self.entity_store) > 0:
            best = self._fallback_seeds(question)

        return [(hid, vidx, sim) for hid, (vidx, sim) in best.items()]

    def _fallback_seeds(self, question: str) -> Dict[str, Tuple[int, float]]:
        """Two-stage no-NER seed recovery; see :meth:`_seed_entities`."""
        # 1) Cheap word-boundary substring scan.
        gaz = self._ensure_gazetteer()
        if gaz is not None:
            counts = find_literal_matches(question, gaz)
            literal_hits: Dict[str, Tuple[int, float]] = {}
            for hid, count in counts.items():
                if hid in self._name_to_vidx:
                    # Score = 1.0 (literal match is unambiguous); count
                    # not used for ranking since each hit is a distinct
                    # entity in the question.
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

        actived: Dict[str, Tuple[int, float, int]] = {}
        for hid, vidx, sim in seeds:
            actived[hid] = (vidx, sim, 1)
            entity_weights[vidx] = sim

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

    # ----------------------------------------------------------- PPR

    def _run_ppr(self, node_weights: np.ndarray) -> List[Tuple[str, float]]:
        ranked, _scores_arr = self._run_ppr_full(node_weights)
        return ranked

    def _run_ppr_full(
        self, node_weights: np.ndarray
    ) -> Tuple[List[Tuple[str, float]], Optional[np.ndarray]]:
        """Run PPR and return both the ranked passage list and the full
        per-vertex score array.

        The score array enables post-hoc views (cluster_scores, surface
        attribution) without re-running PPR; it's used internally by
        :meth:`retrieve_subgraph` and the agent's graph_explore tool.
        Returns ``([], None)`` when ``reset.sum() <= 0`` (no seed mass
        to propagate).
        """
        cfg = self.config
        reset = np.where(np.isnan(node_weights) | (node_weights < 0), 0.0, node_weights)
        if reset.sum() <= 0:
            return [], None
        scores = self.graph.personalized_pagerank(
            vertices=range(self.graph.vcount()),
            damping=cfg.ppr_damping,
            directed=False,
            weights="weight" if "weight" in self.graph.es.attributes() else None,
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
        # spaCy state from interleaving with a concurrent agent call
        # on the shared channel singleton AND keeps a concurrent
        # ``reload()`` from swapping the entity_store / graph out
        # mid-payload-assembly. Earlier versions released the lock
        # after ``_materialize_hits`` and then read
        # ``self.entity_store.hash_id_to_text`` outside; under refresh
        # that yielded vertex indices from the old graph paired with
        # surface text from the new entity store.
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
