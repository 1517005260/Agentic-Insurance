"""Graph PPR channel — strict port of ``projects/LinearRAG`` retrieval.

Pipeline (one PPR run per query):

1. NER on ``query + " " + rewrite`` → raw question surfaces.
2. ``normalize_for_hash`` each surface so it lives in the same canonical
   space as the stored entities.
3. For each canonical question entity, find the single best-matching stored
   entity by cosine similarity (matches LinearRAG.get_seed_entities).
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
Cluster-level evidence display lives in a separate (future) component
per ``docs/entity_alignment.md`` §4.
"""

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import igraph as ig
import numpy as np

from config import LinearRAGConfig, RAGConfig
from config.settings import (
    faiss_graph_dir,
    faiss_graph_entity_dir,
    faiss_graph_passage_dir,
    faiss_graph_sentence_dir,
    models_root,
)
from ingestion.index.linear_rag.ner import SpacyNER
from ingestion.index.linear_rag.normalize import normalize_for_hash
from model_client import EmbeddingClient
from rag.channels.base import BaseChannel, ChannelHit
from rag.preprocess import QueryContext
from storage import EmbeddingStore


logger = logging.getLogger(__name__)


class GraphPPRChannel(BaseChannel):
    name = "graph_ppr"

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        linear_config: Optional[LinearRAGConfig] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        spacy_model_name: str = "en_core_web_trf",
        zh_spacy_model_name: Optional[str] = "zh_core_web_trf",
    ):
        self.config = config or RAGConfig()
        self.embedding_client = embedding_client or EmbeddingClient()
        self.linear_config = linear_config or LinearRAGConfig(
            embedding_client=self.embedding_client
        )

        self.passage_store = EmbeddingStore(faiss_graph_passage_dir(), namespace="passage")
        self.entity_store = EmbeddingStore(faiss_graph_entity_dir(), namespace="entity")
        self.sentence_store = EmbeddingStore(faiss_graph_sentence_dir(), namespace="sentence")

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
                # Older graphs (pre vertex_type) — fall back to "is in passage store".
                passage_hashes = set(self.passage_store.hash_ids)
                self._passage_vidx = [
                    v.index for v in self.graph.vs if v["name"] in passage_hashes
                ]

        # Lazy NER + lazy mention-graph init — not all queries reach PPR.
        self._spacy: Optional[SpacyNER] = None
        self._spacy_model_name = spacy_model_name
        self._zh_spacy_model_name = zh_spacy_model_name
        self._entity_to_sents: Optional[Dict[str, List[str]]] = None
        self._sent_to_entities: Optional[Dict[str, List[str]]] = None

        # Debug snapshot — populated each retrieve() call. Inspect with
        # ``channel.last_debug`` after a query for tracing.
        self.last_debug: Dict[str, object] = {}

    # ----------------------------------------------------------- public API

    def retrieve(self, ctx: QueryContext) -> List[ChannelHit]:
        self.last_debug = {}
        if self.graph is None or len(self.passage_store) == 0:
            self.last_debug["mode"] = "no_graph"
            return []

        question = (ctx.query + " " + (ctx.rewrite or "")).strip()
        seeds = self._seed_entities(question)

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
        question_emb = self.embedding_client.encode(question)
        entity_weights, actived = self._calculate_entity_scores(question_emb, seeds)
        passage_weights = self._calculate_passage_scores(
            question, question_emb, actived
        )
        node_weights = entity_weights + passage_weights
        self.last_debug["actived_entities"] = len(actived)

        ranked = self._run_ppr(node_weights)
        return self._materialize_hits(ranked, ctx.file_ids)

    # ----------------------------------------------------------- seeding

    def _ensure_spacy(self) -> SpacyNER:
        if self._spacy is None:
            spacy_path = (models_root() / self._spacy_model_name).resolve()
            zh_path: Optional[str] = None
            if self._zh_spacy_model_name:
                cand = (models_root() / self._zh_spacy_model_name).resolve()
                if (cand / "config.cfg").is_file():
                    zh_path = str(cand)
            self._spacy = SpacyNER(str(spacy_path), zh_spacy_model=zh_path)
        return self._spacy

    def _seed_entities(self, question: str) -> List[Tuple[str, int, float]]:
        """Return ``[(hash_id, vertex_idx, sim), ...]`` — at most one per
        question entity, deduped by hash_id (best score wins).
        """
        try:
            spacy = self._ensure_spacy()
        except Exception as exc:  # spaCy model missing → skip seeds, run dense-only
            logger.warning("graph_ppr: NER unavailable (%s); falling back to dense-only", exc)
            return []
        raw_surfaces = spacy.question_ner(question)
        canonical_seen: List[str] = []
        seen_set: set = set()
        for raw in raw_surfaces:
            can = normalize_for_hash(
                raw, fold_traditional=self.linear_config.fold_traditional
            )
            if can and can not in seen_set:
                seen_set.add(can)
                canonical_seen.append(can)
        if not canonical_seen or len(self.entity_store) == 0:
            return []

        embs = self.embedding_client.encode(canonical_seen)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)

        best: Dict[str, Tuple[int, float]] = {}
        for vec in embs:
            top1 = self.entity_store.topk(vec, 1)
            if not top1:
                continue
            hid, score = top1[0]
            if hid in self._name_to_vidx and (hid not in best or score > best[hid][1]):
                best[hid] = (self._name_to_vidx[hid], float(score))
        return [(hid, vidx, sim) for hid, (vidx, sim) in best.items()]

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
        sent_emb = self.sentence_store.embeddings
        if sent_emb.shape[0] == 0:
            return entity_weights, actived
        q = question_emb if question_emb.ndim == 2 else question_emb.reshape(-1, 1)
        sentence_similarities = np.dot(sent_emb, q).flatten()

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

        passage_emb = self.passage_store.embeddings
        if passage_emb.shape[0] == 0:
            return passage_weights
        q = question_emb if question_emb.ndim == 2 else question_emb.reshape(-1, 1)
        sims = np.dot(passage_emb, q).flatten()

        # Min-max normalize so the dpr term lives on [0, 1] before being
        # combined with the log entity bonus (matches LinearRAG's
        # ``min_max_normalize`` at projects/LinearRAG/src/utils.py — when
        # all values are equal it returns ones, NOT zeros, so a single-
        # passage corpus or all-tied DPR doesn't zero out passage weights).
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
        cfg = self.config
        reset = np.where(np.isnan(node_weights) | (node_weights < 0), 0.0, node_weights)
        if reset.sum() <= 0:
            return []
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
            return []
        passage_scores = scores_arr[self._passage_vidx]
        order = np.argsort(passage_scores)[::-1]
        out: List[Tuple[str, float]] = []
        for j in order[: max(self.config.ppr_topk * 4, self.config.ppr_topk)]:
            vidx = self._passage_vidx[int(j)]
            name = self.graph.vs[vidx]["name"]
            out.append((str(name), float(passage_scores[int(j)])))
        return out

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
