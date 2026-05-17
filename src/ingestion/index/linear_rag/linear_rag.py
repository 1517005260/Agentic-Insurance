"""LinearRAG build-time graph construction (incremental-by-default).

Each call ``LinearRAG.index(passages, file_id, page_numbers)`` appends the
file's contents into the global stores and graph:

    passages (plain page Markdown — no metadata prefix)
        → embed (passage store; meta carries file_id + page_number)
        → spaCy NER per new passage (NER cache reuses existing)
        → entity_nodes / sentence_nodes / passage→entities
        → embed sentences and entities (hash dedup)
        → entity↔sentence and sentence↔entity hash-id maps
        → entity↔passage edges weighted by mention count (delta only)
        → adjacent passage edges within file_id namespace (delta only)
        → write LinearRAG.graphml

State persists across calls — graphml is loaded if it exists, NER cache is
loaded if it exists, all three faiss stores auto-load. No
``dataset_name`` / ``working_dir``: storage layout is fixed by
``config.settings``.

Passage identity (file_id, page_number) is stored as meta columns on the
passage embedding store rather than encoded into the passage text, so the
text fed to embeddings / NER / lang routing is exactly the document
content with no metadata pollution.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Set

import igraph as ig

from config.settings import (
    faiss_graph_dir,
    faiss_graph_entity_dir,
    faiss_graph_passage_dir,
    faiss_graph_sentence_dir,
)
from ingestion.index.linear_rag.disambig import (
    ACCEPTANCE_HANDLER_OVERLAY,
    ADMISSION_RULE_VERSION,
    add_alias_edges,
    cluster_shape_metrics,
    compute_clusters,
    compute_clusters_for_collapse,
    get_clusters,
    gradient_topk_candidates,
    is_composite_surface,
    invalidate_clusters,
    load_reverse_map,
    merge_topk_candidates,
    on_alias_accepted,
    propagation_policy,
    reranker_veto,
    save_reverse_map,
)
from ingestion.index.linear_rag.ner import GLiNERAdapter
from ingestion.index.linear_rag.normalize import canonical_form, normalize_for_hash
from storage import EmbeddingStore
from storage.embedding_store import get_or_create_store

import numpy as np

logger = logging.getLogger(__name__)


class LinearRAG:
    def __init__(self, global_config):
        self.config = global_config
        logger.info("Initializing LinearRAG with config: %s", self.config)

        # Three stores come from the process-wide cache so they're the
        # **same Python objects** as the lifespan-built GraphPPRChannel
        # holds. Without sharing, each ingest would load ~1.5 GB of
        # duplicate faiss / parquet — a major OOM source on 8 GB hosts.
        # Sharing is safe because: writes (add()) are serialised by
        # INGEST_LOCK; reads (PPR queries) tolerate seeing the in-memory
        # new vectors before save() flushes to disk — that matches the
        # refresh-hook contract.
        self.passage_embedding_store = get_or_create_store(
            faiss_graph_passage_dir(), namespace="passage"
        )
        self.entity_embedding_store = get_or_create_store(
            faiss_graph_entity_dir(), namespace="entity"
        )
        self.sentence_embedding_store = get_or_create_store(
            faiss_graph_sentence_dir(), namespace="sentence"
        )

        self.ner = GLiNERAdapter(
            model_id=self.config.gliner_model_id,
            labels=self.config.gliner_labels,
            threshold=self.config.gliner_threshold,
            batch_size=self.config.gliner_batch_size,
            max_span_chars=self.config.ner_max_span_chars,
            noise_labels=self.config.gliner_noise_labels,
        )

        self._ner_results_path = faiss_graph_dir() / "ner_results.json"
        self._graphml_path = faiss_graph_dir() / "LinearRAG.graphml"
        # Collapse-mode persistence — empty / absent in overlay mode, so
        # this is a zero-cost read for the default ingest path.
        self._reverse_map_path = faiss_graph_dir() / "reverse_map.json"
        self._reverse_map: dict = load_reverse_map(self._reverse_map_path)

        if self._graphml_path.exists():
            self.graph = ig.Graph.Read_GraphML(str(self._graphml_path))
            logger.info(
                "Loaded existing graph: %d vertices, %d edges",
                self.graph.vcount(),
                self.graph.ecount(),
            )
        else:
            self.graph = ig.Graph(directed=False)

    # ---------------------------------------------------------- NER caching

    # Schema version for ``ner_results.json``. Bump when the surface
    # extraction / split / cleanup pipeline changes so older caches are
    # invalidated and every passage re-runs through the new path.
    # A cache built under an older pipeline can carry sentence-shaped
    # entities or alias edges the current filters would reject, so
    # merging cached and freshly-extracted entities is unsafe; force
    # a clean rebuild on every schema change.
    NER_CACHE_VERSION = 5

    def load_existing_data(self, passage_hash_ids: Iterable[str]):
        if self._ner_results_path.exists():
            try:
                existing = json.loads(
                    self._ner_results_path.read_text(encoding="utf-8")
                )
            except Exception:
                existing = None
            cached_version = (
                existing.get("version") if isinstance(existing, dict) else None
            )
            if (
                isinstance(existing, dict)
                and cached_version == self.NER_CACHE_VERSION
                and "passage_hash_id_to_entities" in existing
            ):
                existing_passage_hash_id_to_entities = existing[
                    "passage_hash_id_to_entities"
                ]
                existing_sentence_to_entities = existing.get(
                    "sentence_to_entities", {}
                )
                existing_passage_hash_ids = set(
                    existing_passage_hash_id_to_entities.keys()
                )
                new_passage_hash_ids = (
                    set(passage_hash_ids) - existing_passage_hash_ids
                )
                return (
                    existing_passage_hash_id_to_entities,
                    existing_sentence_to_entities,
                    new_passage_hash_ids,
                )
            # Cache version mismatch (missing or older) — drop the
            # cached entities and re-run every passage through the
            # current NER pipeline. Otherwise stale composite spans
            # would survive forever.
            logger.info(
                "ner_results.json schema mismatch (cached=%s, current=%s); "
                "re-running NER on all passages",
                cached_version,
                self.NER_CACHE_VERSION,
            )
        return {}, {}, set(passage_hash_ids)

    def save_ner_results(self, passage_to_entities, sentence_to_entities):
        self._ner_results_path.parent.mkdir(parents=True, exist_ok=True)
        self._ner_results_path.write_text(
            json.dumps(
                {
                    "version": self.NER_CACHE_VERSION,
                    "passage_hash_id_to_entities": passage_to_entities,
                    "sentence_to_entities": sentence_to_entities,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def merge_ner_results(
        existing_passage_to_entities,
        existing_sentence_to_entities,
        new_passage_to_entities,
        new_sentence_to_entities,
    ):
        existing_passage_to_entities.update(new_passage_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_to_entities, existing_sentence_to_entities

    # --------------------------------------------------------------- index

    def index(
        self,
        passages: List[str],
        file_id: str,
        page_numbers: Optional[List[int]] = None,
    ) -> dict:
        """Append a file's passages into the global graph + stores.

        ``passages[i]`` is the plain page Markdown (no prefix); the
        identity ``(file_id, page_numbers[i])`` is stored as a meta column
        on the passage embedding store so the on-disk passage text matches
        the document body exactly. ``page_numbers`` defaults to
        ``[1, 2, ..., len(passages)]`` for callers that don't track them
        separately.

        Returns a dict of counts: ``{passages, entities, sentences,
        alias_edges}`` — only counts items added by THIS call (hash dedup
        means re-running on the same content yields zeros across the board).
        """
        if page_numbers is None:
            page_numbers = list(range(1, len(passages) + 1))
        if len(page_numbers) != len(passages):
            raise ValueError(
                f"page_numbers length {len(page_numbers)} != passages length "
                f"{len(passages)}"
            )

        self.node_to_node_stats = defaultdict(dict)

        # 1. Embed new passages (existing ones skip via hash dedup). The
        #    passage text is the page Markdown verbatim; (file_id,
        #    page_number) live as meta columns.
        added_passage_hashes = self._insert_with_dedup(
            self.passage_embedding_store,
            passages,
            extra_metadata={
                "file_id": [file_id] * len(passages),
                "page_number": list(page_numbers),
            },
        )
        new_passage_hash_set: Set[str] = set(added_passage_hashes)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()

        # 2. NER on the diff against the existing cache.
        (
            existing_passage_to_entities,
            existing_sentence_to_entities,
            new_passage_hash_ids,
        ) = self.load_existing_data(hash_id_to_passage.keys())

        if new_passage_hash_ids:
            new_hash_id_to_passage = {h: hash_id_to_passage[h] for h in new_passage_hash_ids}
            new_passage_to_entities, new_sentence_to_entities = self.ner.batch_ner(
                new_hash_id_to_passage, self.config.max_workers
            )
            self.merge_ner_results(
                existing_passage_to_entities,
                existing_sentence_to_entities,
                new_passage_to_entities,
                new_sentence_to_entities,
            )

        # 2b. Post-NER cleanup. Each entity surface goes through
        #     cleanup → junk filter → canonical form. Junk surfaces (LaTeX
        #     wrappers, pure numerics, percentages, durations, currency
        #     amounts, HTML fragments) are dropped; survivors get a
        #     language-aware canonical key so case / article / 繁体 / NFKC
        #     variants collapse to one entity.
        existing_passage_to_entities = self._normalize_entity_surfaces(
            existing_passage_to_entities,
            fold_traditional=self.config.fold_traditional,
            han_fragment_max_chars=self.config.junk_max_han_chars,
        )
        existing_sentence_to_entities = self._normalize_entity_surfaces(
            existing_sentence_to_entities,
            fold_traditional=self.config.fold_traditional,
            han_fragment_max_chars=self.config.junk_max_han_chars,
        )

        self.save_ner_results(existing_passage_to_entities, existing_sentence_to_entities)

        # 3. Materialize node sets and per-passage entity lists from full state.
        (
            entity_nodes,
            sentence_nodes,
            passage_hash_id_to_entities,
        ) = self._extract_nodes(existing_passage_to_entities, existing_sentence_to_entities)

        # 4. Embed sentences and entities (hash dedup → only new ones embedded).
        added_sentence_hashes = self._insert_with_dedup(
            self.sentence_embedding_store, list(sentence_nodes)
        )
        added_entity_hashes = self._insert_with_dedup(
            self.entity_embedding_store, list(entity_nodes)
        )
        new_entity_hash_set: Set[str] = set(added_entity_hashes)
        new_sentence_hash_set: Set[str] = set(added_sentence_hashes)

        # 5. Append entity↔passage edges (delta only — passages just added).
        self._add_entity_to_passage_edges(
            passage_hash_id_to_entities, restrict_passages=new_passage_hash_set
        )

        # 6. Append adjacent-passage edges within this file_id only.
        self._add_adjacent_passage_edges(file_id=file_id)

        # 7. Augment graph: add new vertices, append new edges (preserve existing).
        self._augment_graph(new_passage_hash_set, new_entity_hash_set)

        # 8. Entity disambiguation — add alias edges via dual-query
        #    (bare-surface + mention-centroid) gradient-cutoff recall,
        #    composite-surface admission gates, and a final pairwise
        #    Qwen3-Reranker veto. Physical nodes are NOT merged.
        added_alias_edges = self._add_alias_edges_for_new_entities(new_entity_hash_set)

        # 9. Literal mention backfill (KAG-style "domain mount"). NER
        #    is contextual, so the same surface gets tagged on the page that
        #    introduces it but missed on later pages that refer back; this
        #    pass closes that gap by sweeping every passage with the union
        #    of all entity surfaces and emitting entity_passage edges for
        #    word-boundary literal hits the NER pass missed. See
        #    ingestion.index.linear_rag.backfill for the rationale.
        added_backfill_edges = 0
        if self.config.literal_backfill_enabled:
            from ingestion.index.linear_rag.backfill import literal_backfill_graph

            entity_surfaces = self.entity_embedding_store.hash_id_to_text
            passage_text = self.passage_embedding_store.hash_id_to_text
            added_backfill_edges = literal_backfill_graph(
                self.graph,
                entity_surfaces,
                passage_text,
                min_surface_chars=self.config.literal_backfill_min_chars,
                multi_word_only=self.config.literal_backfill_multi_word_only,
                fold_traditional=self.config.fold_traditional,
            )

        # Cluster cache is now stale — invalidate so it recomputes lazily.
        clusters_path = faiss_graph_dir() / "clusters.json"
        if added_alias_edges:
            invalidate_clusters(clusters_path)

        # Persist reverse_map for collapse handlers; in overlay mode the
        # map stays empty and we skip the write to keep ingest output
        # bit-identical to the pre-v0.5 layout (no spurious file).
        is_collapse = self.config.acceptance_handler != ACCEPTANCE_HANDLER_OVERLAY
        if is_collapse and self._reverse_map:
            save_reverse_map(self._reverse_map_path, self._reverse_map)

        # Drop the auto-generated 'id' vertex attribute before writing —
        # igraph stashes the structural <node id="..."> as a vertex attr
        # called "id" on Read_GraphML, which then conflicts with the next
        # write. Removing it keeps the graphml round-trip warning-free.
        if "id" in self.graph.vs.attributes():
            del self.graph.vs["id"]

        self._graphml_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph.write_graphml(str(self._graphml_path))

        # Cluster shape metrics, computed against the freshly written
        # graph so the returned dict matches the on-disk state.
        if is_collapse:
            cluster_partition = compute_clusters_for_collapse(self.graph, self._reverse_map)
        else:
            cluster_partition = compute_clusters(self.graph)
        cluster_shape = cluster_shape_metrics(
            self.graph,
            cluster_partition,
            entity_store=self.entity_embedding_store,
            is_collapse=is_collapse,
        )

        logger.info(
            "index() done for file_id=%s: graph=(%d v, %d e), "
            "added passages=%d entities=%d sentences=%d alias_edges=%d backfill_edges=%d",
            file_id,
            self.graph.vcount(),
            self.graph.ecount(),
            len(new_passage_hash_set),
            len(new_entity_hash_set),
            len(new_sentence_hash_set),
            added_alias_edges,
            added_backfill_edges,
        )

        return {
            "passages": len(new_passage_hash_set),
            "entities": len(new_entity_hash_set),
            "sentences": len(new_sentence_hash_set),
            "alias_edges": added_alias_edges,
            "backfill_edges": added_backfill_edges,
            "cluster_shape": cluster_shape,
        }

    # ------------------------------------------------------ disambiguation

    def _add_alias_edges_for_new_entities(self, new_entity_hashes: Set[str]) -> int:
        """Run dual-query gradient ER for every newly added entity.

        Two parallel queries against the entity store, merged by max
        score per candidate:

        * **Bare-surface query** — the entity's own surface embedding.
          Since the store also holds bare-surface entries, this is a
          true symmetric cos sim and reliably surfaces character-level
          variants (singular/plural, abbreviation, light reordering).
        * **Mention-context centroid query** — average of up to
          ``config.centroid_max_mentions`` distinct sentence embeddings
          mentioning the entity. Provides the semantic recall path.
          Falls back to the bare-surface embedding when fewer than 2
          mentions are available (no second signal to combine).

        ``alias_min_sim`` applies uniformly to the merged list. The
        bare-surface arm is symmetric (query and store are both
        bare-surface embeddings), so a single floor is sufficient even
        when the centroid arm is asymmetric.
        """
        if not new_entity_hashes:
            return 0
        cfg = self.config
        # Build entity → set(mention sentences) over the WHOLE corpus
        # (existing + just-added). The map is recomputed each call rather
        # than maintained incrementally — at our scale (sub-10k entities)
        # the cost is negligible and it keeps the path easy to reason about.
        entity_text_to_mentions = self._collect_entity_mentions()
        total_added = 0
        for hash_id in new_entity_hashes:
            entity_text = self.entity_embedding_store.get_text(hash_id)
            # Composite-surface admission gate — entity-side. Surfaces
            # that look like multiple mentions glued together (chain
            # of products, conjunction-joined SKU codes) have a
            # mixture-centroid embedding and would pull in cleanly-
            # named neighbours, polluting the alias subgraph by
            # transitivity. Skip outbound alias generation for them
            # entirely; they remain in the graph as standalone
            # vertices and PPR can still hit them via passage edges.
            if is_composite_surface(entity_text):
                continue
            mention_sentences = entity_text_to_mentions.get(entity_text, [])
            bare_emb = self.entity_embedding_store.get_embedding(hash_id)
            bare_cands = gradient_topk_candidates(
                bare_emb,
                self.entity_embedding_store,
                k=cfg.alias_top_k,
                g=cfg.alias_gradient,
                min_sim=cfg.alias_min_sim,
                self_hash_id=hash_id,
            )
            centroid_emb = self._mention_centroid(mention_sentences)
            if centroid_emb is not None:
                centroid_cands = gradient_topk_candidates(
                    centroid_emb,
                    self.entity_embedding_store,
                    k=cfg.alias_top_k,
                    g=cfg.alias_gradient,
                    min_sim=cfg.alias_min_sim,
                    self_hash_id=hash_id,
                )
                cands = merge_topk_candidates(bare_cands, centroid_cands)
            else:
                cands = bare_cands
            # Composite-surface admission gate — candidate-side.
            # Mirror the entity-side check: never add an alias edge
            # whose **target** is composite either, otherwise a clean
            # entity could still be transitively pulled into a
            # garbage-bucket cluster through a single composite hop.
            cands = [
                c for c in cands
                if not is_composite_surface(
                    self.entity_embedding_store.get_text(c.hash_id)
                )
            ]
            # Reranker veto — final low-confidence gate that filters
            # ordered-tier false merges (option 1 vs option 2 etc) the
            # cosine-similarity path cannot tell apart. The threshold is
            # an absolute pairwise score boundary; below it we don't
            # build the edge. See disambig.reranker_veto for the AUC
            # caveat (~0.66 — veto only, not identity classification).
            if cfg.reranker_enabled and cands:
                cands = reranker_veto(
                    entity_text,
                    cands,
                    self.entity_embedding_store,
                    threshold=cfg.reranker_threshold,
                    instruction=cfg.reranker_instruction,
                )
            if not cands:
                continue
            # Decouple admission (boolean above) from propagation
            # strength (continuous). One features dict per surviving
            # candidate; w_prop is the policy's verdict on that dict.
            features_list: List[dict] = []
            w_prop_list: List[float] = []
            for c in cands:
                feats = {
                    "cos_sim": float(c.score),
                    "reranker_score": c.rerank_yes_prob,
                    "admission_rule_version": ADMISSION_RULE_VERSION,
                    "accepted_by": "gradient_er",
                }
                features_list.append(feats)
                w_prop_list.append(float(propagation_policy(feats, cfg)))
            total_added += on_alias_accepted(
                cfg.acceptance_handler,
                self.graph,
                hash_id,
                cands,
                features_list,
                w_prop_list,
                reverse_map=self._reverse_map,
            )
        return total_added

    def _collect_entity_mentions(self) -> dict:
        """Return ``entity_text → list[unique_sentence_text]`` over the cache.

        Built from the persisted ``ner_results.json`` reverse index
        ``sentence_to_entities``. Sentences are deduped per entity to keep
        the centroid from collapsing on boilerplate.
        """
        if not self._ner_results_path.exists():
            return {}
        ner = json.loads(self._ner_results_path.read_text(encoding="utf-8"))
        sentence_to_entities = ner.get("sentence_to_entities", {})
        out: dict[str, list[str]] = {}
        seen: dict[str, set[str]] = {}
        for sent, ents in sentence_to_entities.items():
            for ent in ents:
                if ent not in seen:
                    seen[ent] = set()
                    out[ent] = []
                if sent not in seen[ent]:
                    seen[ent].add(sent)
                    out[ent].append(sent)
        return out

    def _mention_centroid(
        self, mention_sentences: List[str]
    ) -> Optional[np.ndarray]:
        """L2-normalized centroid of up to N distinct mention-sentence embeddings.

        Returns ``None`` when fewer than 2 distinct mention sentences
        are available — in that case the caller relies on the
        bare-surface query alone (no semantic second signal to add).
        Centroid is L2-normalized so its cos sim against bare-surface
        store entries is in the same range as the bare-vs-bare query.
        """
        cap = max(1, int(self.config.centroid_max_mentions))
        sentence_embeddings: List[np.ndarray] = []
        for sent in mention_sentences[:cap]:
            sent_hash = self.sentence_embedding_store.text_to_hash_id.get(sent)
            if sent_hash is None:
                continue
            sentence_embeddings.append(
                self.sentence_embedding_store.get_embedding(sent_hash)
            )
        if len(sentence_embeddings) < 2:
            return None
        centroid = np.mean(np.stack(sentence_embeddings, axis=0), axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid.astype(np.float32)

    @staticmethod
    def _normalize_entity_surfaces(
        mapping,
        *,
        fold_traditional: bool = True,
        han_fragment_max_chars: int = 15,
    ):
        """Apply ``normalize_for_hash`` to every entity surface in place.

        ``mapping`` maps either passage_hash_id or sentence_text → list of
        raw entity surfaces. Junk surfaces are dropped; survivors are
        replaced with their canonical key. Duplicate canonicals after
        normalization are collapsed.

        ``han_fragment_max_chars`` controls the Chinese sentence-fragment
        cutoff used by ``is_junk``; pass the value from
        ``LinearRAGConfig.junk_max_han_chars`` so the per-domain admin
        tuning takes effect.
        """
        out = {}
        for key, ents in mapping.items():
            seen: list[str] = []
            seen_set: set[str] = set()
            for raw in ents:
                canonical = normalize_for_hash(
                    raw,
                    fold_traditional=fold_traditional,
                    han_fragment_max_chars=han_fragment_max_chars,
                )
                if canonical is None:
                    continue
                if canonical in seen_set:
                    continue
                seen_set.add(canonical)
                seen.append(canonical)
            if seen:
                out[key] = seen
        return out

    # ------------------------------------------------------------ helpers

    def _insert_with_dedup(
        self, store: EmbeddingStore, texts: List[str], extra_metadata=None
    ):
        """Wrap insert_text returning only the newly added hashes."""
        if not texts:
            return []
        before = set(store.hash_id_to_idx.keys())
        store.insert_text(
            texts,
            embedding_client=self.config.embedding_client,
            extra_metadata=extra_metadata,
        )
        after = set(store.hash_id_to_idx.keys())
        return list(after - before)

    @staticmethod
    def _extract_nodes(passage_to_entities, sentence_to_entities):
        entity_nodes: Set[str] = set()
        sentence_nodes: Set[str] = set()
        passage_hash_id_to_entities = defaultdict(set)
        for passage_hash_id, entities in passage_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        for sentence, entities in sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_nodes.add(entity)
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities

    def _add_entity_to_passage_edges(
        self, passage_hash_id_to_entities, restrict_passages: Set[str]
    ):
        """Weight edges by mention count / total mentions (per passage).

        After normalization, ``entity`` is the canonical key (lowercased
        English / Simplified-Chinese / NFKC) but ``passage_text`` is the
        original. We canonicalize the passage on-the-fly with the SAME
        function so the count works regardless of casing or 繁/简 difference.
        """
        for passage_hash_id in restrict_passages:
            entities = passage_hash_id_to_entities.get(passage_hash_id, set())
            if not entities:
                continue
            raw_text = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            search_text = canonical_form(raw_text, fold_traditional=self.config.fold_traditional)
            counts = {}
            total = 0
            for entity in entities:
                if entity not in self.entity_embedding_store.text_to_hash_id:
                    continue
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
                count = search_text.count(entity)
                if count <= 0:
                    continue
                counts[entity_hash_id] = count
                total += count
            if total == 0:
                continue
            for entity_hash_id, count in counts.items():
                self.node_to_node_stats[passage_hash_id][entity_hash_id] = (
                    count / total,
                    "entity_passage",
                )

    def _add_adjacent_passage_edges(self, file_id: str):
        """Link adjacent passages within the same file_id by page_number.

        Reads the passage store's ``file_id`` and ``page_number`` meta
        columns rather than parsing a text prefix — keeps the passage text
        clean of metadata.
        """
        hash_ids = self.passage_embedding_store.hash_ids
        file_ids = self.passage_embedding_store.meta_column("file_id")
        page_numbers = self.passage_embedding_store.meta_column("page_number")

        items: List[tuple[int, str]] = []
        for h, fid, pn in zip(hash_ids, file_ids, page_numbers):
            if fid != file_id or pn is None:
                continue
            try:
                items.append((int(pn), h))
            except (TypeError, ValueError):
                continue
        items.sort(key=lambda x: x[0])
        for i in range(len(items) - 1):
            current = items[i][1]
            nxt = items[i + 1][1]
            self.node_to_node_stats[current][nxt] = (1.0, "adjacent_passage")

    def _augment_graph(
        self, new_passage_hashes: Set[str], new_entity_hashes: Set[str]
    ):
        """Add missing vertices and append new edges (preserves existing edge weights).

        Each vertex carries ``vertex_type`` ∈ {``"passage"``, ``"entity"``} so
        downstream maintenance (orphan sweep) can distinguish a leaf passage
        from a free-floating entity.
        """
        existing_names = {v["name"] for v in self.graph.vs if "name" in v.attributes()}

        entity_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_id_to_text = self.passage_embedding_store.get_hash_id_to_text()

        for hash_id in new_passage_hashes:
            if hash_id in existing_names:
                continue
            self.graph.add_vertex(
                name=hash_id,
                content=passage_id_to_text.get(hash_id, ""),
                vertex_type="passage",
            )
            existing_names.add(hash_id)
        for hash_id in new_entity_hashes:
            if hash_id in existing_names:
                continue
            self.graph.add_vertex(
                name=hash_id,
                content=entity_id_to_text.get(hash_id, ""),
                vertex_type="entity",
            )
            existing_names.add(hash_id)

        # Append edges that don't exist yet, set per-edge weight + edge_type
        # without touching pre-existing edges' attributes.
        new_edge_pairs = []
        new_edge_weights = []
        new_edge_types = []
        for src, neighbors in self.node_to_node_stats.items():
            for dst, (weight, edge_type) in neighbors.items():
                if src == dst:
                    continue
                if not self.graph.are_connected(src, dst):
                    new_edge_pairs.append((src, dst))
                    new_edge_weights.append(weight)
                    new_edge_types.append(edge_type)
        if new_edge_pairs:
            start = self.graph.ecount()
            self.graph.add_edges(new_edge_pairs)
            # Seed the alias-side attrs even on non-alias edges so the
            # first add_alias_edges call doesn't promote those attrs to
            # the whole edge schema and leave pre-existing edges with
            # GraphML round-tripped ``None``. ``w_prop`` mirrors ``weight``
            # under policy=cos; ``features_json`` is an empty record.
            for offset, (w, t) in enumerate(zip(new_edge_weights, new_edge_types)):
                eidx = start + offset
                self.graph.es[eidx]["weight"] = w
                self.graph.es[eidx]["edge_type"] = t
                self.graph.es[eidx]["w_prop"] = float(w)
                self.graph.es[eidx]["features_json"] = ""


