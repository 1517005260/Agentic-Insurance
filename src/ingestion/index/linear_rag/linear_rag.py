"""LinearRAG build-time graph construction (incremental-by-default).

Each call ``LinearRAG.index(passages, file_id, page_numbers)`` appends the
file's contents into the global stores and graph:

    passages (plain page Markdown — no metadata prefix)
        → embed (passage store; meta carries file_id + page_number)
        → GLiNER NER per new passage (NER cache reuses existing)
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
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Sequence, Set

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
    AliasCandidate,
    cluster_shape_metrics,
    compute_clusters,
    compute_clusters_for_collapse,
    get_clusters,
    gradient_topk_candidates,
    idf_weighted_overlap,
    is_composite_surface,
    invalidate_clusters,
    load_reverse_map,
    mutual_knn_pairs,
    on_alias_accepted,
    propagation_policy,
    save_reverse_map,
    smoothed_idf,
    tokenize_surface,
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
            calibration_enabled=self.config.gliner_calibration_enabled,
            temperature=self.config.gliner_temperature,
            label_thresholds=self.config.gliner_label_thresholds,
            stopword_languages=self.config.gliner_stopword_languages,
            stopword_confidence_floor=self.config.gliner_stopword_confidence_floor,
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

        # Per-instance index() call counter for graphml-flush cadence.
        # Default cadence = 1 (write every call), so the per-file
        # production builder writes every doc. A persistent bulk driver
        # (one instance over many
        # docs) sets graphml_flush_every > 1 so the O(V+E) graphml
        # (de)serialisation is amortised instead of paid every doc.
        self._index_calls = 0

        # NER cache held in memory across index() calls. On __init__ we
        # load any prior on-disk cache once; on every subsequent doc we
        # only mutate the in-memory dicts (normalize the delta, merge,
        # update the mention reverse index). save_ner_results runs only
        # on the flush cadence (alongside graphml). This eliminates the
        # per-doc 38 MB JSON write and the O(N) re-normalisation /
        # re-load of already-processed entries that was the dominant
        # per-doc cost in the bulk-build profile.
        (
            self._passage_to_entities,
            self._sentence_to_entities,
            self._passage_to_sentences,
        ) = self._load_ner_cache_or_empty()
        self._mentions_cache: dict[str, list[str]] = {}
        self._mention_seen: dict[str, set[str]] = {}
        self._rebuild_mention_index(self._sentence_to_entities)

        # Incremental ER caches (mirror the mentions-cache pattern so the
        # flush-time _resolve_entities batch is O(|pending|), not O(store) per
        # flush — which would make a cadenced / per-file build O(N²)):
        #   _token_df / _idf_surface_count — per-token corpus document
        #     frequency, the IDF source for the lexical gate; bumped as
        #     entities are inserted (never recomputed over the whole store).
        #   _name_to_vidx — entity/passage name → igraph vertex index, so the
        #     batch never rebuilds the O(V) vertex map per flush; kept current
        #     by _augment_graph as vertices are added.
        self._token_df: dict[str, int] = {}
        self._idf_surface_count = 0
        self._name_to_vidx: dict[str, int] = {
            v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()
        }
        self._warm_token_df()

        # Dirty flags for deferred persistence + literal backfill.
        # _ner_dirty: ner_results.json has unflushed mutations.
        # _backfill_pending_new_entities / _backfill_pending_new_passages:
        # entities / passages added since the last backfill run. The
        # literal-backfill pass at flush time only needs to scan
        # (new_passages × all_entities ∪ all_passages × new_entities) to
        # be complete — old × old pairs were processed by a prior flush.
        self._ner_dirty = False
        self._backfill_pending_new_entities: Set[str] = set()
        self._backfill_pending_new_passages: Set[str] = set()
        # Entities added since the last entity-resolution batch. ER is a
        # flush-time batch (``_resolve_entities``), not a per-document step —
        # this is the delta it resolves so the pass stays O(N) per flush
        # instead of re-resolving the whole store every document.
        self._er_pending_entities: Set[str] = set()

        # Cached cluster_shape returned from the most recent on-cadence
        # compute. Off-cadence index() calls reuse this snapshot instead
        # of re-running compute_clusters (which is O(E_alias) even for
        # the connected_components partitioner, and Leiden is much
        # heavier). It is a derived, monitoring-only view so being a few
        # docs stale is acceptable.
        self._cached_cluster_shape: Optional[dict] = None

    def flush_graphml(self) -> None:
        """Atomically persist the graph to graphml (tmp + os.replace).

        Atomic rename also removes the torn-file risk of the old
        in-place hundreds-of-MB write (a kill mid-write previously
        corrupted the whole graph with no backup).
        """
        if "id" in self.graph.vs.attributes():
            del self.graph.vs["id"]
        self._graphml_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._graphml_path.with_suffix(".graphml.tmp")
        self.graph.write_graphml(str(tmp))
        os.replace(tmp, self._graphml_path)

    def _save_ner_config(self) -> None:
        """Persist the GLiNER knobs this KG was built with, so query-time NER
        (GraphPPRChannel) follows ingest rather than a runtime default. Each
        dataset (insurance / Double-Bench / multi-hop) fixes its own labels at
        ingest; the runtime channel reloads them from here so the two never
        drift. Tiny, written every flush, independent of the NER cache version
        (a label change must always be reflected even when the cache is warm)."""
        cfg = self.config
        payload = {
            "gliner_model_id": cfg.gliner_model_id,
            "gliner_labels": list(cfg.gliner_labels),
            "gliner_noise_labels": list(cfg.gliner_noise_labels),
            "gliner_threshold": cfg.gliner_threshold,
            "gliner_batch_size": cfg.gliner_batch_size,
            "ner_max_span_chars": cfg.ner_max_span_chars,
            "gliner_calibration_enabled": cfg.gliner_calibration_enabled,
            "gliner_temperature": cfg.gliner_temperature,
            "gliner_label_thresholds": cfg.gliner_label_thresholds,
            "gliner_stopword_languages": cfg.gliner_stopword_languages,
            "gliner_stopword_confidence_floor": cfg.gliner_stopword_confidence_floor,
        }
        path = faiss_graph_dir() / "ner_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def flush_all(self) -> int:
        """Run all deferred persistence + literal backfill, then graphml.

        The single sync point that drains everything index() defers:

        1. Three faiss/parquet stores — save() is a no-op when not
           dirty so the per-file API path (graphml_flush_every=1, fresh
           store per call) is unaffected and a bulk path with cadence>1
           amortises one full ~735 MB rewrite across many docs.
        2. NER results JSON — same dirty-flag short-circuit.
        3. Literal-substring backfill over only the pending delta
           (new_passages × all_entities ∪ all_passages × new_entities).
           Old × old pairs were covered by a prior flush so this is a
           complete cover — see backfill.py for the safety argument.
        4. graphml.

        Returns the number of backfill edges added (callers that want
        per-flush accounting can log this).
        """
        self.passage_embedding_store.save()
        self.sentence_embedding_store.save()
        self.entity_embedding_store.save()
        if self._ner_dirty:
            self._save_ner_results()

        added_backfill_edges = 0
        if self.config.literal_backfill_enabled and (
            self._backfill_pending_new_passages
            or self._backfill_pending_new_entities
        ):
            added_backfill_edges = self._run_pending_backfill()

        # Entity resolution batch — blocking → mutual-kNN → precision gate →
        # alias edges, over the entities added since the last flush. Run AFTER
        # backfill so the co-occurrence veto sees the complete entity_passage
        # edge set. One pass per flush keeps every stage O(N); see
        # ``_resolve_entities``. No-op when alias edges are disabled.
        is_collapse = self.config.acceptance_handler != ACCEPTANCE_HANDLER_OVERLAY
        if self.config.alias_edges_enabled and self._er_pending_entities:
            added_alias = self._resolve_entities(self._er_pending_entities)
            self._er_pending_entities.clear()
            if added_alias:
                invalidate_clusters(faiss_graph_dir() / "clusters.json")
            if is_collapse and self._reverse_map:
                save_reverse_map(self._reverse_map_path, self._reverse_map)

        # Refresh cluster_shape against the post-ER graph so the final-flush
        # monitoring snapshot is current rather than carrying whatever stale
        # value the cadenced index() loop left behind. Cheap path (cheap=True).
        if is_collapse:
            partition = compute_clusters_for_collapse(self.graph, self._reverse_map)
        else:
            partition = compute_clusters(
                self.graph,
                algorithm=self.config.cluster_algorithm,
                leiden_resolution=self.config.cluster_leiden_resolution,
                leiden_weighted=self.config.cluster_leiden_weighted,
            )
        self._cached_cluster_shape = cluster_shape_metrics(
            self.graph,
            partition,
            entity_store=self.entity_embedding_store,
            is_collapse=is_collapse,
            cheap=True,
        )

        self.flush_graphml()
        self._save_ner_config()
        self._save_er_config()
        return added_backfill_edges

    def _run_pending_backfill(self) -> int:
        """Run literal_backfill restricted to the pending delta.

        Two passes cover every new (entity, passage) pair without
        re-scanning old × old (which a previous flush handled):

        * pass A: gazetteer = ALL entities, target = pending NEW passages
        * pass B: gazetteer = pending NEW entities, target = ALL passages

        Pairs in (NEW × NEW) are touched twice but the backfill function
        dedups against the existing edge set, so the second hit is a
        cheap skip. Pending sets are cleared on success.
        """
        from ingestion.index.linear_rag.backfill import literal_backfill_graph

        all_entities = self.entity_embedding_store.hash_id_to_text
        all_passages = self.passage_embedding_store.hash_id_to_text
        cfg = self.config

        added = 0

        if self._backfill_pending_new_passages:
            new_passage_text = {
                h: all_passages[h]
                for h in self._backfill_pending_new_passages
                if h in all_passages
            }
            added += literal_backfill_graph(
                self.graph,
                all_entities,
                new_passage_text,
                min_surface_chars=cfg.literal_backfill_min_chars,
                multi_word_only=cfg.literal_backfill_multi_word_only,
                fold_traditional=cfg.fold_traditional,
            )

        if self._backfill_pending_new_entities:
            new_entity_surfaces = {
                h: all_entities[h]
                for h in self._backfill_pending_new_entities
                if h in all_entities
            }
            added += literal_backfill_graph(
                self.graph,
                new_entity_surfaces,
                all_passages,
                min_surface_chars=cfg.literal_backfill_min_chars,
                multi_word_only=cfg.literal_backfill_multi_word_only,
                fold_traditional=cfg.fold_traditional,
            )

        self._backfill_pending_new_passages.clear()
        self._backfill_pending_new_entities.clear()
        return added

    # ---------------------------------------------------------- NER caching

    # Schema version for ``ner_results.json``. Bump when the surface
    # extraction / split / cleanup pipeline changes so older caches are
    # invalidated and every passage re-runs through the new path.
    # A cache built under an older pipeline can carry sentence-shaped
    # entities or alias edges the current filters would reject, so
    # merging cached and freshly-extracted entities is unsafe; force
    # a clean rebuild on every schema change.
    NER_CACHE_VERSION = 6

    def _load_ner_cache_or_empty(self) -> tuple[dict, dict, dict]:
        """Load (passage_to_entities, sentence_to_entities,
        passage_to_sentences) from disk once.

        Returns empty dicts on missing file, parse error, or schema
        version mismatch. A version mismatch silently drops the cache
        so every passage re-runs through the current NER pipeline on
        the first index() call (which sees its hash absent from the
        in-memory dict and therefore "new"); otherwise stale composite
        spans would survive forever.
        """
        if not self._ner_results_path.exists():
            return {}, {}, {}
        try:
            payload = json.loads(self._ner_results_path.read_text(encoding="utf-8"))
        except Exception:
            return {}, {}, {}
        if not isinstance(payload, dict):
            return {}, {}, {}
        if payload.get("version") != self.NER_CACHE_VERSION:
            logger.info(
                "ner_results.json schema mismatch (cached=%s, current=%s); "
                "treating cache as empty — every passage will re-run NER",
                payload.get("version"),
                self.NER_CACHE_VERSION,
            )
            return {}, {}, {}
        return (
            payload.get("passage_hash_id_to_entities", {}) or {},
            payload.get("sentence_to_entities", {}) or {},
            payload.get("passage_hash_id_to_sentences", {}) or {},
        )

    def _save_ner_results(self) -> None:
        """Persist the in-memory NER cache to disk. Caller must check
        ``self._ner_dirty`` to avoid no-op writes."""
        self._ner_results_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._ner_results_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "version": self.NER_CACHE_VERSION,
                    "passage_hash_id_to_entities": self._passage_to_entities,
                    "sentence_to_entities": self._sentence_to_entities,
                    "passage_hash_id_to_sentences": self._passage_to_sentences,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, self._ner_results_path)
        self._ner_dirty = False

    def _rebuild_mention_index(self, sentence_to_entities) -> None:
        """Populate ``_mentions_cache`` (entity → unique mention sentences)
        from a sentence→entities dict, in O(Σ|entities_per_sentence|).

        Called once on __init__ to warm the cache from disk, then never
        again — subsequent updates use ``_update_mention_index`` on the
        per-doc delta only.
        """
        for sent, ents in sentence_to_entities.items():
            for ent in ents:
                seen = self._mention_seen.setdefault(ent, set())
                if sent not in seen:
                    seen.add(sent)
                    self._mentions_cache.setdefault(ent, []).append(sent)

    def _update_mention_index(self, new_sentence_to_entities) -> None:
        """Append per-doc new sentence→entity links into the cached
        mention reverse index, deduping per entity."""
        for sent, ents in new_sentence_to_entities.items():
            for ent in ents:
                seen = self._mention_seen.setdefault(ent, set())
                if sent not in seen:
                    seen.add(sent)
                    self._mentions_cache.setdefault(ent, []).append(sent)

    def _warm_token_df(self) -> None:
        """One-time warm of the token document-frequency table from any entity
        surfaces already in the store (mirrors the mentions-cache warm)."""
        han_ngram = int(self.config.er_han_token_ngram)
        for surface in self.entity_embedding_store.get_hash_id_to_text().values():
            self._idf_surface_count += 1
            for t in set(tokenize_surface(surface, han_ngram=han_ngram)):
                self._token_df[t] = self._token_df.get(t, 0) + 1

    def _update_token_df(self, new_entity_hashes) -> None:
        """Bump the token document-frequency table for newly added entity
        surfaces (each unique surface = one document). O(|new|·tokens) — uses
        per-hash ``get_text`` so it never copies the whole store map."""
        if not new_entity_hashes:
            return
        han_ngram = int(self.config.er_han_token_ngram)
        store = self.entity_embedding_store
        for h in new_entity_hashes:
            surface = store.get_text(h)
            if not surface:
                continue
            self._idf_surface_count += 1
            for t in set(tokenize_surface(surface, han_ngram=han_ngram)):
                self._token_df[t] = self._token_df.get(t, 0) + 1

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

        # 2. NER on the diff against the in-memory cache. The cache is
        #    loaded once in __init__ and persisted on flush cadence; per-
        #    doc reads are O(1) attribute access.
        new_passage_hash_ids = (
            set(hash_id_to_passage.keys()) - set(self._passage_to_entities.keys())
        )

        if new_passage_hash_ids:
            new_hash_id_to_passage = {h: hash_id_to_passage[h] for h in new_passage_hash_ids}
            (
                new_passage_to_entities,
                new_sentence_to_entities,
                new_passage_to_sentences,
            ) = self.ner.batch_ner(
                new_hash_id_to_passage, self.config.max_workers
            )
            # 2b. Post-NER cleanup, applied ONLY to the per-doc delta.
            #     Each entity surface goes through cleanup → junk filter
            #     → canonical form (LaTeX wrappers, pure numerics, etc.
            #     dropped; survivors collapsed to a language-aware
            #     canonical key). Restricting cleanup to the delta avoids
            #     re-normalising cached entries, which were normalised
            #     when they were the delta on their own ingest call (a
            #     version-bump forces a full rebuild via
            #     _load_ner_cache_or_empty when the normalisation rules
            #     change).
            new_passage_to_entities_norm = self._normalize_entity_surfaces(
                new_passage_to_entities,
                fold_traditional=self.config.fold_traditional,
                han_fragment_max_chars=self.config.junk_max_han_chars,
            )
            new_sentence_to_entities = self._normalize_entity_surfaces(
                new_sentence_to_entities,
                fold_traditional=self.config.fold_traditional,
                han_fragment_max_chars=self.config.junk_max_han_chars,
            )
            # Record EVERY processed passage hash, including those whose
            # NER produced no entities (or whose entities were all
            # dropped by the junk filter / normalization). Without this,
            # the next index() call's diff
            # ``set(hash_ids) - self._passage_to_entities.keys()`` would
            # re-flag empty-result passages as "new" and pay the full
            # GLiNER cost again every doc.
            for h in new_passage_hash_ids:
                self._passage_to_entities[h] = new_passage_to_entities_norm.get(h, [])
                # Record sentence list (even when empty) so the diff
                # against the cache stays consistent on re-runs.
                self._passage_to_sentences[h] = new_passage_to_sentences.get(h, [])
            self._sentence_to_entities.update(new_sentence_to_entities)
            self._update_mention_index(new_sentence_to_entities)
            self._ner_dirty = True

        # 3. Materialize node sets and per-passage entity lists from full state.
        (
            entity_nodes,
            sentence_nodes,
            passage_hash_id_to_entities,
        ) = self._extract_nodes(
            self._passage_to_entities,
            self._sentence_to_entities,
            self._passage_to_sentences,
        )

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
        if self.config.adjacent_passage_edges_enabled:
            self._add_adjacent_passage_edges(file_id=file_id)

        # 7. Augment graph: add new vertices, append new edges (preserve existing).
        self._augment_graph(new_passage_hash_set, new_entity_hash_set)

        # 8. Entity resolution is deferred to flush_all() — it is a flush-time
        #    BATCH over the entities added since the last flush, not a per-doc
        #    step. Running it per document re-resolved against the whole
        #    accumulated store every call; batching keeps the pass O(N) and
        #    lets mutual-kNN see every candidate's neighbourhood at once.
        if self.config.alias_edges_enabled:
            self._er_pending_entities |= new_entity_hash_set
            self._update_token_df(new_entity_hash_set)
        added_alias_edges = 0

        # 9. Literal mention backfill is deferred to flush_all(). NER is
        #    contextual (same surface tagged on intro page, missed on
        #    later reference pages) so the pass remains necessary; but
        #    its cost is O(N_entities × N_passages) per invocation, and
        #    paying it per doc made a 650-doc build O(N²). At flush
        #    cadence we run it once over only the pending delta
        #    (new_passages × all_entities ∪ all_passages × new_entities
        #    is a complete cover — old × old pairs were handled by a
        #    prior flush). Per-doc accounting reports 0; the flush
        #    summary carries the real count.
        if self.config.literal_backfill_enabled:
            self._backfill_pending_new_passages |= new_passage_hash_set
            self._backfill_pending_new_entities |= new_entity_hash_set
        added_backfill_edges = 0

        # Alias edges + reverse_map + clusters.json are all produced by the
        # flush-time ER batch (see flush_all), so nothing alias-related is
        # touched per document here.
        is_collapse = self.config.acceptance_handler != ACCEPTANCE_HANDLER_OVERLAY

        # Cluster shape (a derived, monitoring-only view). On cadence
        # we recompute against the live graph + cache the result; off
        # cadence we reuse the cached snapshot. Collapse mode always
        # recomputes — it walks the reverse_map, not the alias subgraph.
        # Recomputing off cadence would run
        # compute_clusters(connected_components), which is O(E) per doc
        # (alias-edge filter + subgraph extract + union-find) — paying
        # that per doc is its own O(N²) tail. Cached reuse drops it to
        # a dict reference.
        self._index_calls += 1
        _cl_every = max(1, int(getattr(self.config, "cluster_shape_every", 1)))
        _cluster_on_cadence = self._index_calls % _cl_every == 0
        if is_collapse:
            cluster_partition = compute_clusters_for_collapse(
                self.graph, self._reverse_map
            )
            cluster_shape = cluster_shape_metrics(
                self.graph,
                cluster_partition,
                entity_store=self.entity_embedding_store,
                is_collapse=is_collapse,
                cheap=True,
            )
            self._cached_cluster_shape = cluster_shape
        elif _cluster_on_cadence or self._cached_cluster_shape is None:
            cluster_partition = compute_clusters(
                self.graph,
                algorithm=self.config.cluster_algorithm,
                leiden_resolution=self.config.cluster_leiden_resolution,
                leiden_weighted=self.config.cluster_leiden_weighted,
            )
            cluster_shape = cluster_shape_metrics(
                self.graph,
                cluster_partition,
                entity_store=self.entity_embedding_store,
                is_collapse=is_collapse,
                cheap=True,
            )
            self._cached_cluster_shape = cluster_shape
        else:
            cluster_shape = self._cached_cluster_shape

        # Persist on a cadence. Default graphml_flush_every=1 → flush
        # every index() call (bit-identical to before; the per-file API
        # builder constructs a fresh instance per file so its counter is
        # always 1). A persistent bulk driver sets it >1 and force-
        # calls flush_all() at checkpoints / end, so the O(V+E) graphml
        # round-trip, the 3-store faiss+parquet rewrite, the NER JSON
        # save, and the literal backfill are all amortised together
        # instead of paid every doc. On cadence hits we capture the
        # backfill edge count so the per-doc return dict stays accurate
        # for the API path (graphml_flush_every=1 → every doc gets its
        # backfill count, matching pre-deferral semantics).
        _every = max(1, int(getattr(self.config, "graphml_flush_every", 1)))
        if self._index_calls % _every == 0:
            added_backfill_edges = self.flush_all()

        logger.info(
            "index() done for file_id=%s: graph=(%d v, %d e), "
            "added passages=%d entities=%d sentences=%d alias_edges=%d",
            file_id,
            self.graph.vcount(),
            self.graph.ecount(),
            len(new_passage_hash_set),
            len(new_entity_hash_set),
            len(new_sentence_hash_set),
            added_alias_edges,
        )

        return {
            "passages": len(new_passage_hash_set),
            "entities": len(new_entity_hash_set),
            "sentences": len(new_sentence_hash_set),
            "alias_edges": added_alias_edges,
            "backfill_edges": added_backfill_edges,
            "cluster_shape": cluster_shape,
        }

    def bulk_index_passages(
        self,
        items: Sequence,
        *,
        final_flush: bool = True,
    ) -> dict:
        """Index many passages — each its own ``(file_id, page_number)`` — in ONE
        pass. The O(N) build path for corpora of many tiny documents (e.g.
        cross-document multi-hop wiki: 6k-12k single-passage file_ids).

        Calling :meth:`index` once per passage is O(N²): each call re-materialises
        node sets from the full accumulated NER cache (``_extract_nodes``) and
        re-hashes the full sentence/entity sets (``_insert_with_dedup``). Here
        every stage runs once over the whole batch: passage embed (1 dedup call),
        NER (1 batched ``batch_ner``), node extraction (delta-only via
        ``_extract_nodes_for_passages``), sentence/entity embed (1 each),
        entity→passage edges, adjacency (1 grouped scan, gated), graph augment (1
        call), alias ER (1 call, gated), and a single final ``flush_all``.

        ``items`` = sequence of ``(file_id, page_number, text)``; ``text`` is the
        plain passage markdown (no metadata prefix). Per-dataset knobs
        (``alias_edges_enabled`` / ``adjacent_passage_edges_enabled`` /
        ``literal_backfill_enabled``) come from ``self.config``. ``final_flush``
        drains the deferred backfill + graphml + ner_config once; pass False to
        chain several bulk calls before a manual ``flush_all``.
        """
        rows = [
            (str(f), (0 if p is None else int(p)), t)
            for f, p, t in items
            if t and t.strip()
        ]
        if not rows:
            return {"passages": 0, "entities": 0, "sentences": 0,
                    "alias_edges": 0, "backfill_edges": 0}
        self.node_to_node_stats = defaultdict(dict)

        texts = [t for _, _, t in rows]
        file_ids = [f for f, _, _ in rows]
        page_numbers = [p for _, p, _ in rows]
        input_hashes = {self.passage_embedding_store.hash_for(t) for t in texts}

        # 1. Embed all passages once (hash dedup); identity in meta columns.
        added_passage_hashes = self._insert_with_dedup(
            self.passage_embedding_store,
            texts,
            extra_metadata={"file_id": file_ids, "page_number": page_numbers},
        )
        new_passage_hash_set: Set[str] = set(added_passage_hashes)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()

        # 2. NER the passages not already cached — ONE batched call.
        missing = input_hashes - set(self._passage_to_entities.keys())
        if missing:
            batch = {h: hash_id_to_passage[h] for h in missing if h in hash_id_to_passage}
            new_p2e, new_s2e, new_p2s = self.ner.batch_ner(batch, self.config.max_workers)
            new_p2e = self._normalize_entity_surfaces(
                new_p2e,
                fold_traditional=self.config.fold_traditional,
                han_fragment_max_chars=self.config.junk_max_han_chars,
            )
            new_s2e = self._normalize_entity_surfaces(
                new_s2e,
                fold_traditional=self.config.fold_traditional,
                han_fragment_max_chars=self.config.junk_max_han_chars,
            )
            for h in missing:
                self._passage_to_entities[h] = new_p2e.get(h, [])
                self._passage_to_sentences[h] = new_p2s.get(h, [])
            self._sentence_to_entities.update(new_s2e)
            self._update_mention_index(new_s2e)
            self._ner_dirty = True

        # 3. Node sets for THIS batch only (delta), then embed once each.
        entity_nodes, sentence_nodes, passage_hash_id_to_entities = (
            self._extract_nodes_for_passages(input_hashes)
        )
        added_sentence_hashes = self._insert_with_dedup(
            self.sentence_embedding_store, list(sentence_nodes)
        )
        added_entity_hashes = self._insert_with_dedup(
            self.entity_embedding_store, list(entity_nodes)
        )
        new_entity_hash_set: Set[str] = set(added_entity_hashes)

        # 4. Edges + graph augment (each once over the batch).
        self._add_entity_to_passage_edges(
            passage_hash_id_to_entities, restrict_passages=new_passage_hash_set
        )
        if self.config.adjacent_passage_edges_enabled:
            self._add_adjacent_passage_edges_for_file_ids(set(file_ids))
        self._augment_graph(new_passage_hash_set, new_entity_hash_set)

        # 5. Defer entity resolution to flush_all() — one batch over all
        #    pending entities (no-op when alias_edges_enabled is False).
        if self.config.alias_edges_enabled:
            self._er_pending_entities |= new_entity_hash_set
            self._update_token_df(new_entity_hash_set)
        added_alias_edges = 0

        # 6. Defer backfill to the final flush.
        if self.config.literal_backfill_enabled:
            self._backfill_pending_new_passages |= new_passage_hash_set
            self._backfill_pending_new_entities |= new_entity_hash_set

        self._index_calls += 1
        added_backfill_edges = self.flush_all() if final_flush else 0

        logger.info(
            "bulk_index_passages done: graph=(%d v, %d e), added passages=%d "
            "entities=%d sentences=%d alias_edges=%d",
            self.graph.vcount(),
            self.graph.ecount(),
            len(new_passage_hash_set),
            len(new_entity_hash_set),
            len(set(added_sentence_hashes)),
            added_alias_edges,
        )
        return {
            "passages": len(new_passage_hash_set),
            "entities": len(new_entity_hash_set),
            "sentences": len(set(added_sentence_hashes)),
            "alias_edges": added_alias_edges,
            "backfill_edges": added_backfill_edges,
        }

    # ------------------------------------------------------ disambiguation

    def _resolve_entities(self, pending: Set[str]) -> int:
        """Flush-time batch entity resolution: blocking → mutual-kNN →
        precision gate → outdegree-capped alias edges.

        ``pending`` = entity hash_ids added since the last flush. For a bulk
        single-flush build that is every entity, so mutual-kNN is exact over
        the whole store; in incremental use it is the per-flush delta and we
        additionally recall each candidate's neighbourhood so the reciprocity
        test stays exact without rescanning the store. Every stage is
        O(|pending|·k) — no O(N²) per-document rescan.

        Recall and precision use DIFFERENT signal classes (record linkage):
        embedding ANN recalls candidates; admission is decided by an IDF
        lexical-overlap gate (for surface-similar pairs) plus a co-occurrence
        must-not-link veto. Physical nodes are never merged (overlay); collapse
        handlers still route through ``on_alias_accepted``.

        The gate decisions (veto / lexical / context) use a single snapshot of
        ``passages`` taken before any acceptance. In OVERLAY mode (default) this
        is exact — adding alias edges never moves passage membership. In COLLAPSE
        mode (the destructive DEG-RAG-style foil tier) an early collapse can
        redirect a passage onto a canonical, so a later pair's veto reads
        pre-collapse membership; collapse is therefore intentionally slightly
        more aggressive within a batch — acceptable for a baseline we compare
        against, not our shipped overlay path.
        """
        cfg = self.config
        store = self.entity_embedding_store
        # O(1) membership per hash (``hash_id_to_idx`` is a property that COPIES
        # the whole map — never touch it on the per-flush path).
        pending = {h for h in pending if store.has(h)}
        if not pending or len(store) == 0:
            return 0

        # --- Stage 0: per-batch handles (all incremental/cached — never
        #     O(store) per flush, which would make a cadenced build O(N²)) ---
        name_to_idx = self._name_to_vidx          # maintained in _augment_graph
        mentions = self._mentions_cache
        han_ngram = int(cfg.er_han_token_ngram)
        n_surfaces = max(1, self._idf_surface_count)
        token_df = self._token_df                  # corpus df, maintained on insert

        text_of: dict[str, str] = {}

        def _text(h: str) -> str:
            t = text_of.get(h)
            if t is None:
                t = store.get_text(h) or ""        # O(1) per hash, no full-map copy
                text_of[h] = t
            return t

        # --- Stage A: dual-query recall for pending + their candidates ---
        bare: dict[str, dict[str, float]] = {}
        ctx: dict[str, dict[str, float]] = {}
        neighbors: dict[str, Set[str]] = {}

        def _recall(h: str) -> None:
            if h in neighbors:
                return
            text = _text(h)
            # Composite surfaces (glued mention chains) have mixture-centroid
            # embeddings that pull in clean neighbours — never an alias source
            # or target.
            if is_composite_surface(text):
                bare[h], ctx[h], neighbors[h] = {}, {}, set()
                return
            b = {
                c.hash_id: c.score
                for c in gradient_topk_candidates(
                    store.get_embedding(h), store, k=cfg.alias_top_k,
                    g=cfg.alias_gradient, min_sim=cfg.alias_min_sim, self_hash_id=h,
                )
                if not is_composite_surface(_text(c.hash_id))
            }
            c_map: dict[str, float] = {}
            centroid = self._mention_centroid(mentions.get(text, []))
            if centroid is not None:
                c_map = {
                    c.hash_id: c.score
                    for c in gradient_topk_candidates(
                        centroid, store, k=cfg.alias_top_k, g=cfg.alias_gradient,
                        min_sim=cfg.alias_min_sim, self_hash_id=h,
                    )
                    if not is_composite_surface(_text(c.hash_id))
                }
            bare[h], ctx[h] = b, c_map
            neighbors[h] = set(b) | set(c_map)

        for h in pending:
            _recall(h)
        for h in list(pending):
            for c in neighbors[h]:
                _recall(c)  # candidate neighbourhood → exact reciprocity

        # Co-occurrence passage sets ONLY for involved entities (pending ∪ their
        # candidates) — scoped to O(|involved|·deg), never an O(E) full scan.
        involved = set(neighbors)
        passages = self._entity_passage_sets(
            [name_to_idx[h] for h in involved if h in name_to_idx]
        )

        # --- Stage B: mutual-kNN (reciprocal pairs touching pending) ---
        if cfg.er_mutual_knn:
            all_pairs = mutual_knn_pairs(neighbors)
        else:
            all_pairs = {
                frozenset((h, c)) for h, nbrs in neighbors.items() for c in nbrs
            }
        pairs = [p for p in all_pairs if p & pending]

        # --- Stage C: precision gate + co-occurrence veto ---
        # Lexical-overlap value routes the decision (NOT a cos_bare regime
        # split, which leaves a hole where a high-bare-cos zero-overlap
        # abbreviation can reach neither accept path):
        #   lex >= τ_lex          → true surface variant            → accept
        #   0 < lex < τ_lex       → shares only template/common tokens, distinct
        #                           head tokens = template collision → reject
        #                           (do NOT fall through to context: template
        #                            collisions have similar contexts too)
        #   lex == 0              → no shared tokens (cross-surface synonym like
        #                           US/USA, or unrelated)            → arbitrate
        #                           by mention-context cosine
        idf: dict[str, float] = {}
        tokens: dict[str, list] = {}

        def _tok(h: str) -> list:
            t = tokens.get(h)
            if t is None:
                t = tokenize_surface(_text(h), han_ngram=han_ngram)
                tokens[h] = t
                for tk in t:
                    if tk not in idf:
                        idf[tk] = smoothed_idf(token_df.get(tk, 0), n_surfaces)
            return t

        accepted: dict[str, list] = defaultdict(list)  # src -> [(other, score, feat)]
        veto_cap = cfg.er_max_df_for_veto
        for pair in pairs:
            a, b = tuple(pair)
            if cfg.er_cooccur_veto:
                pa, pb = passages.get(a), passages.get(b)
                # Skip the veto for hubs (df above the absolute cap) — they are
                # handled by mutual-kNN + outdegree cap + Leiden; this keeps the
                # intersection work O(cap) per pair.
                if pa and pb and len(pa) <= veto_cap and len(pb) <= veto_cap:
                    small, large = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
                    if any(p in large for p in small):
                        continue  # share a passage → distinct co-occurring entities
            cos_bare = max(bare.get(a, {}).get(b, 0.0), bare.get(b, {}).get(a, 0.0))
            cos_ctx = max(ctx.get(a, {}).get(b, 0.0), ctx.get(b, {}).get(a, 0.0))
            score = max(cos_bare, cos_ctx)
            lex = idf_weighted_overlap(_tok(a), _tok(b), idf)
            if lex >= cfg.er_idf_lex_threshold:
                feat = {
                    "cos_sim": float(score), "cos_bare": float(cos_bare),
                    "idf_overlap": float(lex), "arm": "lexical",
                    "admission_rule_version": ADMISSION_RULE_VERSION,
                    "accepted_by": "idf_mutualknn_er",
                }
            elif lex > 0.0:
                continue  # partial template overlap → template collision
            else:
                if cos_ctx < cfg.er_ctx_synonym_threshold:
                    continue
                feat = {
                    "cos_sim": float(score), "cos_ctx": float(cos_ctx),
                    "idf_overlap": 0.0, "arm": "centroid",
                    "admission_rule_version": ADMISSION_RULE_VERSION,
                    "accepted_by": "idf_mutualknn_er",
                }
            accepted[a].append((b, score, feat))
            accepted[b].append((a, score, feat))

        # --- Stage D: outdegree cap (symmetric top-L) + build edges ---
        # The cap bounds TOTAL per-entity alias degree (percolation bound must
        # hold across flushes), so each entity's budget starts from its
        # already-present incident alias edges, not from zero.
        cap = max(1, cfg.er_max_alias_degree)
        existing_deg = self._existing_alias_degree(set(accepted))
        topl: dict[str, Set[str]] = {}
        for h, lst in accepted.items():
            budget = cap - existing_deg.get(h, 0)
            if budget <= 0:
                topl[h] = set()
                continue
            lst.sort(key=lambda t: t[1], reverse=True)
            topl[h] = {c for c, _, _ in lst[:budget]}
        kept: dict[frozenset, tuple] = {}
        for h, lst in accepted.items():
            for c, sc, feat in lst:
                if c in topl.get(h, set()) and h in topl.get(c, set()):
                    kept[frozenset((h, c))] = (sc, feat)

        # One add_alias_edges call per source vertex, with the prebuilt
        # name_to_idx so no per-source O(V) vertex-map rebuild.
        by_source: dict[str, list] = defaultdict(list)
        for pair in kept:
            src, _ = sorted(tuple(pair))
            by_source[src].append(pair)
        total = 0
        for src, keys in by_source.items():
            cands: List[AliasCandidate] = []
            feats: List[dict] = []
            wprops: List[float] = []
            for pair in keys:
                other = next(x for x in pair if x != src)
                sc, feat = kept[pair]
                cands.append(AliasCandidate(other, sc))
                feats.append(feat)
                wprops.append(float(propagation_policy(feat, cfg)))
            total += on_alias_accepted(
                cfg.acceptance_handler,
                self.graph,
                src,
                cands,
                feats,
                wprops,
                reverse_map=self._reverse_map,
                name_to_idx=name_to_idx,
            )
        return total

    def _entity_passage_sets(self, involved_vidx) -> dict:
        """entity hash_id → set(passage hash_id) for the given entity vertex
        indices, from their incident entity_passage edges.

        Scoped to ``involved_vidx`` (the ER batch's pending ∪ candidate
        entities) and read via per-vertex O(1) access, so the co-occurrence
        veto costs O(|involved|·deg) — never an O(V+E) full-graph materialise
        per flush."""
        out: dict[str, set] = defaultdict(set)
        g = self.graph
        if "edge_type" not in g.es.attributes() or "vertex_type" not in g.vs.attributes():
            return out
        for vidx in involved_vidx:
            if g.vs[vidx]["vertex_type"] != "entity":
                continue
            ename = g.vs[vidx]["name"]
            for e in g.incident(vidx):
                edge = g.es[e]
                if edge["edge_type"] != "entity_passage":
                    continue
                other = edge.target if edge.source == vidx else edge.source
                if g.vs[other]["vertex_type"] == "passage":
                    out[ename].add(g.vs[other]["name"])
        return out

    def _existing_alias_degree(self, entity_hashes) -> dict:
        """Count each entity's already-present incident alias edges (so the
        outdegree cap bounds total degree across flushes). O(|hashes|·deg)."""
        out: dict[str, int] = {}
        g = self.graph
        if "edge_type" not in g.es.attributes():
            return out
        for h in entity_hashes:
            vidx = self._name_to_vidx.get(h)
            if vidx is None:
                continue
            d = sum(1 for e in g.incident(vidx) if g.es[e]["edge_type"] == "alias")
            if d:
                out[h] = d
        return out

    def _save_er_config(self) -> None:
        """Persist the ER knobs this KG was built with (audit / reproducibility).

        ER is build-time only — query-time reads the derived clusters.json — so
        this file is not consumed at runtime; it exists so a graph's alias edges
        can be traced back to the exact gate that produced them (the intrinsic
        transparency the maintenance / over-merge-rate reporting depends on)."""
        cfg = self.config
        payload = {
            "admission_rule_version": ADMISSION_RULE_VERSION,
            "alias_edges_enabled": cfg.alias_edges_enabled,
            "acceptance_handler": cfg.acceptance_handler,
            "alias_top_k": cfg.alias_top_k,
            "alias_gradient": cfg.alias_gradient,
            "er_recall_floor": cfg.alias_min_sim,
            "er_mutual_knn": cfg.er_mutual_knn,
            "er_idf_lex_threshold": cfg.er_idf_lex_threshold,
            "er_han_token_ngram": cfg.er_han_token_ngram,
            "er_ctx_synonym_threshold": cfg.er_ctx_synonym_threshold,
            "er_cooccur_veto": cfg.er_cooccur_veto,
            "er_max_df_for_veto": cfg.er_max_df_for_veto,
            "er_max_alias_degree": cfg.er_max_alias_degree,
            "cluster_algorithm": cfg.cluster_algorithm,
            "cluster_leiden_resolution": cfg.cluster_leiden_resolution,
        }
        path = faiss_graph_dir() / "er_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

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
    def _extract_nodes(passage_to_entities, sentence_to_entities, passage_to_sentences):
        """Materialise the node sets the graph + faiss stores need.

        ``sentence_nodes`` includes EVERY sentence ``split_sentences``
        produced (drawn from ``passage_to_sentences``), not only the
        entity-bearing subset from ``sentence_to_entities``. The
        entity-bearing subset alone misses ~20 % of the corpus's
        sentences (those whose GLiNER spans were either dropped or
        never tagged) — fine for the original chain-mode edge
        scoring (only co-occurrence sentences carry meaningful
        weight) but breaks the question-conditioned preview snippet,
        which looks for the top cos(q, sent) sentence on a candidate
        page via ``GraphPPRChannel.passage_sentence_embs``. Without
        the no-entity sentences in the store, the preview silently
        picks a sub-optimal entity-bearing sentence when the answer
        sentence carries no NER hit.
        """
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
        for sentences in passage_to_sentences.values():
            for sentence in sentences:
                sentence_nodes.add(sentence)
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities

    def _extract_nodes_for_passages(self, passage_hash_ids):
        """Delta sibling of :meth:`_extract_nodes`: materialise node sets from
        ONLY the given passage hashes, not the whole accumulated cache. This is
        what keeps :meth:`bulk_index_passages` O(target) per pass instead of the
        O(total) full-state walk ``index()`` does — the difference between O(N)
        and O(N²) over a corpus of N single-passage documents."""
        entity_nodes: Set[str] = set()
        sentence_nodes: Set[str] = set()
        passage_hash_id_to_entities = defaultdict(set)
        for ph in passage_hash_ids:
            for entity in self._passage_to_entities.get(ph, []):
                entity_nodes.add(entity)
                passage_hash_id_to_entities[ph].add(entity)
            for sentence in self._passage_to_sentences.get(ph, []):
                sentence_nodes.add(sentence)
                for entity in self._sentence_to_entities.get(sentence, []):
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

    def _add_adjacent_passage_edges_for_file_ids(self, file_ids: Set[str]) -> None:
        """Bulk sibling of :meth:`_add_adjacent_passage_edges`: ONE scan over the
        passage metadata, grouped by file_id, linking adjacent pages within each
        — equivalent to calling the per-file version once per file_id but without
        re-scanning the whole store per file."""
        hash_ids = self.passage_embedding_store.hash_ids
        meta_fid = self.passage_embedding_store.meta_column("file_id")
        meta_pn = self.passage_embedding_store.meta_column("page_number")
        by_file: dict = defaultdict(list)
        for h, fid, pn in zip(hash_ids, meta_fid, meta_pn):
            if fid not in file_ids or pn is None:
                continue
            try:
                by_file[fid].append((int(pn), h))
            except (TypeError, ValueError):
                continue
        for fid, rows in by_file.items():
            rows.sort(key=lambda x: x[0])
            for i in range(len(rows) - 1):
                self.node_to_node_stats[rows[i][1]][rows[i + 1][1]] = (
                    1.0,
                    "adjacent_passage",
                )

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
            v = self.graph.add_vertex(
                name=hash_id,
                content=passage_id_to_text.get(hash_id, ""),
                vertex_type="passage",
            )
            existing_names.add(hash_id)
            self._name_to_vidx[hash_id] = v.index  # keep the ER batch's map current
        for hash_id in new_entity_hashes:
            if hash_id in existing_names:
                continue
            v = self.graph.add_vertex(
                name=hash_id,
                content=entity_id_to_text.get(hash_id, ""),
                vertex_type="entity",
            )
            existing_names.add(hash_id)
            self._name_to_vidx[hash_id] = v.index

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


