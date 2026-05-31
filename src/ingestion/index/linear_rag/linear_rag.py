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
from typing import List, Optional, Set

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

        # Refresh cluster_shape against the post-backfill graph so the
        # final-flush monitoring snapshot is current rather than carrying
        # whatever stale value the cadenced index() loop happened to
        # leave behind. Cheap path (cheap=True) — same as the per-doc
        # cluster_shape computation.
        is_collapse = self.config.acceptance_handler != ACCEPTANCE_HANDLER_OVERLAY
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
            #     canonical key). Previously this was run over the full
            #     accumulated state every doc — pure waste, since cached
            #     entries were normalised when they were the delta on
            #     their own ingest call (version-bump forces a full
            #     rebuild via _load_ner_cache_or_empty when the
            #     normalisation rules change).
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
        self._add_adjacent_passage_edges(file_id=file_id)

        # 7. Augment graph: add new vertices, append new edges (preserve existing).
        self._augment_graph(new_passage_hash_set, new_entity_hash_set)

        # 8. Entity disambiguation — add alias edges via dual-query
        #    (bare-surface + mention-centroid) gradient-cutoff recall,
        #    composite-surface admission gates, and a final pairwise
        #    Qwen3-Reranker veto. Physical nodes are NOT merged.
        added_alias_edges = self._add_alias_edges_for_new_entities(new_entity_hash_set)

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

        # Cluster shape (a derived, monitoring-only view). On cadence
        # we recompute against the live graph + cache the result; off
        # cadence we reuse the cached snapshot. Collapse mode always
        # recomputes — it walks the reverse_map, not the alias subgraph.
        # The previous off-cadence "fallback" path still ran
        # compute_clusters(connected_components), which is O(E) per doc
        # (alias-edge filter + subgraph extract + union-find) — paying
        # that per doc was its own O(N²) tail. Cached reuse drops it to
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
        """Return ``entity_text → list[unique_sentence_text]`` (cached).

        Maintained incrementally: warmed once in __init__ from any
        on-disk ner_results.json, then updated by ``_update_mention_index``
        on every per-doc delta. Reading is now O(1) — the previous
        implementation re-read the 38 MB JSON file and rebuilt the
        reverse index from scratch on every ingest call, which was the
        dominant fixed cost in the bulk-build profile.
        """
        return self._mentions_cache

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


