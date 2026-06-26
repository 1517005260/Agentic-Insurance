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
from ingestion.index.linear_rag.ner import GLiNERAdapter
from ingestion.index.linear_rag.normalize import canonical_form, normalize_for_hash
from storage import EmbeddingStore
from storage.embedding_store import get_or_create_store

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
        self._occurrences_path = faiss_graph_dir() / "passage_occurrences.json"

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
            self._entity_to_label,
        ) = self._load_ner_cache_or_empty()
        self._mentions_cache: dict[str, list[str]] = {}
        self._mention_seen: dict[str, set[str]] = {}
        self._rebuild_mention_index(self._sentence_to_entities)

        # Passage-occurrence map: file_id → [[page_number, passage_hash], ...]
        # for EVERY occurrence, including content-hash duplicates the embedding
        # store dedups away. Adjacency is a property of (file_id, page) pairs,
        # not of deduped passage vertices — two files sharing a byte-identical
        # passage (or one file repeating a page) collapse to one vertex whose
        # single store-meta row names only the first inserter, so building
        # adjacency from store meta silently drops the other occurrence's link.
        # ``_occurrence_seen`` is the in-memory dedup set (rebuilt on load, not
        # persisted), keeping recording O(1) per page instead of O(P²).
        self._passage_occurrences = self._load_passage_occurrences()
        self._occurrence_seen: dict[str, set] = {
            fid: {(int(p), h) for p, h in seq}
            for fid, seq in self._passage_occurrences.items()
        }
        self._occurrences_dirty = False

        # entity/passage name → igraph vertex index, kept current by
        # _augment_graph as vertices are added so no flush path rebuilds
        # the O(V) vertex map.
        self._name_to_vidx: dict[str, int] = {
            v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()
        }

        # _ner_dirty: ner_results.json has unflushed mutations.
        self._ner_dirty = False

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

    def flush_all(self, *, emit_evidence_fs: bool = False) -> int:
        """Drain deferred persistence, then graphml.

        The single sync point for everything index() defers:

        1. Three faiss/parquet stores — save() is a no-op when not
           dirty so the per-file API path (graphml_flush_every=1, fresh
           store per call) is unaffected and a bulk path with cadence>1
           amortises one full ~735 MB rewrite across many docs.
        2. NER results JSON — same dirty-flag short-circuit.
        3. graphml + ner_config.

        The KG is built physical-only (entity↔sentence, entity↔passage,
        adjacent-passage edges) — no entity resolution, no alias edges,
        no clusters, no literal backfill.

        ``emit_evidence_fs`` is OFF by default because EvidenceFS emission
        re-materializes the *whole corpus* (every doc's TSV/views) in one
        pass; running it on the per-doc ``graphml_flush_every`` cadence would
        make a corpus build O(#docs × corpus). Only the explicit
        end-of-corpus flush — ``bulk_index_passages(final_flush=True)`` and
        ``GraphIndexBuilder.flush()`` — passes ``True`` so the FS is written
        exactly once.

        Returns 0 (kept as an int for callers that log a per-flush count).
        """
        self.passage_embedding_store.save()
        self.sentence_embedding_store.save()
        self.entity_embedding_store.save()
        if self._ner_dirty:
            self._save_ner_results()
        if self._occurrences_dirty:
            self._save_passage_occurrences()

        self.flush_graphml()
        self._save_ner_config()

        # Emit EvidenceFS — the shell-operable evidence filesystem — from the
        # exact-offset segmentation of each document's combined.md. Once, at
        # corpus commit. Wrapped so a malformed corpus (a doc whose combined.md
        # was deleted / truncated) logs a warning instead of aborting the build.
        if emit_evidence_fs and getattr(self.config, "evidence_fs_enabled", True):
            try:
                from config.settings import evidence_fs_root, paddle_ocr_root
                from ingestion.index.linear_rag.evidence_fs import write_evidence_fs

                write_evidence_fs(self, evidence_fs_root(), paddle_ocr_root())
            except Exception:
                logger.warning("EvidenceFS emission failed; build continues", exc_info=True)
        return 0

    # ---------------------------------------------------------- NER caching

    # Schema version for ``ner_results.json``. Bump when the surface
    # extraction / split / cleanup pipeline changes so older caches are
    # invalidated and every passage re-runs through the new path.
    # A cache built under an older pipeline can carry sentence-shaped
    # entities or alias edges the current filters would reject, so
    # merging cached and freshly-extracted entities is unsafe; force
    # a clean rebuild on every schema change.
    NER_CACHE_VERSION = 6

    def _load_ner_cache_or_empty(self) -> tuple[dict, dict, dict, dict]:
        """Load (passage_to_entities, sentence_to_entities,
        passage_to_sentences, entity_to_label) from disk once.

        Returns empty dicts on missing file, parse error, or schema
        version mismatch. A version mismatch silently drops the cache
        so every passage re-runs through the current NER pipeline on
        the first index() call (which sees its hash absent from the
        in-memory dict and therefore "new"); otherwise stale composite
        spans would survive forever. ``entity_to_label`` defaults to ``{}``
        when the key is absent (caches written before label persistence).
        """
        if not self._ner_results_path.exists():
            return {}, {}, {}, {}
        try:
            payload = json.loads(self._ner_results_path.read_text(encoding="utf-8"))
        except Exception:
            return {}, {}, {}, {}
        if not isinstance(payload, dict):
            return {}, {}, {}, {}
        if payload.get("version") != self.NER_CACHE_VERSION:
            logger.info(
                "ner_results.json schema mismatch (cached=%s, current=%s); "
                "treating cache as empty — every passage will re-run NER",
                payload.get("version"),
                self.NER_CACHE_VERSION,
            )
            return {}, {}, {}, {}
        return (
            payload.get("passage_hash_id_to_entities", {}) or {},
            payload.get("sentence_to_entities", {}) or {},
            payload.get("passage_hash_id_to_sentences", {}) or {},
            payload.get("entity_to_label", {}) or {},
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
                    "entity_to_label": self._entity_to_label,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, self._ner_results_path)
        self._ner_dirty = False

    def _load_passage_occurrences(self) -> dict:
        """Load the file_id → [[page_number, passage_hash], …] occurrence map.

        Independent of the NER cache (its own file) so a schema change to one
        never invalidates the other. A missing / unreadable file yields an empty
        map — occurrences are then rebuilt as documents are (re)indexed."""
        if not self._occurrences_path.exists():
            return {}
        try:
            payload = json.loads(self._occurrences_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_passage_occurrences(self) -> None:
        """Persist the occurrence map. Caller checks ``_occurrences_dirty``."""
        self._occurrences_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._occurrences_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._passage_occurrences, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, self._occurrences_path)
        self._occurrences_dirty = False

    def _record_passage_occurrences(self, file_ids, page_numbers, texts) -> None:
        """Append every (file_id, page_number, passage_hash) occurrence, deduped
        per file by ``_occurrence_seen``. Re-indexing identical content is a
        no-op; a genuinely new page is appended. O(1) per page."""
        store = self.passage_embedding_store
        for fid, page, text in zip(file_ids, page_numbers, texts):
            key = (int(page) if page is not None else 0, store.hash_for(text))
            seen = self._occurrence_seen.setdefault(fid, set())
            if key in seen:
                continue
            seen.add(key)
            self._passage_occurrences.setdefault(fid, []).append([key[0], key[1]])
            self._occurrences_dirty = True

    def _link_adjacent_passages(self, occurrences) -> None:
        """Link consecutive pages (by page_number) of one file with a weight-1
        ``adjacent_passage`` edge, skipping self-loops from repeated pages."""
        items = sorted(occurrences, key=lambda x: x[0])
        for i in range(len(items) - 1):
            current, nxt = items[i][1], items[i + 1][1]
            if current != nxt:
                self.node_to_node_stats[current][nxt] = (1.0, "adjacent_passage")

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

        Returns a dict of counts: ``{passages, entities, sentences}`` —
        only counts items added by THIS call (hash dedup means re-running
        on the same content yields zeros across the board).
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
                new_entity_to_label,
            ) = self.ner.batch_ner(
                new_hash_id_to_passage, self.config.max_workers
            )
            # Persist per-surface NER labels (raw keys re-normalized to the
            # canonical surface_norm) for EvidenceFS surfaces.tsv.
            self._merge_entity_labels(new_entity_to_label)
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
            self._record_passage_occurrences(
                [file_id] * len(passages), page_numbers, passages
            )
            self._add_adjacent_passage_edges(file_id=file_id)

        # 7. Augment graph: add new vertices, append new edges (preserve existing).
        self._augment_graph(new_passage_hash_set, new_entity_hash_set)

        # Persist on a cadence. Default graphml_flush_every=1 → flush
        # every index() call (bit-identical to before; the per-file API
        # builder constructs a fresh instance per file so its counter is
        # always 1). A persistent bulk driver sets it >1 and force-
        # calls flush_all() at checkpoints / end, so the O(V+E) graphml
        # round-trip, the 3-store faiss+parquet rewrite, and the NER JSON
        # save are amortised together instead of paid every doc.
        self._index_calls += 1
        _every = max(1, int(getattr(self.config, "graphml_flush_every", 1)))
        if self._index_calls % _every == 0:
            self.flush_all()

        logger.info(
            "index() done for file_id=%s: graph=(%d v, %d e), "
            "added passages=%d entities=%d sentences=%d",
            file_id,
            self.graph.vcount(),
            self.graph.ecount(),
            len(new_passage_hash_set),
            len(new_entity_hash_set),
            len(new_sentence_hash_set),
        )

        return {
            "passages": len(new_passage_hash_set),
            "entities": len(new_entity_hash_set),
            "sentences": len(new_sentence_hash_set),
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
        call), and a single final ``flush_all``.

        ``items`` = sequence of ``(file_id, page_number, text)``; ``text`` is the
        plain passage markdown (no metadata prefix). Per-dataset knobs
        (``adjacent_passage_edges_enabled``) come from ``self.config``.
        ``final_flush`` drains the graphml + ner_config once; pass False to
        chain several bulk calls before a manual ``flush_all``.
        """
        rows = [
            (str(f), (0 if p is None else int(p)), t)
            for f, p, t in items
            if t and t.strip()
        ]
        if not rows:
            return {"passages": 0, "entities": 0, "sentences": 0}
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
            new_p2e, new_s2e, new_p2s, new_e2l = self.ner.batch_ner(
                batch, self.config.max_workers
            )
            self._merge_entity_labels(new_e2l)
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
            self._record_passage_occurrences(file_ids, page_numbers, texts)
            self._add_adjacent_passage_edges_for_file_ids(set(file_ids))
        self._augment_graph(new_passage_hash_set, new_entity_hash_set)

        self._index_calls += 1
        if final_flush:
            self.flush_all(emit_evidence_fs=True)

        logger.info(
            "bulk_index_passages done: graph=(%d v, %d e), added passages=%d "
            "entities=%d sentences=%d",
            self.graph.vcount(),
            self.graph.ecount(),
            len(new_passage_hash_set),
            len(new_entity_hash_set),
            len(set(added_sentence_hashes)),
        )
        return {
            "passages": len(new_passage_hash_set),
            "entities": len(new_entity_hash_set),
            "sentences": len(set(added_sentence_hashes)),
        }

    def _merge_entity_labels(self, raw_entity_to_label: dict) -> None:
        """Normalize raw NER label-map keys and merge into the persistent cache.

        ``raw_entity_to_label`` maps a RAW surface piece (pre-normalization)
        → GLiNER label. We re-key it through ``normalize_for_hash`` — the
        SAME canonicaliser ``_normalize_entity_surfaces`` applies to entity
        surfaces — so the keys become the canonical ``surface_norm`` that
        EvidenceFS uses. Keys that normalize to ``None`` (junk) are skipped;
        first-wins on collisions so an earlier-seen label is never clobbered.
        """
        for raw, label in raw_entity_to_label.items():
            if not label:
                continue
            norm = normalize_for_hash(
                raw,
                fold_traditional=self.config.fold_traditional,
                han_fragment_max_chars=self.config.junk_max_han_chars,
            )
            if norm is None:
                continue
            self._entity_to_label.setdefault(norm, label)

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
                    # ``entities`` is GLiNER ground truth: this entity WAS
                    # mentioned in this passage. The substring count is only a
                    # frequency proxy, and it reads 0 when the canonical key
                    # isn't a literal substring of ``canonical_form(passage)`` —
                    # entity keys pass through ``cleanup()`` before
                    # ``canonical_form`` while the passage text does not, so the
                    # two normalizations can diverge. Floor the weight to 1 so a
                    # real mention is never dropped from the graph.
                    count = 1
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
        """Link adjacent passages within ``file_id`` by page_number.

        Reads the authoritative ``_passage_occurrences`` map, not the deduped
        passage-store meta: the store keeps one row per content-hash naming only
        the first inserter, so a passage shared across files (or a repeated page)
        would lose this file's link. The occurrence map records every page.
        """
        self._link_adjacent_passages(self._passage_occurrences.get(file_id, []))

    def _add_adjacent_passage_edges_for_file_ids(self, file_ids: Set[str]) -> None:
        """Bulk sibling of :meth:`_add_adjacent_passage_edges`: link each file's
        pages from the occurrence map (one pass per file_id)."""
        for fid in file_ids:
            self._link_adjacent_passages(self._passage_occurrences.get(fid, []))

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
            self._name_to_vidx[hash_id] = v.index  # keep the vertex-index map current
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
            # Seed ``w_prop`` / ``features_json`` on every physical edge so the
            # GraphML edge schema is uniform: the query-time alias overlay
            # (disambig.add_alias_edges, still used by graph_service/graph_ppr)
            # reads these attrs, and a uniform schema avoids round-tripping
            # pre-existing edges to ``None``. ``w_prop`` mirrors ``weight``
            # under policy=cos; ``features_json`` is an empty record.
            for offset, (w, t) in enumerate(zip(new_edge_weights, new_edge_types)):
                eidx = start + offset
                self.graph.es[eidx]["weight"] = w
                self.graph.es[eidx]["edge_type"] = t
                self.graph.es[eidx]["w_prop"] = float(w)
                self.graph.es[eidx]["features_json"] = ""


