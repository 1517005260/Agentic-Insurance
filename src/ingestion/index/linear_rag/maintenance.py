"""Local-only maintenance APIs.

Three operations on the persisted graph + EmbeddingStores + caches, none of
which call external services:

* :func:`unalias`        — delete a single alias edge.
* :func:`split_cluster`  — break a cluster into multiple by removing alias edges.
* :func:`remove_file`    — drop every artifact tagged with ``file_id``.

All operations are idempotent and persist their effect to disk before
returning. The cluster cache is invalidated on any structural change.
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import faiss
import igraph as ig
import numpy as np

from config.settings import (
    bm25_root,
    faiss_dense_dir,
    faiss_graph_dir,
    faiss_graph_entity_dir,
    faiss_graph_passage_dir,
    faiss_graph_sentence_dir,
    faiss_visual_dir,
    inventory_path,
    page_assets_path,
    page_assets_root,
    paddle_ocr_root,
    passage_atoms_path,
    table_row_atoms_path,
    upload_path,
    uploads_root,
)
from ingestion.index.linear_rag.disambig import ALIAS_EDGE_TYPE, invalidate_clusters
from storage import EmbeddingStore
from storage.embedding_store import get_or_create_store


def _open_graph() -> tuple[ig.Graph, Path]:
    path = faiss_graph_dir() / "LinearRAG.graphml"
    if not path.exists():
        raise FileNotFoundError(f"LinearRAG.graphml not found: {path}")
    return ig.Graph.Read_GraphML(str(path)), path


def _save_graph(graph: ig.Graph, path: Path) -> None:
    """Write the graph atomically and strip the auto-generated ``id``
    vertex attribute that igraph injects on ``Read_GraphML``. Mirrors the
    same strip done in ``LinearRAG.index()`` so round-trips stay clean."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if "id" in graph.vs.attributes():
        del graph.vs["id"]
    graph.write_graphml(str(path))


def _clusters_path() -> Path:
    return faiss_graph_dir() / "clusters.json"


# ---------------------------------------------------------------- unalias

def unalias(entity_a_hash: str, entity_b_hash: str) -> bool:
    """Delete the alias edge between two entities. Returns True if removed."""
    graph, gpath = _open_graph()
    name_to_idx = {v["name"]: v.index for v in graph.vs if "name" in v.attributes()}
    if entity_a_hash not in name_to_idx or entity_b_hash not in name_to_idx:
        return False
    u, v = name_to_idx[entity_a_hash], name_to_idx[entity_b_hash]
    eid = graph.get_eid(u, v, error=False)
    if eid == -1:
        return False
    if "edge_type" in graph.es.attributes() and graph.es[eid]["edge_type"] != ALIAS_EDGE_TYPE:
        return False
    graph.delete_edges([eid])
    _save_graph(graph, gpath)
    invalidate_clusters(_clusters_path())
    return True


# ----------------------------------------------------------- split_cluster

def split_cluster(member_partition: Sequence[Sequence[str]]) -> int:
    """Force a cluster to split. Returns the count of alias edges deleted."""
    graph, gpath = _open_graph()
    if "edge_type" not in graph.es.attributes():
        return 0

    member_to_group: Dict[str, int] = {}
    for group_idx, group in enumerate(member_partition):
        for hash_id in group:
            member_to_group[hash_id] = group_idx
    if not member_to_group:
        return 0

    delete_eids: List[int] = []
    for e in graph.es:
        if e["edge_type"] != ALIAS_EDGE_TYPE:
            continue
        u_name = graph.vs[e.source]["name"]
        v_name = graph.vs[e.target]["name"]
        if u_name not in member_to_group or v_name not in member_to_group:
            continue
        if member_to_group[u_name] != member_to_group[v_name]:
            delete_eids.append(e.index)

    if not delete_eids:
        return 0
    graph.delete_edges(delete_eids)
    _save_graph(graph, gpath)
    invalidate_clusters(_clusters_path())
    return len(delete_eids)


# ------------------------------------------------------------- remove_file

def remove_file(
    file_id: str,
    *,
    keep_upload: bool = False,
    upload_suffix: str | None = None,
) -> Dict[str, int]:
    """Wipe every artifact tagged with ``file_id``.

    ``keep_upload=True`` preserves ``uploads/<file_id>.*`` so a re-ingest
    can purge stale indexes and then re-feed the same original blob to
    the parser. Default ``False`` is the full-delete path the web's
    DELETE route wants.

    ``upload_suffix`` (e.g. ``".pdf"``) makes the uploads-dir delete
    EXACT — only ``uploads/<file_id><suffix>`` is removed. Without it
    we fall back to a stem match that requires
    ``entry.with_suffix("").name == file_id``, which is the only safe
    inverse of ``upload_path()`` and avoids the prefix hazard where
    purging ``abc_hash`` would otherwise also match
    ``abc_hash.v2_<other>.pdf``.

    Steps (idempotent):

    1. Delete page_assets/<file_id>.json
    2. Delete inventory/<file_id>.json + inventory_atoms/{passages,table_rows}/<file_id>.json
       (lazy-built derivative caches; safe to drop, will rebuild on demand)
    3. (unless keep_upload) Delete uploads/<file_id>.* — the cached original PDF.
       Skipped during re-ingest because parse_and_index needs to read it.
    4. Delete paddle_ocr/<file_id>/ (raw outputs)
    5. Drop rows with file_id from dense + visual faiss stores
    6. Drop passage rows with file_id from the graph passage store
    7. Drop NER cache entries for those passage hashes; orphan-clean
       sentence_to_entities to surviving passage texts
    8. Drop sentence and entity rows whose surface no longer appears in any
       surviving passage (true orphans)
    9. Delete graph vertices for the dropped passage hashes; sweep orphan
       entity vertices (no surviving passage edges)
    10. Rebuild BM25 from the surviving page_assets
    11. Invalidate clusters cache

    Returns counts for telemetry. Each step is a no-op if the artifact
    isn't present, so calling twice is harmless.
    """
    counts: Dict[str, int] = {}

    # 1. Page assets JSON.
    pa = page_assets_path(file_id)
    if pa.exists():
        pa.unlink()
        counts["page_assets_json"] = 1

    # 2. Lazy derivative caches. Each is a single per-file JSON; the
    # store classes rebuild them from page_assets on next request, so
    # leaving them around after a delete would surface stale rows.
    inv = inventory_path(file_id)
    if inv.exists():
        inv.unlink()
        counts["inventory_json"] = 1
    pas_atoms = passage_atoms_path(file_id)
    if pas_atoms.exists():
        pas_atoms.unlink()
        counts["passage_atoms_json"] = 1
    trow_atoms = table_row_atoms_path(file_id)
    if trow_atoms.exists():
        trow_atoms.unlink()
        counts["table_row_atoms_json"] = 1

    # 3. Cached upload original. Caller-supplied ``upload_suffix``
    # turns this into a single-file lookup; otherwise we iterate the
    # uploads dir and only match files whose ``with_suffix("").name``
    # equals ``file_id`` exactly. The naive ``startswith(f"{file_id}.")``
    # approach would also match e.g. ``abc_hash.v2_<other>.pdf`` when
    # purging ``abc_hash``, deleting an unrelated file's blob.
    if not keep_upload:
        n_uploads = 0
        if upload_suffix is not None:
            target = upload_path(file_id, upload_suffix)
            if target.exists():
                target.unlink()
                n_uploads = 1
        else:
            up_root = uploads_root()
            if up_root.exists():
                for entry in up_root.iterdir():
                    if not entry.is_file():
                        continue
                    if entry.name.startswith(".") and entry.suffix == ".part":
                        continue
                    # Inverse of ``upload_path(file_id, suffix)``: strip
                    # the final extension and require an EXACT match.
                    if entry.with_suffix("").name == file_id or entry.name == file_id:
                        entry.unlink()
                        n_uploads += 1
        if n_uploads:
            counts["uploads"] = n_uploads

    # 4. PaddleOCR output dir.
    paddle_dir = paddle_ocr_root() / file_id
    if paddle_dir.exists():
        shutil.rmtree(paddle_dir)
        counts["paddle_ocr_dir"] = 1

    # 5. Dense + visual stores: drop by file_id.
    dense = get_or_create_store(faiss_dense_dir(), namespace="dense")
    counts["dense_rows"] = _drop_store_rows(dense, file_id=file_id)
    visual = get_or_create_store(faiss_visual_dir(), namespace="visual")
    counts["visual_rows"] = _drop_store_rows(visual, file_id=file_id)

    # 6. Graph passage store: drop passages with this file_id, remember their hashes.
    passage_store = get_or_create_store(faiss_graph_passage_dir(), namespace="passage")
    dropped_passage_hashes = _list_store_rows_by_file(passage_store, file_id)
    counts["graph_passage_rows"] = _drop_store_rows(passage_store, file_id=file_id)

    surviving_passages = passage_store.hash_id_to_text  # after drop

    # 7 + 8. NER cache + sentence/entity orphan cleanup.
    ner_path = faiss_graph_dir() / "ner_results.json"
    dropped_sentences: List[str] = []
    dropped_entities: List[str] = []
    if ner_path.exists():
        ner = json.loads(ner_path.read_text(encoding="utf-8"))
        passage_to_entities: Dict[str, List[str]] = ner.get(
            "passage_hash_id_to_entities", {}
        )
        sentence_to_entities: Dict[str, List[str]] = ner.get(
            "sentence_to_entities", {}
        )

        for h in dropped_passage_hashes:
            passage_to_entities.pop(h, None)

        # Recompute the surviving sentence universe from the surviving
        # passage texts (rather than substring-matching, which would
        # keep an A-only sentence that happens to appear inside a
        # longer B sentence). Must mirror the ingest-side NER input
        # pipeline exactly — ``preclean_for_ner`` (HTML / table /
        # LaTeX strip) then ``split_sentences`` — otherwise sentence
        # keys here drift from the keys ingest wrote into
        # ``sentence_to_entities`` and the membership test below
        # rejects every legitimate entry. The bug surfaced as a
        # cascading wipe of the NER mention map on any
        # ``remove_file`` call against an HTML-heavy corpus.
        from ingestion.index._sentence import split_sentences
        from ingestion.index.linear_rag.ner import preclean_for_ner

        surviving_sentence_set: set = set()
        for passage_text in surviving_passages.values():
            for sent in split_sentences(preclean_for_ner(passage_text)):
                surviving_sentence_set.add(sent)

        kept_sentences: Dict[str, List[str]] = {}
        for sent, ents in sentence_to_entities.items():
            if sent in surviving_sentence_set:
                kept_sentences[sent] = ents
            else:
                dropped_sentences.append(sent)
        ner["passage_hash_id_to_entities"] = passage_to_entities
        ner["sentence_to_entities"] = kept_sentences
        ner_path.write_text(json.dumps(ner, ensure_ascii=False), encoding="utf-8")
        counts["ner_passages_dropped"] = len(dropped_passage_hashes)
        counts["ner_sentences_dropped"] = len(dropped_sentences)

        # Entity orphan = entity surface not mentioned in any surviving
        # passage (across the whole corpus, after dropping file_id).
        all_surviving_entities = set()
        for ents in passage_to_entities.values():
            all_surviving_entities.update(ents)

        entity_store = get_or_create_store(faiss_graph_entity_dir(), namespace="entity")
        for h, surface in entity_store.hash_id_to_text.items():
            if surface not in all_surviving_entities:
                dropped_entities.append(h)
        if dropped_entities:
            counts["graph_entity_rows"] = _drop_store_rows_by_hash(
                entity_store, dropped_entities
            )

        # Sentence store: drop rows whose text is in dropped_sentences.
        sentence_store = get_or_create_store(faiss_graph_sentence_dir(), namespace="sentence")
        dropped_sentence_set = set(dropped_sentences)
        sent_drop_hashes = [
            h for h, t in sentence_store.hash_id_to_text.items()
            if t in dropped_sentence_set
        ]
        if sent_drop_hashes:
            counts["graph_sentence_rows"] = _drop_store_rows_by_hash(
                sentence_store, sent_drop_hashes
            )

    # 9. Graph vertices.
    graphml_path = faiss_graph_dir() / "LinearRAG.graphml"
    if graphml_path.exists():
        graph = ig.Graph.Read_GraphML(str(graphml_path))
        names_to_drop = [
            v.index for v in graph.vs
            if v.attributes().get("name") in (set(dropped_passage_hashes) | set(dropped_entities))
        ]
        if names_to_drop:
            graph.delete_vertices(names_to_drop)
            counts["graph_vertices_dropped"] = len(names_to_drop)

        # Orphan sweep: only entity vertices, only those without passage
        # evidence. A passage vertex with no edges (single-page file with
        # zero entities) is still legitimate state and must be preserved.
        if "vertex_type" in graph.vs.attributes():
            kill: List[int] = []
            for v in graph.vs:
                if v.attributes().get("vertex_type") != "entity":
                    continue
                inc = graph.incident(v.index)
                if not inc:
                    kill.append(v.index)
                    continue
                if "edge_type" in graph.es.attributes():
                    has_evidence = any(
                        graph.es[eid].attributes().get("edge_type") != ALIAS_EDGE_TYPE
                        for eid in inc
                    )
                    if not has_evidence:
                        kill.append(v.index)
            if kill:
                graph.delete_vertices(kill)
                counts["graph_vertices_orphans"] = len(kill)
        _save_graph(graph, graphml_path)

    invalidate_clusters(_clusters_path())

    # 10. BM25 rebuild from surviving page_assets/.
    counts["bm25_rebuild"] = _rebuild_bm25(skip_file_id=file_id)

    return counts


# ----------------------------------------------------------- internals

def _list_store_rows_by_file(store: EmbeddingStore, file_id: str) -> List[str]:
    # Snapshot meta under the per-store lock so a concurrent ``add()``
    # can't change the column set or row count between the column
    # check and the row select.
    with store._lock:
        if "file_id" not in store._meta.columns:
            return []
        mask = store._meta["file_id"] == file_id
        return store._meta.loc[mask, "hash_id"].tolist()


def _drop_store_rows(store: EmbeddingStore, file_id: str) -> int:
    # Mask compute + rebuild must be one critical section; otherwise a
    # concurrent ``add()`` between the two would lose the just-added
    # rows when ``_rebuild_store`` overwrites ``_index`` / ``_meta``.
    with store._lock:
        if "file_id" not in store._meta.columns or len(store) == 0:
            return 0
        keep_mask = store._meta["file_id"] != file_id
        n_dropped = int((~keep_mask).sum())
        if n_dropped == 0:
            return 0
        _rebuild_store(store, keep_mask)
        return n_dropped


def _drop_store_rows_by_hash(store: EmbeddingStore, hash_ids: Sequence[str]) -> int:
    if not hash_ids or len(store) == 0:
        return 0
    with store._lock:
        if len(store) == 0:
            return 0
        drop_set = set(hash_ids)
        keep_mask = ~store._meta["hash_id"].isin(drop_set)
        n_dropped = int((~keep_mask).sum())
        if n_dropped == 0:
            return 0
        _rebuild_store(store, keep_mask)
        return n_dropped


def _rebuild_store(store: EmbeddingStore, keep_mask) -> None:
    """Rebuild faiss index + meta from ``keep_mask``.

    Held under ``store._lock`` so the three-step mutation
    (``_index`` / ``_meta`` / ``_hash_id_to_idx``) is atomic vs concurrent
    readers. Re-entrant lock so the inner ``store.save()`` (which also
    grabs the lock) doesn't deadlock.
    """
    with store._lock:
        keep_idx = store._meta.index[keep_mask].tolist()
        survivors_meta = store._meta.loc[keep_mask].reset_index(drop=True)

        if store._index is not None and store._index.ntotal > 0:
            all_emb = store._index.reconstruct_n(0, store._index.ntotal)
            new_emb = all_emb[keep_idx]
        else:
            new_emb = np.zeros((0, store.dim or 0), dtype=np.float32)

        new_index = faiss.IndexFlatIP(store.dim or new_emb.shape[1])
        if new_emb.size > 0:
            new_index.add(new_emb)
        store._index = new_index
        store._meta = survivors_meta
        store._hash_id_to_idx = {h: i for i, h in enumerate(survivors_meta["hash_id"].tolist())}
        # In-place mutation bypasses add() so we must mark dirty manually
        # and bump _gen to invalidate the derived-view memo caches.
        store._dirty = True
        store._gen += 1
        store.save()


def _rebuild_bm25(skip_file_id: str) -> int:
    """Rebuild the global BM25 index from the surviving page_assets JSONs.

    Returns the count of (file_id, page) documents indexed after rebuild.
    """
    out_dir = bm25_root()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    survivors = sorted(page_assets_root().glob("*.json"))
    if not survivors:
        return 0

    import tantivy

    index_path = out_dir / "index"
    index_path.mkdir(parents=True, exist_ok=True)
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("file_id", stored=True)
    schema_builder.add_text_field("page_id", stored=True)
    schema_builder.add_text_field("text", stored=True)
    schema = schema_builder.build()
    index = tantivy.Index(schema, path=str(index_path))
    writer = index.writer()

    file_counts: Dict[str, int] = {}
    total = 0
    for fp in survivors:
        if fp.stem == skip_file_id:
            continue
        rows = json.loads(fp.read_text(encoding="utf-8"))
        for row in rows:
            doc = tantivy.Document()
            doc.add_text("file_id", row.get("file_id", fp.stem))
            doc.add_text("page_id", row.get("page_id", ""))
            doc.add_text("text", row.get("text_markdown") or row.get("text") or "")
            writer.add_document(doc)
            file_counts[row.get("file_id", fp.stem)] = (
                file_counts.get(row.get("file_id", fp.stem), 0) + 1
            )
            total += 1
    writer.commit()
    writer.wait_merging_threads()

    (out_dir / "meta.json").write_text(
        json.dumps(
            {"fields": ["file_id", "page_id", "text"], "file_counts": file_counts},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return total
