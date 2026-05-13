"""Synthetic graph + entity-store fixture for invariant tests.

Real LinearRAG ingestion needs GLiNER + embedding API + reranker — too
heavy for a unit-style invariant test and too slow for CI. We build a
small in-memory graph (8 entities, 4 passages, hand-crafted alias /
entity_passage edges) plus a tiny faiss-backed entity store in a
``tmp_path`` so every test exercises the production code paths
(graph mutation, PPR, cluster aggregation, citation resolution) on
data we fully understand.

The fixture is intentionally minimal: no NER, no embedding model, no
RAG channel — pure ingestion-layer artifacts at the shape the downstream
``GraphPPRChannel`` and tools expect.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import igraph as ig
import numpy as np

from ingestion.index.linear_rag.disambig import (
    ADMISSION_RULE_VERSION,
    ALIAS_EDGE_TYPE,
)
from storage.embedding_store import EmbeddingStore


# Each entry: (entity_hash, surface, "cluster_tag" used to generate the
# synthetic embedding so alias-cluster pairs land at high cos sim).
ENTITIES: List[Tuple[str, str, str]] = [
    ("entity-a1", "Apex Plan", "apex"),
    ("entity-a2", "Apex Plan (Premium)", "apex"),
    ("entity-a3", "APEX PLAN", "apex"),
    ("entity-b1", "Zenith Cover", "zenith"),
    ("entity-b2", "Zenith Coverage", "zenith"),
    ("entity-c1", "Eclipse Note", "eclipse"),
    ("entity-c2", "Eclipse Memo", "eclipse"),
    ("entity-d1", "Orphan Term", "orphan"),
]

PASSAGES: List[Tuple[str, str, str, int]] = [
    # (passage_hash, text, file_id, page_number)
    ("passage-p1", "Apex Plan is the headline product.", "doc_alpha", 1),
    ("passage-p2", "Zenith Cover and Apex Plan share an underwriter.", "doc_alpha", 2),
    ("passage-p3", "Eclipse Note documents an obscure clause.", "doc_alpha", 3),
    ("passage-p4", "Orphan Term has no nearby cousins.", "doc_alpha", 4),
]

# (entity_hash, passage_hash, weight)
ENTITY_PASSAGE_EDGES: List[Tuple[str, str, float]] = [
    ("entity-a1", "passage-p1", 1.0),
    ("entity-a2", "passage-p1", 0.5),  # variant also mentioned
    ("entity-a1", "passage-p2", 0.4),
    ("entity-b1", "passage-p2", 0.6),
    ("entity-b2", "passage-p2", 0.3),
    ("entity-c1", "passage-p3", 1.0),
    ("entity-c2", "passage-p3", 0.5),
    ("entity-d1", "passage-p4", 1.0),
]

# (entity_a, entity_b, cos_sim, rerank_yes_prob)
ALIAS_EDGES: List[Tuple[str, str, float, float]] = [
    ("entity-a1", "entity-a2", 0.93, 0.88),
    ("entity-a1", "entity-a3", 0.91, 0.85),
    ("entity-b1", "entity-b2", 0.92, 0.84),
    ("entity-c1", "entity-c2", 0.90, 0.81),
]


_EMBEDDING_DIM = 16


def _make_embedding(tag: str, salt: int) -> np.ndarray:
    """Deterministic L2-normalized vector keyed by ``tag`` + ``salt``.

    Same ``tag`` → near-identical vectors (cos sim ≈ 0.95+); different
    tags → low cos sim. Tied to the synthetic alias clusters so the
    ``EmbeddingStore.topk`` query path produces expected neighbour
    sets without running a real embedding model.
    """
    rng = np.random.default_rng(hash((tag, "base")) % (2**32))
    base = rng.standard_normal(_EMBEDDING_DIM)
    salt_rng = np.random.default_rng(hash((tag, "salt", salt)) % (2**32))
    perturbation = 0.05 * salt_rng.standard_normal(_EMBEDDING_DIM)
    vec = base + perturbation
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def build_synthetic_artifacts(tmp_path: Path) -> Dict[str, object]:
    """Materialize a fresh graphml + entity-store + passage-store under ``tmp_path``.

    Returns a dict with handles the tests can poke at:

    * ``graph``         — the in-memory igraph (also persisted as graphml)
    * ``graph_path``    — on-disk graphml location
    * ``entity_store``  — populated faiss store (~8 rows)
    * ``passage_store`` — populated faiss store (~4 rows, with file_id/page_number meta)
    * ``faiss_dir``     — root directory parent (caller can monkeypatch settings)
    """
    faiss_dir = tmp_path / "faiss" / "graph"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    entity_store = EmbeddingStore(faiss_dir / "entity", namespace="entity", dim=_EMBEDDING_DIM)
    ent_hashes = [e[0] for e in ENTITIES]
    ent_texts = [e[1] for e in ENTITIES]
    ent_embs = np.stack(
        [_make_embedding(tag, i) for i, (_, _, tag) in enumerate(ENTITIES)],
        axis=0,
    )
    entity_store.add(ent_hashes, ent_texts, ent_embs)

    passage_store = EmbeddingStore(faiss_dir / "passage", namespace="passage", dim=_EMBEDDING_DIM)
    pas_hashes = [p[0] for p in PASSAGES]
    pas_texts = [p[1] for p in PASSAGES]
    pas_embs = np.stack(
        [_make_embedding(f"passage-{i}", 0) for i, _ in enumerate(PASSAGES)],
        axis=0,
    )
    passage_store.add(
        pas_hashes,
        pas_texts,
        pas_embs,
        extra_metadata={
            "file_id": [p[2] for p in PASSAGES],
            "page_number": [p[3] for p in PASSAGES],
        },
    )

    sentence_store = EmbeddingStore(faiss_dir / "sentence", namespace="sentence", dim=_EMBEDDING_DIM)

    graph = ig.Graph(directed=False)
    for h, surface, _ in ENTITIES:
        graph.add_vertex(name=h, content=surface, vertex_type="entity")
    for h, text, _, _ in PASSAGES:
        graph.add_vertex(name=h, content=text, vertex_type="passage")

    name_to_idx = {v["name"]: v.index for v in graph.vs}

    # Entity-passage edges
    e_pairs = []
    e_weights = []
    e_types = []
    e_features = []
    e_w_props = []
    for ent, pas, w in ENTITY_PASSAGE_EDGES:
        e_pairs.append((name_to_idx[ent], name_to_idx[pas]))
        e_weights.append(w)
        e_types.append("entity_passage")
        e_features.append("")
        e_w_props.append(w)
    # Alias edges with features_json + w_prop
    for a, b, cos, rk in ALIAS_EDGES:
        e_pairs.append((name_to_idx[a], name_to_idx[b]))
        e_weights.append(cos)
        e_types.append(ALIAS_EDGE_TYPE)
        feats = {
            "cos_sim": cos,
            "reranker_score": rk,
            "admission_rule_version": ADMISSION_RULE_VERSION,
            "accepted_by": "gradient_er",
        }
        e_features.append(json.dumps(feats, ensure_ascii=False))
        e_w_props.append(cos)

    graph.add_edges(e_pairs)
    graph.es["weight"] = e_weights
    graph.es["edge_type"] = e_types
    graph.es["features_json"] = e_features
    graph.es["w_prop"] = e_w_props

    graphml_path = faiss_dir / "LinearRAG.graphml"
    graph.write_graphml(str(graphml_path))

    return {
        "graph": graph,
        "graph_path": graphml_path,
        "entity_store": entity_store,
        "passage_store": passage_store,
        "sentence_store": sentence_store,
        "faiss_dir": faiss_dir,
    }


def resolve_citation(
    graph: ig.Graph,
    passage_hash: str,
    reverse_map: Dict[str, str] | None = None,
) -> Tuple[str | None, int | None]:
    """Resolve a stored passage hash back to ``(file_id, page_number)``.

    The synthetic ``content`` payload is the passage text — for citation
    stability we don't need it; we use the ``name`` (hash) as the
    ``source_span_hash``. A real production resolver would read
    ``passage_store.meta_column``; this stripped-down version keeps the
    test free of an embedding store handle.
    """
    if passage_hash not in [v["name"] for v in graph.vs]:
        return None, None
    # In collapse mode, citations to an absorbed (hidden) physical
    # entity should resolve to the canonical via reverse_map. Passage
    # nodes are never collapsed, so reverse_map is irrelevant for
    # passage hashes — but we accept it so call sites can stay uniform.
    _ = reverse_map  # documented; unused for passage citations.
    return passage_hash, None
