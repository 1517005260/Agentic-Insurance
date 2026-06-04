"""End-to-end gate test for the flush-time entity resolution batch.

Builds a tiny model-free entity store + graph and drives
``LinearRAG._resolve_entities`` directly (bypassing GLiNER via ``__new__``) to
assert the three core decisions of the redesign:

* template collision (high embedding cosine, disjoint distinctive tokens) →
  NO alias edge,
* true surface variant (high cosine, shared distinctive tokens) → alias edge,
* co-occurring distinct entities (share a passage) → NO alias edge (veto),
* outdegree cap bounds per-entity alias edges.

No embedding model is loaded — vectors are synthesised so same-"tag" entities
sit at cos≈1 and different tags at cos≈0.
"""
import numpy as np
import pytest

from config import LinearRAGConfig
from ingestion.index.linear_rag.linear_rag import LinearRAG
from storage import EmbeddingStore

import igraph as ig

DIM = 32


def _emb(tag: str, i: int) -> np.ndarray:
    base = np.random.default_rng(abs(hash(tag)) % (2**31)).standard_normal(DIM)
    noise = np.random.default_rng((abs(hash(tag)) + i) % (2**31)).standard_normal(DIM)
    v = base + 0.02 * noise
    return (v / np.linalg.norm(v)).astype("float32")


# (hash, surface, embedding-tag)
ENTITIES = [
    ("e1", "3rd baron acton", "t"),       # template cluster
    ("e2", "3rd baroness herbert", "t"),  # high cos w/ e1, disjoint head tokens
    ("e3", "dark river (2017 film)", "v"),
    ("e4", "dark river", "v"),            # high cos w/ e3, shared head tokens
    ("e5", "john smith", "s"),
    ("e6", "john smith jr", "s"),         # high cos + shared tokens, but co-occur
]
# entity → passage (e5,e6 share P1 → veto; the rest are disjoint)
ENT_PASSAGE = [("e1", "P2"), ("e2", "P3"), ("e3", "P4"), ("e4", "P5"),
               ("e5", "P1"), ("e6", "P1")]
PASSAGES = ["P1", "P2", "P3", "P4", "P5"]


def _build(tmp_path):
    store = EmbeddingStore(tmp_path / "entity", namespace="entity", dim=DIM)
    store.add(
        [h for h, _, _ in ENTITIES],
        [s for _, s, _ in ENTITIES],
        np.stack([_emb(tag, i) for i, (_, _, tag) in enumerate(ENTITIES)]),
    )
    store.save()

    g = ig.Graph(directed=False)
    for h, s, _ in ENTITIES:
        g.add_vertex(name=h, content=s, vertex_type="entity")
    for p in PASSAGES:
        g.add_vertex(name=p, content=p, vertex_type="passage")
    name_to_idx = {v["name"]: v.index for v in g.vs}
    pairs = [(name_to_idx[e], name_to_idx[p]) for e, p in ENT_PASSAGE]
    start = g.ecount()
    g.add_edges(pairs)
    for off in range(len(pairs)):
        g.es[start + off]["edge_type"] = "entity_passage"
        g.es[start + off]["weight"] = 1.0

    lr = LinearRAG.__new__(LinearRAG)  # bypass __init__ (no GLiNER load)
    lr.config = LinearRAGConfig(alias_edges_enabled=True)
    lr.entity_embedding_store = store
    lr.sentence_embedding_store = EmbeddingStore(
        tmp_path / "sentence", namespace="sentence", dim=DIM
    )
    lr.graph = g
    lr._mentions_cache = {}      # no mentions → centroid arm inert; surface arm only
    lr._reverse_map = {}
    lr._name_to_vidx = {v["name"]: v.index for v in g.vs}
    lr._token_df = {}
    lr._idf_surface_count = 0
    lr._warm_token_df()
    return lr, g


def _alias_pairs(g):
    if "edge_type" not in g.es.attributes():
        return set()
    out = set()
    for e in g.es:
        if e["edge_type"] == "alias":
            out.add(frozenset((g.vs[e.source]["name"], g.vs[e.target]["name"])))
    return out


def test_er_gate_decisions(tmp_path):
    lr, g = _build(tmp_path)
    added = lr._resolve_entities({h for h, _, _ in ENTITIES})
    pairs = _alias_pairs(g)

    # True surface variant: shared distinctive tokens → accepted.
    assert frozenset(("e3", "e4")) in pairs, f"variant pair missing: {pairs}"
    # Template collision: high cosine but disjoint head tokens → rejected.
    assert frozenset(("e1", "e2")) not in pairs, f"template collision leaked: {pairs}"
    # Co-occurring distinct entities (share P1) → vetoed even though lexical
    # overlap is high.
    assert frozenset(("e5", "e6")) not in pairs, f"co-occurrence veto failed: {pairs}"
    assert added >= 1


def test_zero_overlap_synonym_via_centroid(tmp_path):
    # us / usa: ZERO shared tokens but a true abbreviation. High bare cosine
    # (would force the old cos_bare regime split into the lexical branch and
    # reject); the nuanced rule routes lex==0 to the centroid branch, which
    # admits it when the mention-context cosine is high.
    store = EmbeddingStore(tmp_path / "entity", namespace="entity", dim=DIM)
    store.add(["u", "a"], ["us", "usa"], np.stack([_emb("z", 0), _emb("z", 1)]))
    store.save()
    sent = EmbeddingStore(tmp_path / "sentence", namespace="sentence", dim=DIM)
    s_texts = ["the us did x", "the us did y", "the usa did x", "the usa did y"]
    sent.add([f"s{i}" for i in range(4)], s_texts,
             np.stack([_emb("z", 10 + i) for i in range(4)]))  # ctx aligned w/ bare
    sent.save()
    g = ig.Graph(directed=False)
    g.add_vertex(name="u", content="us", vertex_type="entity")
    g.add_vertex(name="a", content="usa", vertex_type="entity")

    lr = LinearRAG.__new__(LinearRAG)
    lr.config = LinearRAGConfig(alias_edges_enabled=True, er_cooccur_veto=False)
    lr.entity_embedding_store = store
    lr.sentence_embedding_store = sent
    lr.graph = g
    lr._mentions_cache = {"us": s_texts[:2], "usa": s_texts[2:]}
    lr._reverse_map = {}
    lr._name_to_vidx = {v["name"]: v.index for v in g.vs}
    lr._token_df = {}
    lr._idf_surface_count = 0
    lr._warm_token_df()
    lr._resolve_entities({"u", "a"})
    assert frozenset(("u", "a")) in _alias_pairs(g), "zero-overlap synonym not admitted via centroid"


def test_outdegree_cap(tmp_path):
    # Three mutually-near, lexically-overlapping surfaces; cap=1 keeps ≤1 edge
    # incident per entity (symmetric top-L).
    store = EmbeddingStore(tmp_path / "entity", namespace="entity", dim=DIM)
    ents = [("a1", "alpha bravo charlie"), ("a2", "alpha bravo delta"),
            ("a3", "alpha bravo echo")]
    store.add([h for h, _ in ents], [s for _, s in ents],
              np.stack([_emb("g", i) for i in range(len(ents))]))
    store.save()
    g = ig.Graph(directed=False)
    for h, s in ents:
        g.add_vertex(name=h, content=s, vertex_type="entity")

    lr = LinearRAG.__new__(LinearRAG)
    lr.config = LinearRAGConfig(alias_edges_enabled=True, er_max_alias_degree=1,
                                er_cooccur_veto=False)
    lr.entity_embedding_store = store
    lr.sentence_embedding_store = EmbeddingStore(
        tmp_path / "sentence", namespace="sentence", dim=DIM
    )
    lr.graph = g
    lr._mentions_cache = {}
    lr._reverse_map = {}
    lr._name_to_vidx = {v["name"]: v.index for v in g.vs}
    lr._token_df = {}
    lr._idf_surface_count = 0
    lr._warm_token_df()
    lr._resolve_entities({h for h, _ in ents})

    for v in g.vs:
        deg = sum(1 for e in g.incident(v.index) if g.es[e]["edge_type"] == "alias")
        assert deg <= 1, f"{v['name']} alias degree {deg} > cap 1"
