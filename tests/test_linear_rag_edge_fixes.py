"""Most-correct fixes for two LinearRAG construction caveats.

Caveat ①  adjacency is a property of (file_id, page) occurrences, not of the
          content-deduped passage vertices — two files sharing a byte-identical
          passage must each keep their full intra-doc chain through the shared
          vertex.
Caveat ②  an entity GLiNER tagged in a passage must keep its entity→passage
          edge even when its canonical key is not a literal substring of
          canonical_form(passage) (the count proxy reads 0); the weight floors
          to 1, the edge is never dropped.

Both are exercised on stub stores so the test is hermetic (no embeddings, NER,
faiss, or igraph) and pins the edge-construction logic in isolation.
"""

import hashlib
from collections import defaultdict
from types import SimpleNamespace

from ingestion.index.linear_rag.linear_rag import LinearRAG


def _bare_linear_rag() -> LinearRAG:
    """A LinearRAG with only the attributes the edge methods touch — built
    without __init__ so no heavy stores / models are constructed."""
    rag = object.__new__(LinearRAG)
    rag.node_to_node_stats = defaultdict(dict)
    rag._passage_occurrences = {}
    rag._occurrence_seen = {}
    rag._occurrences_dirty = False
    return rag


class _FakeStore:
    def hash_for(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------- caveat ① ----

def test_shared_passage_keeps_both_files_adjacency():
    rag = _bare_linear_rag()
    rag.passage_embedding_store = _FakeStore()
    shared = "This identical clause appears verbatim in two different files."

    rag._record_passage_occurrences(
        ["A", "A", "A"], [1, 2, 3], ["A-page-1", shared, "A-page-3"]
    )
    rag._record_passage_occurrences(
        ["B", "B", "B"], [1, 2, 3], ["B-page-1", shared, "B-page-3"]
    )

    h = _FakeStore().hash_for
    hS = h(shared)
    rag._add_adjacent_passage_edges(file_id="A")
    rag._add_adjacent_passage_edges(file_id="B")

    # File B's chain must run B1 → shared → B3 — the bug dropped these because
    # the shared vertex's store meta named only file A.
    assert rag.node_to_node_stats[h("B-page-1")][hS] == (1.0, "adjacent_passage")
    assert rag.node_to_node_stats[hS][h("B-page-3")] == (1.0, "adjacent_passage")
    # File A's chain is intact too.
    assert rag.node_to_node_stats[h("A-page-1")][hS] == (1.0, "adjacent_passage")
    assert rag.node_to_node_stats[hS][h("A-page-3")] == (1.0, "adjacent_passage")


def test_repeated_identical_page_is_not_self_looped():
    rag = _bare_linear_rag()
    rag.passage_embedding_store = _FakeStore()
    dup = "A boilerplate page repeated twice in one file."
    rag._record_passage_occurrences(["A", "A", "A"], [1, 2, 3], ["p1", dup, dup])
    rag._add_adjacent_passage_edges(file_id="A")

    h = _FakeStore().hash_for
    # p1 → dup is a real edge; dup → dup must NOT be (no self-loop).
    assert h(dup) in rag.node_to_node_stats[h("p1")]
    assert h(dup) not in rag.node_to_node_stats.get(h(dup), {})


def test_record_occurrences_is_idempotent():
    rag = _bare_linear_rag()
    rag.passage_embedding_store = _FakeStore()
    rag._record_passage_occurrences(["A", "A"], [1, 2], ["p1", "p2"])
    rag._occurrences_dirty = False
    rag._record_passage_occurrences(["A", "A"], [1, 2], ["p1", "p2"])  # re-index
    assert len(rag._passage_occurrences["A"]) == 2
    assert rag._occurrences_dirty is False  # nothing new → no write needed


# ----------------------------------------------------------- caveat ② ----

def _entity_edge_rag(passage_text, entities, entity_keys):
    rag = _bare_linear_rag()
    rag.config = SimpleNamespace(fold_traditional=True)
    rag.passage_embedding_store = SimpleNamespace(hash_id_to_text={"p1": passage_text})
    rag.entity_embedding_store = SimpleNamespace(
        text_to_hash_id={k: f"e_{i}" for i, k in enumerate(entity_keys)}
    )
    rag._add_entity_to_passage_edges({"p1": set(entities)}, restrict_passages={"p1"})
    return rag


def test_entity_edge_survives_substring_miss():
    # "acme corp" (cleanup folded the hyphen to a space in the entity key) is
    # not a literal substring of canonical_form("...ACME-Corp...") = "acme-corp
    # ...". GLiNER asserted presence, so the edge must still exist at weight 1.
    rag = _entity_edge_rag(
        "The ACME-Corp annual report for shareholders.",
        entities=["acme corp"],
        entity_keys=["acme corp"],
    )
    assert rag.node_to_node_stats["p1"]["e_0"] == (1.0, "entity_passage")


def test_substring_hit_keeps_frequency_weighting():
    # Two real substring hits vs one — relative weights must be unchanged by the
    # floor fix (it only touches the count==0 case).
    rag = _entity_edge_rag(
        "alpha alpha beta",
        entities=["alpha", "beta"],
        entity_keys=["alpha", "beta"],
    )
    assert rag.node_to_node_stats["p1"]["e_0"] == (2 / 3, "entity_passage")  # alpha
    assert rag.node_to_node_stats["p1"]["e_1"] == (1 / 3, "entity_passage")  # beta
