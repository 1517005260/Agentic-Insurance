"""Unit tests for the redesigned chain_entity retrieval primitives.

Covers the three correctness-critical pieces of the 2-mode rewrite, without
needing a built corpus (objects are constructed via ``__new__`` with the
minimal state each primitive reads):

* ``GraphPPRChannel.cooccurrence_neighbors`` — sentence-first expansion: a
  rare but query-relevant bridge survives, a high-count low-relevance hub is
  ranked below it, and a same-cluster (alias) partner is dropped.
* ``reciprocal_rank_fusion`` — the only constant (k=60) in path/page scoring.
* ``GraphExploreTool._rank_paths_rrf`` — a path connecting two focus clusters
  ranks above an open-ended path (target-aware comparison).
"""
import numpy as np

from rag.channels.base import reciprocal_rank_fusion
from rag.channels.graph_ppr import GraphPPRChannel
from agentic.tools.acquisition.graph_explore import GraphExploreTool


def _make_channel():
    """A GraphPPRChannel with just the maps cooccurrence_neighbors reads."""
    ch = GraphPPRChannel.__new__(GraphPPRChannel)
    # Tail co-occurs with: bridge (1 high-cos sentence), hub (5 low-cos
    # sentences), and a same-cluster alias variant (1 mid-cos sentence).
    ent_to_sents = {
        "tail": ["s_bridge", "s_hub1", "s_hub2", "s_hub3", "s_hub4", "s_hub5", "s_var"],
        "bridge": ["s_bridge"],
        "hub": ["s_hub1", "s_hub2", "s_hub3", "s_hub4", "s_hub5"],
        "variant": ["s_var"],
    }
    sent_to_ents = {
        "s_bridge": ["tail", "bridge"],
        "s_hub1": ["tail", "hub"], "s_hub2": ["tail", "hub"],
        "s_hub3": ["tail", "hub"], "s_hub4": ["tail", "hub"],
        "s_hub5": ["tail", "hub"], "s_var": ["tail", "variant"],
    }
    ch._entity_to_sents = ent_to_sents
    ch._sent_to_entities = sent_to_ents
    # tail and its variant share a cluster; bridge and hub are their own.
    ch._cluster_cache = {"c_tail": ["tail", "variant"], "c_bridge": ["bridge"], "c_hub": ["hub"]}
    ch._member_to_cluster_cache = {
        "tail": "c_tail", "variant": "c_tail", "bridge": "c_bridge", "hub": "c_hub",
    }
    ch._name_to_vidx = {"tail": 0, "bridge": 1, "hub": 2, "variant": 3}
    ch._is_hidden = lambda vidx: False  # nothing collapsed in this fixture
    return ch


def _sent_sims():
    order = ["s_bridge", "s_hub1", "s_hub2", "s_hub3", "s_hub4", "s_hub5", "s_var"]
    sims = {"s_bridge": 0.9, "s_hub1": 0.1, "s_hub2": 0.1, "s_hub3": 0.1,
            "s_hub4": 0.1, "s_hub5": 0.1, "s_var": 0.5}
    sent_idx = {h: i for i, h in enumerate(order)}
    arr = np.array([sims[h] for h in order], dtype=np.float64)
    return arr, sent_idx


def test_cooccurrence_neighbors_sentence_first_keeps_rare_bridge():
    ch = _make_channel()
    sims, sent_idx = _sent_sims()
    # top_s=2 keeps only the two highest-cos tail sentences (s_bridge=0.9,
    # s_var=0.5). The 5 hub sentences (cos 0.1) never enter the window — a
    # count-first prefilter would have kept the hub and dropped the bridge.
    nbrs = ch.cooccurrence_neighbors("tail", sims, sent_idx, top_s=2, top_l=20)
    clusters = [n["cluster_id"] for n in nbrs]
    assert "c_bridge" in clusters, "rare high-cos bridge must survive sentence-first"
    assert "c_hub" not in clusters, "low-cos hub must not enter the top-S window"
    assert "c_tail" not in clusters, "same-cluster alias variant must be dropped"


def test_cooccurrence_neighbors_ranks_bridge_over_hub():
    ch = _make_channel()
    sims, sent_idx = _sent_sims()
    nbrs = ch.cooccurrence_neighbors("tail", sims, sent_idx, top_s=32, top_l=20)
    clusters = [n["cluster_id"] for n in nbrs]
    assert clusters[0] == "c_bridge", "highest query-cos edge ranks first, not highest count"
    assert "c_hub" in clusters and "c_tail" not in clusters
    bridge = next(n for n in nbrs if n["cluster_id"] == "c_bridge")
    hub = next(n for n in nbrs if n["cluster_id"] == "c_hub")
    assert bridge["max_cos"] > hub["max_cos"]
    assert hub["support"] == 5  # support is metadata, not a penalty


def test_reciprocal_rank_fusion_basic():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]], k=60)
    # 'a': 1/61 + 1/62 ; 'b': 1/62 + 1/61 → tie; 'c': only 1/63 → lowest.
    assert abs(fused["a"] - fused["b"]) < 1e-12
    assert fused["a"] > fused["c"]
    # An item in only one list still scores.
    assert reciprocal_rank_fusion([["x"]])["x"] == 1.0 / 61


def test_rank_paths_rrf_prefers_two_focus_cluster_bridge():
    tool = GraphExploreTool.__new__(GraphExploreTool)

    class _Stub:
        def cluster_passage_count(self, cid):  # specificity term; uniform here
            return 1

    tool._channel = _Stub()
    m2c = {"A": "cA", "B": "cB", "x": "cX", "y": "cY"}
    focus = {"cA", "cB"}
    seed_clusters = {"cA", "cB"}
    # p_bridge connects the two focus clusters; p_open wanders to a non-focus
    # endpoint with an equally strong edge.
    p_bridge = {"nodes": ["A", "B"], "edges": [{"max_cos": 0.7}]}
    p_open = {"nodes": ["A", "x"], "edges": [{"max_cos": 0.7}]}
    ranked = tool._rank_paths_rrf([p_open, p_bridge], seed_clusters, focus, m2c)
    assert ranked[0] is p_bridge, "path joining both focus clusters must rank first"
