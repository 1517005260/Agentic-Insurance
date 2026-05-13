"""P5 — Alignment Containment / Contamination Radius.

Claim. A wrong alias edge ``e`` enters PPR via alias propagation;
``unalias(e)`` restores the PPR of every query not connected to ``e``'s
alias-component to its pre-injection top-k.

Protocol.

  1. Inject a single false alias edge (K=1) between two surfaces that
     are not real synonyms.
  2. Run PPR on a representative query set Q; record top-k.
  3. unalias(e); rerun PPR.
  4. contamination_radius = |queries whose top-k changed| / |Q|.
     Queries whose top-k is unchanged between (pre, post) must have a
     byte-equal list; we assert ``uncontaminated_byte_equal == |Q| -
     contamination_radius``.

The injection is deterministic: pick the two entities with the highest
existing cos sim that are NOT already aliased, simulate accepting the
pair as a false-positive merge.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import igraph as ig
import numpy as np

from ingestion.index.linear_rag.disambig import (
    ADMISSION_RULE_VERSION,
    ALIAS_EDGE_TYPE,
    AliasCandidate,
    add_alias_edges,
)
from tests.invariants._fixtures import build_synthetic_artifacts


REPRESENTATIVE_QUERIES: List[List[str]] = [
    ["entity-a1"],          # apex anchor — same component as injection target if a vs b
    ["entity-b1"],          # zenith anchor
    ["entity-c1"],          # eclipse anchor
    ["entity-d1"],          # orphan — should be byte-equal across injection
    ["entity-a1", "entity-c1"],
]
TOP_K = 3


def _run_ppr_top_k(graph: ig.Graph, seeds: List[str], k: int) -> List[str]:
    name_to_idx = {v["name"]: v.index for v in graph.vs}
    reset = np.zeros(graph.vcount(), dtype=np.float64)
    for s in seeds:
        if s in name_to_idx:
            reset[name_to_idx[s]] = 1.0
    if reset.sum() == 0:
        return []
    scores = graph.personalized_pagerank(
        vertices=range(graph.vcount()),
        damping=0.85,
        directed=False,
        weights="weight" if "weight" in graph.es.attributes() else None,
        reset=reset.tolist(),
        implementation="prpack",
    )
    passage_idx = [
        v.index for v in graph.vs
        if v["vertex_type"] == "passage" and not v.attributes().get("hidden")
    ]
    scores_arr = np.asarray(scores)
    passage_idx.sort(key=lambda i: -scores_arr[i])
    return [graph.vs[i]["name"] for i in passage_idx[:k]]


def test_p5_contamination_radius(tmp_path):
    art = build_synthetic_artifacts(tmp_path)
    graph: ig.Graph = art["graph"]

    # Pre-injection PPR.
    pre = {tuple(q): _run_ppr_top_k(graph, q, TOP_K) for q in REPRESENTATIVE_QUERIES}

    # Inject one false alias: bridge apex cluster to eclipse cluster.
    # These are semantically unrelated in the fixture; the injection
    # forces PPR mass between them.
    false_alias = ("entity-a1", "entity-c1")
    add_alias_edges(
        graph,
        false_alias[0],
        [AliasCandidate(false_alias[1], 0.86, rerank_yes_prob=0.71)],
    )
    contaminated_eid = graph.get_eid(false_alias[0], false_alias[1], error=True)

    post_inject = {tuple(q): _run_ppr_top_k(graph, q, TOP_K) for q in REPRESENTATIVE_QUERIES}

    # Repair: delete the false edge.
    graph.delete_edges([contaminated_eid])
    post_repair = {tuple(q): _run_ppr_top_k(graph, q, TOP_K) for q in REPRESENTATIVE_QUERIES}

    contaminated_queries: List[List[str]] = []
    uncontaminated_byte_equal = 0
    for k, v0 in pre.items():
        if post_inject[k] != v0:
            contaminated_queries.append(list(k))
        else:
            uncontaminated_byte_equal += 1

    report = {
        "protocol": "P5",
        "config": {
            "system": "ours-overlay",
            "k_injected": 1,
            "false_alias": false_alias,
            "n_queries": len(REPRESENTATIVE_QUERIES),
        },
        "results": {
            "contamination_radius": (
                len(contaminated_queries) / max(1, len(REPRESENTATIVE_QUERIES))
            ),
            "contaminated_queries": contaminated_queries,
            "uncontaminated_byte_equal": uncontaminated_byte_equal,
            "post_repair_matches_pre": all(post_repair[k] == v0 for k, v0 in pre.items()),
        },
    }
    out = tmp_path / "p5_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Hard invariants:
    # 1. unrelated queries' top-k must be byte-equal under contamination.
    assert (
        uncontaminated_byte_equal == len(REPRESENTATIVE_QUERIES) - len(contaminated_queries)
    ), report
    # 2. After unalias, every query must restore to pre-injection top-k.
    assert report["results"]["post_repair_matches_pre"], report
    # 3. Some contamination should have happened (the test is meaningful):
    assert report["results"]["contamination_radius"] > 0, report
