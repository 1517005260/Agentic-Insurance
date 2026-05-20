"""P3 — Structural Monotonicity & Retrieval-Behavior Drift.

Claim (structural). Ingesting a new file may only:

* append new physical entity / passage nodes,
* append entity_passage / adjacent_passage edges (file_id-namespaced),
* add alias edges from new physical → old physical entities.

No vertex / non-alias edge with ``file_id != f_new`` may be modified.

Claim (behavior). Retrieval may drift; we measure ``drift_rate`` and
top-k overlap. Citation resolvability must stay 100%.

Protocol.

  1. Build G_0 (the synthetic fixture).
  2. Synthesise a new file's worth of vertices + edges; append (no
     deletes / no modification of pre-existing vertex/edge attributes).
  3. Byte-level diff: structural violations on old file_id must == 0.
  4. Behavior: run PPR on a representative query set Q against G_0
     vs G_1; report drift_rate + mean top-k overlap.

PPR here is the simplified ``personalized_pagerank`` over the
in-memory graph — same algorithm as ``GraphPPRChannel._run_ppr``
but with hand-rolled seeds (we don't have NER in this offline test).
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

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
    ["entity-a1"],
    ["entity-b1"],
    ["entity-c1", "entity-d1"],
    ["entity-d1"],
    ["entity-a2", "entity-b2"],
]

TOP_K = 3


def _vertex_snapshot(graph: ig.Graph) -> Dict[str, Dict[str, object]]:
    """Map vertex name → full attribute dict (deep copy via dict)."""
    return {v["name"]: dict(v.attributes()) for v in graph.vs}


def _edge_snapshot(graph: ig.Graph) -> Dict[Tuple[Tuple[str, str], str], Dict[str, object]]:
    out = {}
    et = "edge_type" in graph.es.attributes()
    for e in graph.es:
        a = graph.vs[e.source]["name"]
        b = graph.vs[e.target]["name"]
        key = (tuple(sorted([a, b])), e["edge_type"] if et else "")
        out[key] = {k: e[k] for k in graph.es.attributes()}
    return out


def _structural_violations(
    old_vertex_snapshot: Dict[str, Dict[str, object]],
    old_edge_snapshot: Dict[Tuple[Tuple[str, str], str], Dict[str, object]],
    new_graph: ig.Graph,
    old_file_id: str,
) -> Dict[str, int]:
    """Detect violations: an old vertex / non-alias edge attribute changed."""
    cur_v = _vertex_snapshot(new_graph)
    cur_e = _edge_snapshot(new_graph)

    vertex_violations = 0
    for name, attrs in old_vertex_snapshot.items():
        if name not in cur_v:
            vertex_violations += 1
            continue
        if cur_v[name] != attrs:
            vertex_violations += 1

    edge_violations = 0
    for key, attrs in old_edge_snapshot.items():
        # Old alias edges are allowed to coexist with new alias edges; if
        # the OLD edge itself was modified we count that as a violation,
        # but new alias edges introduced under new file_id are fine.
        if key not in cur_e:
            # Vanished — only allowed for alias edges (cluster could be
            # re-shaped at ingest); for non-alias we count it as a
            # violation.
            if key[1] != ALIAS_EDGE_TYPE:
                edge_violations += 1
            continue
        if cur_e[key] != attrs:
            edge_violations += 1
    _ = old_file_id  # documented; the old fixture's only file_id is "doc_alpha".
    return {
        "vertex_violations": vertex_violations,
        "edge_violations": edge_violations,
    }


def _simulate_new_file_ingest(graph: ig.Graph) -> List[str]:
    """Append a small new-file payload onto ``graph`` in place.

    Adds 3 entity vertices, 2 passage vertices (file_id=doc_beta),
    entity_passage edges, and a single alias edge linking a new
    entity to an old one (entity-new1 → entity-a1).
    """
    new_entities = [
        ("entity-new1", "Apex Plan Renewal"),  # alias of apex cluster
        ("entity-new2", "Solstice Service"),
        ("entity-new3", "Solstice Variant"),
    ]
    new_passages = [
        ("passage-q1", "doc_beta", 1, "Apex Plan Renewal opens at policy year 5."),
        ("passage-q2", "doc_beta", 2, "Solstice Service is a separate offering."),
    ]

    for h, surface in new_entities:
        graph.add_vertex(name=h, content=surface, vertex_type="entity")
    for h, fid, pn, text in new_passages:
        graph.add_vertex(name=h, content=text, vertex_type="passage")
        # Page-meta on the synthetic fixture is implicit (we don't run
        # a passage store here — citation resolvability test uses
        # passage_hash equality, which is enough for P3 in-scope).

    name_to_idx = {v["name"]: v.index for v in graph.vs}
    # New entity_passage edges
    new_e: List[Tuple[int, int]] = []
    new_w: List[float] = []
    new_t: List[str] = []
    new_f: List[str] = []
    new_w_prop: List[float] = []
    for ent, pas in [
        ("entity-new1", "passage-q1"),
        ("entity-new2", "passage-q2"),
        ("entity-new3", "passage-q2"),
    ]:
        new_e.append((name_to_idx[ent], name_to_idx[pas]))
        new_w.append(1.0)
        new_t.append("entity_passage")
        new_f.append("")
        new_w_prop.append(1.0)
    if new_e:
        start = graph.ecount()
        graph.add_edges(new_e)
        for offset, (w, t, f, wp) in enumerate(zip(new_w, new_t, new_f, new_w_prop)):
            graph.es[start + offset]["weight"] = w
            graph.es[start + offset]["edge_type"] = t
            graph.es[start + offset]["features_json"] = f
            graph.es[start + offset]["w_prop"] = wp
    # One alias edge: entity-new1 → entity-a1 (apex synonym).
    add_alias_edges(
        graph,
        "entity-new1",
        [AliasCandidate("entity-a1", 0.94, rerank_yes_prob=0.89)],
    )
    return [h for h, *_ in new_passages]


def _run_ppr(graph: ig.Graph, seeds: List[str], top_k: int) -> List[str]:
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
    passage_idx = [v.index for v in graph.vs if v["vertex_type"] == "passage"]
    if not passage_idx:
        return []
    scores_arr = np.asarray(scores)
    order = sorted(passage_idx, key=lambda i: -scores_arr[i])
    return [graph.vs[i]["name"] for i in order[:top_k]]


def test_p3_structural_monotonicity_and_drift(tmp_path):
    art = build_synthetic_artifacts(tmp_path)
    graph: ig.Graph = art["graph"]

    old_v_snap = _vertex_snapshot(graph)
    old_e_snap = _edge_snapshot(graph)

    # Behavior baseline.
    r0 = {tuple(q): _run_ppr(graph, q, TOP_K) for q in REPRESENTATIVE_QUERIES}

    new_passages = _simulate_new_file_ingest(graph)
    _ = new_passages  # the test asserts on in-scope passages only.

    violations = _structural_violations(old_v_snap, old_e_snap, graph, "doc_alpha")

    r1 = {tuple(q): _run_ppr(graph, q, TOP_K) for q in REPRESENTATIVE_QUERIES}
    drift_changes = sum(1 for k, v in r0.items() if v != r1.get(k))
    overlaps = []
    for k, v0 in r0.items():
        v1 = r1.get(k, [])
        overlaps.append(len(set(v0) & set(v1)) / max(1, TOP_K))
    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0

    # Citation resolvability — every old passage must still appear in
    # the graph under its original name (in-scope = file_id="doc_alpha").
    old_passages = {n for n, attrs in old_v_snap.items() if attrs.get("vertex_type") == "passage"}
    present = {v["name"] for v in graph.vs}
    resolvability = len(old_passages & present) / max(1, len(old_passages))

    report = {
        "protocol": "P3",
        "config": {"system": "ours-overlay", "queries": len(REPRESENTATIVE_QUERIES)},
        "results": {
            "vertex_violations": violations["vertex_violations"],
            "edge_violations": violations["edge_violations"],
            "drift_rate": drift_changes / max(1, len(REPRESENTATIVE_QUERIES)),
            "mean_top_k_overlap": mean_overlap,
            "citation_resolvability": resolvability,
        },
    }
    out = tmp_path / "p3_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    assert violations["vertex_violations"] == 0, report
    assert violations["edge_violations"] == 0, report
    assert resolvability == 1.0, report
    # Drift is measured, not bounded — we only assert it was computed.
    assert 0.0 <= report["results"]["drift_rate"] <= 1.0
