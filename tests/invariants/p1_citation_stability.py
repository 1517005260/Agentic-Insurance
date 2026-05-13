"""P1 — Citation Stability.

Claim. Alias operations do not modify physical / passage nodes or page
spans; an old citation must still resolve to the same source_span_hash
after random alias add / unalias / split_cluster.

Scope. Alias-only operations: failure rate must equal 0%. remove_file
is out-of-scope (intentional invalidation; caller filters).

Protocol.

  1. Snapshot G_0, sample N=20 historical citations from
     entity_passage edges.
  2. Execute K=5 random alias operations (mix of add / unalias /
     split).
  3. After each op, every in-scope citation must resolve to the same
     ``(passage_hash, page_number)`` it did against G_0.

Output: JSON-shaped report dict.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import igraph as ig
import pytest

from ingestion.index.linear_rag.disambig import (
    ADMISSION_RULE_VERSION,
    ALIAS_EDGE_TYPE,
    AliasCandidate,
    add_alias_edges,
)
from tests.invariants._fixtures import build_synthetic_artifacts, resolve_citation


N_CITATIONS = 20
K_OPS = 5


def _sample_citations(
    graph: ig.Graph, n: int, rng: random.Random
) -> List[Tuple[str, str]]:
    """Return ``[(entity_hash, passage_hash), ...]`` from entity_passage edges."""
    citations: List[Tuple[str, str]] = []
    for e in graph.es:
        if e["edge_type"] != "entity_passage":
            continue
        u = graph.vs[e.source]
        v = graph.vs[e.target]
        if u["vertex_type"] == "entity":
            citations.append((u["name"], v["name"]))
        else:
            citations.append((v["name"], u["name"]))
    if not citations:
        return []
    if len(citations) <= n:
        return citations
    return rng.sample(citations, n)


def _resolve_all(
    graph: ig.Graph, citations: List[Tuple[str, str]]
) -> Dict[Tuple[str, str], Tuple[str | None, int | None]]:
    return {c: resolve_citation(graph, c[1]) for c in citations}


def _random_alias_op(
    graph: ig.Graph, rng: random.Random
) -> str:
    """Execute one random alias op in place; return the op label."""
    entity_idx = [v.index for v in graph.vs if v["vertex_type"] == "entity"]
    op = rng.choice(["add", "unalias", "split"])

    if op == "add":
        # Pick two entities that aren't already aliased.
        for _ in range(20):
            a, b = rng.sample(entity_idx, 2)
            if not graph.are_adjacent(a, b):
                add_alias_edges(
                    graph,
                    graph.vs[a]["name"],
                    [AliasCandidate(graph.vs[b]["name"], 0.92, 0.85)],
                )
                return "add"
        return "add_skipped_no_pair"
    if op == "unalias":
        alias_es = [e for e in graph.es if e["edge_type"] == ALIAS_EDGE_TYPE]
        if alias_es:
            target = rng.choice(alias_es)
            graph.delete_edges([target.index])
        return "unalias"
    # split — remove every alias edge inside a randomly-picked component
    # (forces it to fully split into singletons).
    alias_es = [e.index for e in graph.es if e["edge_type"] == ALIAS_EDGE_TYPE]
    if alias_es:
        sub = graph.subgraph_edges(alias_es, delete_vertices=True)
        components = sub.connected_components()
        if components:
            cid = rng.randrange(len(components))
            member_names = {sub.vs[i]["name"] for i in components[cid]}
            to_delete = [
                e.index
                for e in graph.es
                if e["edge_type"] == ALIAS_EDGE_TYPE
                and graph.vs[e.source]["name"] in member_names
                and graph.vs[e.target]["name"] in member_names
            ]
            graph.delete_edges(to_delete)
    return "split"


def _run_protocol(tmp_path: Path, seed: int = 0xC17A710) -> Dict[str, object]:
    art = build_synthetic_artifacts(tmp_path)
    graph: ig.Graph = art["graph"]
    rng = random.Random(seed)

    citations = _sample_citations(graph, N_CITATIONS, rng)
    g0_resolutions = _resolve_all(graph, citations)

    failures = 0
    op_log: List[str] = []
    for _ in range(K_OPS):
        op_log.append(_random_alias_op(graph, rng))
        cur = _resolve_all(graph, citations)
        for c, ref in g0_resolutions.items():
            if cur[c] != ref:
                failures += 1
    return {
        "protocol": "P1",
        "config": {
            "system": "ours-overlay",
            "n_citations": len(citations),
            "k_operations": K_OPS,
            "scope": "alias_only",
        },
        "results": {
            "failure_rate": failures / max(1, len(citations) * K_OPS),
            "failures": failures,
            "op_log": op_log,
        },
    }


# pytest entry point
def test_p1_citation_stability_alias_only(tmp_path):
    report = _run_protocol(tmp_path)
    # Persist JSON next to the test for offline inspection / paper figure.
    out = tmp_path / "p1_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert report["results"]["failure_rate"] == 0.0, (
        f"P1 violated: failure_rate={report['results']['failure_rate']!r}\n"
        f"op_log={report['results']['op_log']}"
    )
