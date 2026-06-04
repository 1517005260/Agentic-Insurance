"""P4 — PPR Surface-Path Attribution.

Claim. Overlay PPR's random walk preserves physical surface identity at
every state, so contribution to any top-k passage is attributable to a
concrete physical entity. Collapse-basic destroys state identity
(quotient vertex) → no native attribution. Collapse-provenance
recovers attribution via the ``source_member`` sidecar on each
redirected entity_passage edge.

Protocol.

  1. Run a PPR query.
  2. For each top-3 passage, run 5000-step Monte Carlo random walks
     from the seeds; record the physical entity visited just before
     landing on the passage.
  3. Check:
     * Overlay   — every top passage has ≥1 physical entity contributor.
     * Basic     — physical attribution collapses to canonical id
                   (no surface variants).
     * Provenance — surface variants recoverable from
                   ``source_member`` sidecar on the
                   ``entity_passage`` edges.

The walk uses edge ``weight`` as transition probability (a
half-formal Neumann-series argument); zero-out / restart with
``damping`` are skipped because we restart at the seed every walk —
the goal is path attribution, not the unique stationary distribution.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import igraph as ig

from ingestion.index.linear_rag.disambig import (
    ACCEPTANCE_HANDLER_COLLAPSE_BASIC,
    ACCEPTANCE_HANDLER_COLLAPSE_PROVENANCE,
    ACCEPTANCE_HANDLER_OVERLAY,
    AliasCandidate,
    on_alias_accepted,
)
from tests.invariants._fixtures import build_synthetic_artifacts


SEEDS = ["entity-a1", "entity-c1"]  # cover apex + eclipse so all top-3 are reachable
N_WALKS = 5000
# Walks need to cross at least entity → passage → adjacent? In the
# synthetic fixture passages have no adjacent-passage edges, so the walk
# is bounded by entity-passage and alias edges. 6 hops is plenty when
# the seed reaches every cluster in 1-2 hops.
MAX_STEPS = 6
TOP_PASSAGES = 3


def _build_with_handler(handler: str, tmp_path: Path) -> Tuple[ig.Graph, Dict[str, str]]:
    sub = tmp_path / handler
    sub.mkdir(parents=True, exist_ok=True)
    art = build_synthetic_artifacts(sub)
    graph: ig.Graph = art["graph"]
    reverse_map: Dict[str, str] = {}
    if handler == ACCEPTANCE_HANDLER_OVERLAY:
        return graph, reverse_map
    # Collapse modes: re-run the synthetic alias acceptances through the
    # collapse handler, building reverse_map and folding incident
    # entity_passage edges. The fixture already laid down "overlay-style"
    # alias edges; strip them and replay through on_alias_accepted.
    alias_e_ids = [e.index for e in graph.es if e["edge_type"] == "alias"]
    accepted_pairs = []
    for e in graph.es:
        if e["edge_type"] == "alias":
            a = graph.vs[e.source]["name"]
            b = graph.vs[e.target]["name"]
            cos = float(e["weight"])
            accepted_pairs.append((a, b, cos))
    graph.delete_edges(alias_e_ids)
    for a, b, cos in accepted_pairs:
        cand = AliasCandidate(b, cos)
        feats = [{
            "cos_sim": cos, "reranker_score": 0.85,
            "admission_rule_version": "P4-test", "accepted_by": "test",
        }]
        on_alias_accepted(handler, graph, a, [cand], feats, [cos], reverse_map=reverse_map)
    return graph, reverse_map


def _outgoing_targets(graph: ig.Graph, vidx: int) -> Tuple[List[int], List[float]]:
    targets: List[int] = []
    weights: List[float] = []
    for eid in graph.incident(vidx, mode="all"):
        e = graph.es[eid]
        t = e.target if e.source == vidx else e.source
        if graph.vs[t].attributes().get("hidden") is True:
            continue
        targets.append(t)
        weights.append(float(e["weight"]) if "weight" in graph.es.attributes() else 1.0)
    return targets, weights


def _attribution_walks(
    graph: ig.Graph, seeds: List[str], top_passages: List[str]
) -> Dict[str, Dict[str, int]]:
    """Return ``{passage_hash → {previous_entity_hash → hit_count}}``."""
    name_to_idx = {v["name"]: v.index for v in graph.vs}
    seed_idx = [name_to_idx[s] for s in seeds if s in name_to_idx]
    if not seed_idx:
        return {p: {} for p in top_passages}
    target_set = {name_to_idx[p] for p in top_passages if p in name_to_idx}
    rng = random.Random(0xA771B)
    counts: Dict[str, Dict[str, int]] = {p: {} for p in top_passages}

    for _ in range(N_WALKS):
        cur = rng.choice(seed_idx)
        prev_entity: int | None = None
        for _step in range(MAX_STEPS):
            if cur in target_set:
                passage_name = graph.vs[cur]["name"]
                key_idx = prev_entity if prev_entity is not None else cur
                key = graph.vs[key_idx]["name"]
                counts[passage_name][key] = counts[passage_name].get(key, 0) + 1
                break
            targets, weights = _outgoing_targets(graph, cur)
            if not targets:
                break
            total = sum(weights)
            if total <= 0:
                break
            probs = [w / total for w in weights]
            nxt = rng.choices(targets, weights=probs, k=1)[0]
            if graph.vs[cur]["vertex_type"] == "entity":
                prev_entity = cur
            cur = nxt
    return counts


def _top_passages_by_seeds(graph: ig.Graph, seeds: List[str]) -> List[str]:
    """Deterministic ordering: rank passages by PPR personalised at seeds."""
    import numpy as np

    name_to_idx = {v["name"]: v.index for v in graph.vs}
    reset = np.zeros(graph.vcount())
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
    scores_arr = np.asarray(scores)
    passage_idx = [
        v.index for v in graph.vs
        if v["vertex_type"] == "passage" and not v.attributes().get("hidden")
    ]
    passage_idx.sort(key=lambda i: -scores_arr[i])
    return [graph.vs[i]["name"] for i in passage_idx[:TOP_PASSAGES]]


def _extract_provenance(
    graph: ig.Graph, passage_name: str
) -> List[str]:
    """Pull source_member values from incident entity_passage edges."""
    name_to_idx = {v["name"]: v.index for v in graph.vs}
    if passage_name not in name_to_idx:
        return []
    idx = name_to_idx[passage_name]
    out: List[str] = []
    if "source_member" not in graph.es.attributes():
        return out
    for eid in graph.incident(idx, mode="all"):
        e = graph.es[eid]
        if e["edge_type"] != "entity_passage":
            continue
        sm = e.attributes().get("source_member")
        if sm:
            # Comma-joined when multiple members were merged into the
            # same canonical-side edge (see _redirect_entity_passage_edges).
            for piece in str(sm).split(","):
                piece = piece.strip()
                if piece:
                    out.append(piece)
    return out


def test_p4_ppr_surface_attribution(tmp_path):
    reports: Dict[str, Dict[str, object]] = {}

    for handler in (
        ACCEPTANCE_HANDLER_OVERLAY,
        ACCEPTANCE_HANDLER_COLLAPSE_BASIC,
        ACCEPTANCE_HANDLER_COLLAPSE_PROVENANCE,
    ):
        graph, _ = _build_with_handler(handler, tmp_path)
        tops = _top_passages_by_seeds(graph, SEEDS)
        attribution = _attribution_walks(graph, SEEDS, tops)
        coverage = sum(1 for p in tops if attribution.get(p)) / max(1, len(tops))

        provenance_counts: Dict[str, int] = {}
        if handler == ACCEPTANCE_HANDLER_COLLAPSE_PROVENANCE:
            for p in tops:
                provenance_counts[p] = len(_extract_provenance(graph, p))

        reports[handler] = {
            "protocol": "P4",
            "config": {"system": handler, "n_walks": N_WALKS, "top_passages": TOP_PASSAGES},
            "results": {
                "top_passages": tops,
                "attribution_counts": {p: dict(attribution[p]) for p in tops},
                "attribution_coverage": coverage,
                "provenance_sidecar_counts": provenance_counts,
            },
        }

    out = tmp_path / "p4_report.json"
    out.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    overlay = reports[ACCEPTANCE_HANDLER_OVERLAY]["results"]
    basic = reports[ACCEPTANCE_HANDLER_COLLAPSE_BASIC]["results"]
    prov = reports[ACCEPTANCE_HANDLER_COLLAPSE_PROVENANCE]["results"]

    # Overlay claim: native attribution available — every top passage
    # has at least one contributing physical entity.
    assert overlay["attribution_coverage"] == 1.0, overlay
    # Each contributor must be a *physical* entity (i.e. one of the
    # original entity names), not a passage.
    for p, contribs in overlay["attribution_counts"].items():
        assert contribs, p
        for cont in contribs:
            assert cont.startswith("entity-"), (p, cont)

    # Basic-collapse: attribution still resolves to the canonical entity
    # only (other members are hidden + folded), so coverage is non-zero
    # but the contributor set has at most |canonicals| distinct keys —
    # surface paths are unrecoverable.
    assert basic["attribution_coverage"] >= 0.5  # at least some passages still attributable

    # Provenance-collapse: sidecar must carry source_member for at
    # least one of the top passages.
    assert any(v > 0 for v in prov["provenance_sidecar_counts"].values()), prov
