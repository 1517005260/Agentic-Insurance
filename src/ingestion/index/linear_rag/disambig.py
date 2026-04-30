"""Entity disambiguation: gradient top-k + mutual filter + cluster.

Pipeline for each new entity ``e`` (already embedded):

    1. faiss top-k against the entity store → candidates sorted by cos sim
    2. Walk down candidates; cut at the first relative similarity drop > g
       (default g=0.3, i.e. 30% drop between consecutive candidates)
    3. mutual top-k filter: keep candidate ``c`` only if ``e`` would also
       appear in ``c``'s top-k
    4. add (e, c) alias edges with weight = cos sim
    5. logical entities = connected components of the alias subgraph
       (computed lazily and cached to clusters.json)

The new entity is always added as a separate physical node (no merging) so
mistakes are reversible by deleting an alias edge.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import igraph as ig
import numpy as np

from storage import EmbeddingStore


ALIAS_EDGE_TYPE = "alias"
DEFAULT_TOP_K = 5
DEFAULT_GRADIENT = 0.3  # relative drop threshold (sim[i] - sim[i+1]) / sim[i]
# Absolute cosine-similarity floor on alias candidates. With sentence-context
# centroids the false-positive rate at ~0.5 is high (numbers, hierarchical
# entities); 0.85 keeps real synonym/variant pairs while dropping the bulk of
# noise. Tunable via ``LinearRAGConfig.alias_min_sim``.
DEFAULT_MIN_SIM = 0.85
# Slightly stricter floor for entities with too few mentions to form a stable
# context centroid — see :func:`_add_alias_edges_for_new_entities` in linear_rag.py.
DEFAULT_MIN_SIM_LOW_CONTEXT = 0.90


@dataclass
class AliasCandidate:
    hash_id: str
    score: float


def gradient_topk_candidates(
    query_embedding: np.ndarray,
    store: EmbeddingStore,
    k: int = DEFAULT_TOP_K,
    g: float = DEFAULT_GRADIENT,
    min_sim: float = DEFAULT_MIN_SIM,
    self_hash_id: Optional[str] = None,
) -> List[AliasCandidate]:
    """Top-k retrieval with absolute floor + gradient cutoff.

    Two complementary gates:

    * ``min_sim`` — absolute cosine floor. Drops obviously unrelated pairs
      that sneak through gradient cutoff when the relative drop is mild.
    * gradient cutoff — among the candidates that pass the floor, take a
      prefix where consecutive similarity drops stay below ``g``.

    ``self_hash_id`` excludes the entity itself from its own neighborhood.
    """
    if len(store) == 0:
        return []
    raw = store.topk(query_embedding, k + 1)
    cands = [
        AliasCandidate(h, s)
        for h, s in raw
        if h != self_hash_id and s >= min_sim
    ][:k]
    if not cands:
        return []

    cut = len(cands)
    for i in range(len(cands) - 1):
        s_i, s_next = cands[i].score, cands[i + 1].score
        if s_i <= 0:
            cut = i + 1
            break
        drop = (s_i - s_next) / s_i
        if drop > g:
            cut = i + 1
            break
    return cands[:cut]


def mutual_topk_filter(
    new_hash_id: str,
    new_embedding: np.ndarray,
    candidates: Sequence[AliasCandidate],
    store: EmbeddingStore,
    k: int = DEFAULT_TOP_K,
    min_sim: float = DEFAULT_MIN_SIM,
) -> List[AliasCandidate]:
    """Keep candidate ``c`` iff the new entity is in ``c``'s top-k as well.

    Pulls k+1 from each candidate's neighborhood since the candidate sits at
    rank 0 of its own search; after dropping self we have its k actual peers
    and check whether the new entity would have made that cutoff. The same
    absolute ``min_sim`` floor applies — even a "mutual top-k" pair below
    the floor is dropped.
    """
    if not candidates:
        return []
    surviving: List[AliasCandidate] = []
    for cand in candidates:
        cand_emb = store.get_embedding(cand.hash_id).reshape(1, -1)
        cand_topk = store.topk(cand_emb[0], k + 1)
        cand_peers = [(h, s) for h, s in cand_topk if h != cand.hash_id][:k]
        new_to_cand_sim = float(np.dot(new_embedding, cand_emb.T).flatten()[0])
        if new_to_cand_sim < min_sim:
            continue
        if not cand_peers:
            surviving.append(cand)
            continue
        kth_score = cand_peers[-1][1]
        if len(cand_peers) < k or new_to_cand_sim >= kth_score:
            surviving.append(cand)
    return surviving


def add_alias_edges(
    graph: ig.Graph,
    new_hash_id: str,
    candidates: Sequence[AliasCandidate],
) -> int:
    """Add weighted alias edges from ``new_hash_id`` to each candidate.

    Returns the count of edges actually added (skips existing edges).
    """
    if not candidates:
        return 0
    name_to_idx = {v["name"]: v.index for v in graph.vs if "name" in v.attributes()}
    if new_hash_id not in name_to_idx:
        return 0

    pairs: List[Tuple[int, int]] = []
    weights: List[float] = []
    edge_types: List[str] = []
    for cand in candidates:
        if cand.hash_id not in name_to_idx:
            continue
        u = name_to_idx[new_hash_id]
        v = name_to_idx[cand.hash_id]
        if graph.are_connected(u, v):
            continue
        pairs.append((u, v))
        weights.append(float(cand.score))
        edge_types.append(ALIAS_EDGE_TYPE)

    if not pairs:
        return 0
    start = graph.ecount()
    graph.add_edges(pairs)
    for offset, (w, t) in enumerate(zip(weights, edge_types)):
        graph.es[start + offset]["weight"] = w
        graph.es[start + offset]["edge_type"] = t
    return len(pairs)


def compute_clusters(graph: ig.Graph) -> List[Dict[str, object]]:
    """Connected components on the alias subgraph → logical entities.

    Returns a list of cluster dicts. ``canonical`` is the longest member
    surface. Only clusters with ≥2 members are returned (singletons are
    implicit — every physical entity not appearing in this list is its own
    logical entity).
    """
    if graph.ecount() == 0:
        return []
    if "edge_type" not in graph.es.attributes():
        return []
    alias_edges = [e.index for e in graph.es if e["edge_type"] == ALIAS_EDGE_TYPE]
    if not alias_edges:
        return []
    sub = graph.subgraph_edges(alias_edges, delete_vertices=True)
    components = sub.connected_components()

    clusters: List[Dict[str, object]] = []
    for cid, members in enumerate(components):
        names = [sub.vs[i]["name"] for i in members]
        if len(names) < 2:
            continue
        contents = [
            (sub.vs[members[j]].attributes().get("content") or names[j], names[j])
            for j in range(len(members))
        ]
        canonical_text, _ = max(contents, key=lambda t: len(t[0]))
        clusters.append(
            {
                "id": f"c_{cid:04d}",
                "members": names,
                "canonical": canonical_text,
            }
        )
    return clusters


def write_clusters(
    path: Path,
    clusters: List[Dict[str, object]],
    alias_edge_count: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "alias_edge_count": int(alias_edge_count),
        "clusters": clusters,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def invalidate_clusters(path: Path) -> None:
    if path.exists():
        path.unlink()


def get_clusters(graph: ig.Graph, cache_path: Path) -> List[Dict[str, object]]:
    """Lazy-loaded clusters: return cached if fresh, else compute + persist.

    "Fresh" = the cache file exists. Any structural change (alias edge
    added or removed, file removed) calls :func:`invalidate_clusters` so
    the next read recomputes.
    """
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return payload.get("clusters", [])
        except Exception:
            pass  # fall through to recompute on parse error
    clusters = compute_clusters(graph)
    alias_edge_count = (
        sum(1 for e in graph.es if e.attributes().get("edge_type") == ALIAS_EDGE_TYPE)
        if "edge_type" in graph.es.attributes()
        else 0
    )
    write_clusters(cache_path, clusters, alias_edge_count=alias_edge_count)
    return clusters
