"""Entity disambiguation: dual-query gradient top-k + cluster.

Pipeline for each new entity ``e`` (already embedded):

    1. Two faiss top-k queries against the entity store:
       - bare-surface query (symmetric: query and store are both
         bare-surface embeddings → same space, true cos sim)
       - mention-context centroid query (semantic recall path)
       Their results are merged by max cosine similarity per candidate.
       The bare-surface arm carries character-level variants
       (singular/plural, abbreviation, light reordering); the centroid
       arm carries semantic synonyms. Symmetric bare-vs-bare scoring
       removes the need for a separate mutual top-k step.
    2. Apply absolute floor ``alias_min_sim`` to the merged list.
    3. Walk down candidates; cut at the first relative similarity drop
       > g (default g=0.3, i.e. 30% drop between consecutive candidates).
    4. Add (e, c) alias edges with weight = max cos sim from step 1.
    5. Logical entities = connected components of the alias subgraph
       (computed lazily and cached to clusters.json).

The new entity is always added as a separate physical node (no merging) so
mistakes are reversible by deleting an alias edge.
"""
import json
import re
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


def merge_topk_candidates(
    *candidate_lists: Sequence[AliasCandidate],
) -> List[AliasCandidate]:
    """Merge multiple top-k result lists, keeping the max score per hash_id.

    Used to combine the bare-surface and centroid query result lists in
    the dual-query recall path: a candidate that surfaces in either
    list at sufficient similarity should be considered, and we keep
    whichever score is higher so the gradient cutoff downstream sees
    the strongest signal.
    """
    by_hash: Dict[str, float] = {}
    for cands in candidate_lists:
        for c in cands:
            prev = by_hash.get(c.hash_id)
            if prev is None or c.score > prev:
                by_hash[c.hash_id] = c.score
    merged = [AliasCandidate(h, s) for h, s in by_hash.items()]
    merged.sort(key=lambda c: c.score, reverse=True)
    return merged


def reranker_veto(
    anchor_text: str,
    candidates: Sequence[AliasCandidate],
    store: EmbeddingStore,
    *,
    threshold: float,
    instruction: str,
) -> List[AliasCandidate]:
    """Drop candidates whose pairwise reranker score is below ``threshold``.

    Cross-encoder pairwise score is used as a **low-confidence veto** —
    the rerank AUC on cold-start short-surface ER measured around 0.66,
    enough to filter ordered-tier false merges (`option 1` vs `option 2`)
    and a few range/scope variants, but not high enough to use the score
    as an identity classifier. High scores do not certify alias status
    on their own; they only mean "the reranker did not veto". The
    upstream cos-sim + gradient cutoff + composite gate still set the
    real admission criteria.

    Surface-only input (not surface + mention) — empirically the model
    treats long sentence context as semantic-relevance signal, which
    dilutes the identity decision. The hardened instruction is what
    forces the yes/no head toward identity rather than relevance.
    """
    if not candidates:
        return []
    # Lazy import — keeps `disambig.py` cheap when only the gradient
    # path is exercised (e.g. test fixtures without a downloaded
    # reranker checkpoint).
    from model_client import get_cached_rerank_client

    client = get_cached_rerank_client()
    pairs = [
        (anchor_text, store.get_text(c.hash_id)) for c in candidates
    ]
    scores = client.score_pairs(pairs, instruction=instruction)
    return [c for c, s in zip(candidates, scores) if s >= threshold]


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


# Surface-quality scoring — used to pick a cluster's ``canonical``
# representative. Lower (more negative) = noisier; we pick the
# **highest** score in a cluster as the canonical.
#
# The previous "longest surface" rule was empirically wrong on OCR
# noise: spaCy's longest spans are usually mis-bounded chains
# (``A(c1)、B(c2)、C(c3)``) or sentence fragments (``…保单。``),
# both of which beat the clean prefix on length but are the worst
# possible canonical name. The scoring below penalises the structural
# defects that those bad spans exhibit, then breaks ties by
# preferring shorter (cleaner) surfaces.

# Trailing punctuation / whitespace / dangling **opening** bracket that
# strongly indicates a mid-sentence cut. Closing brackets ``)`` ``）``
# are intentionally NOT in this set: a balanced SKU like
# ``"万通危疾加护保(优越版)"`` legitimately ends with ``)`` and would
# otherwise lose a -2 penalty to a cleanup-stripped sentence fragment
# like ``"…保单"`` (whose trailing ``。`` ``cleanup`` already removed).
# Bracket imbalance is handled separately by ``_bracket_imbalance``.
_TRAILING_JUNK_RE = re.compile(r"[。\.,，;；:：、!?！？\s(（]+$")
# A list separator inside the surface — the surface is a chain of
# multiple mentions glued together by spaCy.
_LIST_SEP_INTERIOR_RE = re.compile(r"[、；;•｜|，]")
# Bracket counts (both half- and full-width) for balance check.
_OPEN_BRACKETS = ("(", "（")
_CLOSE_BRACKETS = (")", "）")


def _bracket_imbalance(s: str) -> int:
    """Absolute difference between # opens and # closes (any width)."""
    opens = sum(s.count(c) for c in _OPEN_BRACKETS)
    closes = sum(s.count(c) for c in _CLOSE_BRACKETS)
    return abs(opens - closes)


def _open_bracket_count(s: str) -> int:
    return sum(s.count(c) for c in _OPEN_BRACKETS)


# Conjunction words that can glue multiple mentions into one span but
# also legitimately appear inside organisation names (e.g.
# "保险及再保险公司"). The catalog splitter (``split_catalog_mentions``)
# intentionally does not split on these. The composite gate uses them
# only as a corroborating signal alongside multiple bracket pairs.
_COMPOSITE_CONJUNCTION_RE = re.compile(r"或|及|与")
# Half-width comma — left out of the catalog splitter because it can
# appear inside English company names (``"Inc., Ltd."``). Used here
# only when paired with multiple bracket pairs as a composite signal.
_HALF_WIDTH_COMMA_RE = re.compile(r",")


def is_composite_surface(text: Optional[str]) -> bool:
    """Return True when the surface looks like multiple mentions glued
    into one span and the safe split rules couldn't decompose it.

    Used as an admission gate before alias-edge generation: composite
    surfaces are a mixture centroid in embedding space and pull in
    cleanly-named neighbours, producing the c_0000-style "garbage
    bucket" clusters we observed empirically. Skipping them at the
    alias-edge stage prevents pollution-via-transitivity (entity A
    aliased to composite C aliased to entity B → A and B end up in
    the same cluster despite never being aliased directly).

    Signals (any one is enough):

    1. Interior list separator (``、；;•｜|，``) — defensive: this
       should already have been split by ``split_catalog_mentions``;
       a surviving one means we missed it.
    2. Conjunction (``或/及/与``) **and** ≥2 bracket pairs — likely
       ``A(code1) 或 B(code2)``. Conjunctions alone are not enough
       (e.g. ``"保险及再保险公司"`` is a real org).
    3. Half-width comma **and** ≥2 bracket pairs — likely
       ``A(code1),B(code2)``. Comma alone is not enough (English
       company names).
    4. ≥3 open brackets — a chain like ``A(c1)(c2) B(c3)`` even
       without a separator we recognise.
    """
    if not text:
        return False
    if _LIST_SEP_INTERIOR_RE.search(text):
        return True
    open_count = _open_bracket_count(text)
    if open_count >= 2 and _COMPOSITE_CONJUNCTION_RE.search(text):
        return True
    if open_count >= 2 and _HALF_WIDTH_COMMA_RE.search(text):
        return True
    if open_count >= 3:
        return True
    return False


def surface_quality_score(surface: Optional[str]) -> float:
    """Higher = cleaner. Used to pick a canonical from a cluster.

    Components (penalties stack additively, lower is worse):

    * ``-3`` per bracket imbalance — dangling ``"万通危疾加护保("``.
    * ``-2`` per interior list separator — composite chain like
      ``"A、B、C"`` which should never have been one entity.
    * ``-2`` for trailing punctuation/whitespace/bracket — sentence
      fragment leakage like ``"…保单。"`` or ``"vip 环球医疗保("``.
    * ``-0.05 * len`` — mild length penalty so among equally-clean
      surfaces we prefer the shorter one (the family-level canonical
      ``"万通危疾加护保"`` over the SKU-tagged ``"…(优越版)(phps)"``).

    The mild length penalty is the only "preference" component; the
    other three penalise concrete structural defects that the
    composite-surface gate (P2) and the cleanup pipeline (P0) would
    have caught had spaCy bounded the span correctly.
    """
    if not surface:
        return -1e6
    score = 0.0
    score -= 3.0 * _bracket_imbalance(surface)
    if _LIST_SEP_INTERIOR_RE.search(surface):
        score -= 2.0
    if _TRAILING_JUNK_RE.search(surface):
        score -= 2.0
    score -= 0.05 * len(surface)
    return score


def compute_clusters(graph: ig.Graph) -> List[Dict[str, object]]:
    """Connected components on the alias subgraph → logical entities.

    Returns a list of cluster dicts. ``canonical`` is the cluster
    member with the highest :func:`surface_quality_score` — the
    cleanest surface. Empirically this gives canonicals like
    ``"万通危疾加护保"`` instead of the previous longest-surface rule
    which would pick the noisiest member (chained product list,
    sentence fragment, dangling-bracket token).

    Ties (same score) are broken deterministically by the member's
    insertion order so ``compute_clusters`` is reproducible across
    runs. Only clusters with ≥2 members are returned (singletons are
    implicit — every physical entity not appearing in this list is
    its own logical entity).
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
        # Argmax over surface_quality_score; tuple-as-key ties break by
        # insertion order (stable, reproducible).
        canonical_text, _ = max(
            ((text, name) for text, name in contents),
            key=lambda t: surface_quality_score(t[0]),
        )
        clusters.append(
            {
                "id": f"c_{cid:04d}",
                "members": names,
                "canonical": canonical_text,
            }
        )
    return clusters


# Cluster-cache schema version. Bump when ``compute_clusters`` /
# ``surface_quality_score`` change so older cache files (which encode
# the *previous* canonical-picker output) are silently invalidated and
# recomputed on next read.
CLUSTERS_CACHE_VERSION = 2


def write_clusters(
    path: Path,
    clusters: List[Dict[str, object]],
    alias_edge_count: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CLUSTERS_CACHE_VERSION,
        "alias_edge_count": int(alias_edge_count),
        "clusters": clusters,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def invalidate_clusters(path: Path) -> None:
    if path.exists():
        path.unlink()


def get_clusters(graph: ig.Graph, cache_path: Path) -> List[Dict[str, object]]:
    """Lazy-loaded clusters: return cached if fresh, else compute + persist.

    "Fresh" = file exists AND ``version`` matches the current
    ``CLUSTERS_CACHE_VERSION``. Older versions are silently dropped
    and recomputed — this is the migration path for the canonical-
    picker change in v2 (otherwise an upgraded binary would keep
    serving stale longest-surface canonicals from a v1 cache).
    """
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_version = payload.get("version")
            if cached_version == CLUSTERS_CACHE_VERSION:
                return payload.get("clusters", [])
            # Version mismatch (or missing) — fall through to recompute.
        except Exception:
            pass  # parse error — fall through to recompute
    clusters = compute_clusters(graph)
    alias_edge_count = (
        sum(1 for e in graph.es if e.attributes().get("edge_type") == ALIAS_EDGE_TYPE)
        if "edge_type" in graph.es.attributes()
        else 0
    )
    write_clusters(cache_path, clusters, alias_edge_count=alias_edge_count)
    return clusters
