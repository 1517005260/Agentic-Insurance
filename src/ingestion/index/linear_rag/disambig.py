"""Entity disambiguation: 0-LLM blocking → matching → de-percolation.

Run as a flush-time batch over the entity store (orchestration in
``LinearRAG._resolve_entities``); the functions here are pure helpers:

    1. RECALL (blocking): two faiss top-k queries per entity against the
       entity store — bare-surface and mention-context centroid — give the
       candidate neighbourhood (``gradient_topk_candidates``). ANN search.
    2. SYMMETRIZE: keep only reciprocal-kNN pairs (``mutual_knn_pairs``) so a
       one-directional hub pull cannot chain a giant component.
    3. MATCH (precision): a gate of a DIFFERENT signal class than recall —
       IDF-weighted lexical token overlap (``build_surface_idf`` /
       ``idf_weighted_overlap``) for surface-similar pairs, plus a relational
       co-occurrence veto (entities sharing a passage are distinct entities).
    4. Add (e, c) alias edges (overlay) with weight from ``propagation_policy``,
       capped at the top-L per entity.
    5. Logical entities = Leiden communities of the alias subgraph
       (``compute_clusters``), a recomputable derived view over immutable edges.

The new entity is always a separate physical node (no merging) so mistakes are
reversible by deleting an alias edge.
"""
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import igraph as ig
import numpy as np
import regex  # \p{Han} Unicode property class (third-party, as in normalize.py)

from storage import EmbeddingStore


ALIAS_EDGE_TYPE = "alias"
DEFAULT_TOP_K = 5
DEFAULT_GRADIENT = 0.3  # relative drop threshold (sim[i] - sim[i+1]) / sim[i]
# Absolute cosine-similarity floor on alias candidates. With sentence-context
# centroids the false-positive rate at ~0.5 is high (numbers, hierarchical
# entities); 0.85 keeps real synonym/variant pairs while dropping the bulk of
# noise. Tunable via ``LinearRAGConfig.alias_min_sim``.
DEFAULT_MIN_SIM = 0.85

# Admission rule version — bump whenever the admission pipeline (recall
# top-k, mutual-kNN, IDF lexical gate, co-occurrence veto, outdegree cap)
# changes so each edge's ``features_json["admission_rule_version"]`` records
# which gate set accepted it. Audit and rule-regression depend on this string.
ADMISSION_RULE_VERSION = "D4-idf+mutualknn+cooccur-v1"

@dataclass
class AliasCandidate:
    hash_id: str
    # Max cosine across the recall arms (bare-surface / mention-centroid) that
    # surfaced this candidate; drives the edge weight via ``propagation_policy``.
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


# =============================================================== lexical signal
# Precision matcher of a DIFFERENT signal class than the embedding recall step
# (Fellegi-Sunter / record linkage): IDF-weighted token overlap. A single
# semantic threshold cannot separate template collisions ("3rd Baron Acton" vs
# "3rd Baroness Herbert", high cosine, distinct) from true variants; the lexical
# signal can, because corpus-frequent template tokens carry little IDF weight
# and rare head tokens carry most of it.

_HAN_RUN_RE = regex.compile(r"\p{Han}+")
# Non-Han word tokens: maximal runs of word characters (letters / digits /
# underscore, any script) excluding Han. ``[^\W\p{Han}]`` = "word char AND not
# Han", so punctuation and whitespace are dropped — ``"dark river (2017 film)"``
# → ``dark river 2017 film`` rather than gluing ``(2017`` / ``film)``. Script-
# agnostic (Cyrillic, Greek, accented Latin all count) and never corpus-tuned.
_LATIN_TOKEN_RE = regex.compile(r"[^\W\p{Han}]+")


def tokenize_surface(surface: Optional[str], *, han_ngram: int = 2) -> List[str]:
    """Language-agnostic tokens for the lexical-overlap signal.

    Latin/Cyrillic/digit runs → whitespace-delimited word tokens. Han runs have
    no whitespace, so they are cut into character ``han_ngram``-grams (default
    bigrams) — bigrams keep enough identity to tell ``万通危疾`` from ``富饶万家``
    while staying segmenter-free. Tokens are namespaced (``w:`` / ``h:``) so a
    Latin word can never collide with a Han n-gram in the IDF table.
    """
    if not surface:
        return []
    tokens: List[str] = ["w:" + m for m in _LATIN_TOKEN_RE.findall(surface)]
    n = max(1, int(han_ngram))
    for run in _HAN_RUN_RE.findall(surface):
        if len(run) <= n:
            tokens.append("h:" + run)
        else:
            tokens.extend("h:" + run[i : i + n] for i in range(len(run) - n + 1))
    return tokens


def smoothed_idf(df: int, n: int) -> float:
    """BM25-style smoothed IDF with a positive floor: ``ln((n+1)/(df+0.5))``.

    A plain ``ln(n/df)`` gives a token present in EVERY surface weight 0, so a
    true variant whose shared tokens are all corpus-frequent (e.g. an insurance
    corpus where ``保险`` appears in most product names) underflows to zero
    overlap and is wrongly rejected. The smoothing keeps even a ubiquitous
    token at a small POSITIVE weight while rare head tokens still dominate, so
    the relative ordering (and template-collision rejection) is preserved.
    """
    if n <= 0:
        return 0.0
    return math.log((n + 1.0) / (df + 0.5))


def build_surface_idf(
    surfaces: Sequence[str], *, han_ngram: int = 2
) -> Dict[str, float]:
    """Corpus smoothed-IDF over entity surfaces (each surface = one document).

    ``df[t]`` = number of distinct surfaces containing token ``t``, ``N`` =
    surface count; weight = :func:`smoothed_idf`. Frequent template tokens
    (``baron`` / ``war`` / ``3rd``) get a small weight; rare head tokens
    (``acton``) a large one. One pass: O(Σ tokens). The incremental build path
    maintains ``df`` itself and calls :func:`smoothed_idf` directly.
    """
    df: Dict[str, int] = {}
    n = 0
    for s in surfaces:
        n += 1
        for t in set(tokenize_surface(s, han_ngram=han_ngram)):
            df[t] = df.get(t, 0) + 1
    if n == 0:
        return {}
    return {t: smoothed_idf(c, n) for t, c in df.items()}


def idf_weighted_overlap(
    tokens_a: Sequence[str], tokens_b: Sequence[str], idf: Dict[str, float]
) -> float:
    """IDF-weighted Jaccard over two token sets (record-linkage matcher).

    ``Σ_{t∈A∩B} idf[t] / Σ_{t∈A∪B} idf[t]`` — overlap on distinctive (high-IDF)
    tokens dominates; sharing only frequent template tokens scores near 0.
    Returns 0.0 when either side is empty or the union carries no IDF mass.
    """
    sa, sb = set(tokens_a), set(tokens_b)
    if not sa or not sb:
        return 0.0
    denom = sum(idf.get(t, 0.0) for t in (sa | sb))
    if denom <= 0.0:
        return 0.0
    num = sum(idf.get(t, 0.0) for t in (sa & sb))
    return num / denom


def add_alias_edges(
    graph: ig.Graph,
    new_hash_id: str,
    candidates: Sequence[AliasCandidate],
    *,
    features_list: Optional[Sequence[Dict[str, Any]]] = None,
    w_prop_list: Optional[Sequence[float]] = None,
    name_to_idx: Optional[Dict[str, int]] = None,
) -> int:
    """Add alias edges from ``new_hash_id`` to each candidate.

    Edge attribute layout:

    * ``weight``        — propagation weight ``w_prop`` (drives PPR).
    * ``edge_type``     — ``"alias"``.
    * ``features_json`` — JSON-stringified per-edge features dict
      (cos_sim / idf_overlap / admission_rule_version / accepted_by /
      evidence). GraphML cannot store dict attributes natively, so the
      single JSON string is the audit sidecar.

    A caller that omits features/w_prop falls back to
    ``weight = cand.score`` (policy=cos equivalence), so the explicit
    feature path is optional for tests / scripts. ``name_to_idx`` (name →
    vertex index) may be passed in by a batch caller built ONCE for the whole
    ER pass; omitting it rebuilds the map here (O(V) per call) — passing it is
    what keeps the batch O(N) instead of O(N·V).
    """
    if not candidates:
        return 0
    if name_to_idx is None:
        name_to_idx = {v["name"]: v.index for v in graph.vs if "name" in v.attributes()}
    if new_hash_id not in name_to_idx:
        return 0

    pairs: List[Tuple[int, int]] = []
    weights: List[float] = []
    edge_types: List[str] = []
    features_payloads: List[str] = []
    for i, cand in enumerate(candidates):
        if cand.hash_id not in name_to_idx:
            continue
        u = name_to_idx[new_hash_id]
        v = name_to_idx[cand.hash_id]
        if graph.are_adjacent(u, v):
            continue
        pairs.append((u, v))
        if w_prop_list is not None:
            w_prop = float(w_prop_list[i])
        else:
            w_prop = float(cand.score)
        weights.append(w_prop)
        edge_types.append(ALIAS_EDGE_TYPE)
        if features_list is not None:
            features = dict(features_list[i])
        else:
            features = {
                "cos_sim": float(cand.score),
                "admission_rule_version": ADMISSION_RULE_VERSION,
                "accepted_by": "idf_mutualknn_er",
            }
        features_payloads.append(json.dumps(features, ensure_ascii=False))

    if not pairs:
        return 0
    start = graph.ecount()
    graph.add_edges(pairs)
    for offset, (w, t, fp) in enumerate(zip(weights, edge_types, features_payloads)):
        graph.es[start + offset]["weight"] = w
        graph.es[start + offset]["edge_type"] = t
        graph.es[start + offset]["features_json"] = fp
        # ``w_prop`` mirrors ``weight`` so older readers that look up
        # ``weight`` keep working; new readers can prefer ``w_prop``
        # when distinguishing propagation strength from audit-time
        # features is meaningful (e.g. when policy != cos).
        graph.es[start + offset]["w_prop"] = w
    return len(pairs)


# Surface-quality scoring — used to pick a cluster's ``canonical``
# representative. Lower (more negative) = noisier; we pick the
# **highest** score in a cluster as the canonical.
#
# A "longest surface" rule fails on OCR noise: the NER model's longest
# spans are usually mis-bounded chains (``A(c1)、B(c2)、C(c3)``) or sentence
# fragments (``…保单。``), both of which beat the clean prefix on length
# but are the worst possible canonical name. The scoring below
# penalises the structural defects that those bad spans exhibit, then
# breaks ties by preferring shorter (cleaner) surfaces.

# Trailing punctuation / whitespace / dangling **opening** bracket that
# strongly indicates a mid-sentence cut. Closing brackets ``)`` ``）``
# are intentionally NOT in this set: a balanced SKU like
# ``"万通危疾加护保(优越版)"`` legitimately ends with ``)`` and would
# otherwise lose a -2 penalty to a cleanup-stripped sentence fragment
# like ``"…保单"`` (whose trailing ``。`` ``cleanup`` already removed).
# Bracket imbalance is handled separately by ``_bracket_imbalance``.
_TRAILING_JUNK_RE = re.compile(r"[。\.,，;；:：、!?！？\s(（]+$")
# A list separator inside the surface — the surface is a chain of
# multiple mentions glued together.
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
    cleanly-named neighbours, producing c_0000-style "garbage
    bucket" clusters. Skipping them at the
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
    have caught had the NER span been bounded correctly.
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


CLUSTER_ALGORITHMS = {"connected_components", "leiden_cpm"}


def compute_clusters(
    graph: ig.Graph,
    *,
    algorithm: str = "connected_components",
    leiden_resolution: float = 0.05,
    leiden_weighted: bool = True,
) -> List[Dict[str, object]]:
    """Partition the alias subgraph into logical entities (derived view).

    The alias subgraph is never mutated — clusters are a *recomputable
    derived view* over immutable alias edges, so reversibility / P1 / P4
    hold for any ``algorithm``. Two partitioners:

    * ``connected_components`` — raw transitive closure. Single-linkage;
      percolates to a giant component at open-domain scale (a phase
      transition in N, not a tunable tail). Available as an
      alternative partitioner.
    * ``leiden_cpm`` — Leiden on the Constant-Potts-Model objective
      (igraph ``community_leiden``). Chaining-resistant by construction
      (well-connected, resolution-bounded communities), the principled
      replacement the ER / cross-doc-coref literature converges on.
      ``leiden_resolution`` is the granularity knob (higher → smaller,
      tighter clusters); ``leiden_weighted`` uses the alias edge weight
      (cos-derived propagation strength) so strong aliases resist being
      cut.

    ``canonical`` is the cluster member with the highest
    :func:`surface_quality_score` (cleanest surface). Ties break by
    insertion order so the output is reproducible. Only clusters with
    ≥2 members are returned (singletons are implicit).
    """
    if algorithm not in CLUSTER_ALGORITHMS:
        raise ValueError(f"compute_clusters: unknown algorithm {algorithm!r}")
    if graph.ecount() == 0:
        return []
    if "edge_type" not in graph.es.attributes():
        return []
    alias_edges = [e.index for e in graph.es if e["edge_type"] == ALIAS_EDGE_TYPE]
    if not alias_edges:
        return []
    sub = graph.subgraph_edges(alias_edges, delete_vertices=True)
    if algorithm == "connected_components":
        components = sub.connected_components()
    else:  # leiden_cpm
        weights = (
            sub.es["weight"]
            if leiden_weighted and "weight" in sub.es.attributes()
            else None
        )
        components = sub.community_leiden(
            objective_function="CPM",
            resolution=leiden_resolution,
            weights=weights,
            n_iterations=-1,
        )

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
# ``surface_quality_score`` change so older cache files are silently
# invalidated and recomputed on next read.
CLUSTERS_CACHE_VERSION = 3


def write_clusters(
    path: Path,
    clusters: List[Dict[str, object]],
    alias_edge_count: int = 0,
    *,
    algorithm: str = "connected_components",
    leiden_resolution: float = 0.05,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CLUSTERS_CACHE_VERSION,
        "alias_edge_count": int(alias_edge_count),
        # Recorded for observability / audit — which partitioner produced
        # this cache. Freshness is still version-keyed (a deliberate
        # algorithm flip clears the cache, like any schema-version bump),
        # so multiple readers never thrash-recompute each other.
        "algorithm": algorithm,
        "leiden_resolution": float(leiden_resolution),
        "clusters": clusters,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def invalidate_clusters(path: Path) -> None:
    if path.exists():
        path.unlink()


def get_clusters(
    graph: ig.Graph,
    cache_path: Path,
    *,
    algorithm: str = "connected_components",
    leiden_resolution: float = 0.05,
    leiden_weighted: bool = True,
) -> List[Dict[str, object]]:
    """Lazy-loaded clusters: return cached if fresh, else compute + persist.

    "Fresh" = file exists AND ``version`` matches the current
    ``CLUSTERS_CACHE_VERSION``. Older versions are silently dropped and
    recomputed. Freshness is intentionally version-keyed
    only, NOT keyed on ``algorithm``/``leiden_resolution``: in a
    deployment the partitioner is fixed by config, and a deliberate flip
    clears the cache (ingest invalidates on every alias-edge add), so
    making freshness param-sensitive would only let concurrent readers
    with momentarily-divergent config thrash-recompute each other.
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
    clusters = compute_clusters(
        graph,
        algorithm=algorithm,
        leiden_resolution=leiden_resolution,
        leiden_weighted=leiden_weighted,
    )
    alias_edge_count = (
        sum(1 for e in graph.es if e.attributes().get("edge_type") == ALIAS_EDGE_TYPE)
        if "edge_type" in graph.es.attributes()
        else 0
    )
    write_clusters(
        cache_path,
        clusters,
        alias_edge_count=alias_edge_count,
        algorithm=algorithm,
        leiden_resolution=leiden_resolution,
    )
    return clusters


# ============================================================================
# Reverse map (collapse-mode canonicalisation)
# ============================================================================

REVERSE_MAP_VERSION = 1


def load_reverse_map(path: Path) -> Dict[str, str]:
    """Load ``{other_hash → canonical_hash}`` from disk; empty dict when missing.

    Used in collapse mode to:

    * Redirect alias acceptances that arrive against an already-collapsed
      head (chain compaction: e_old has been folded into e_canon, so a
      fresh alias from e_new lands on e_canon directly).
    * Resolve citations to a hidden (collapsed) physical hash back to its
      canonical at query time.
    """
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("version") != REVERSE_MAP_VERSION:
        return {}
    mapping = payload.get("map") or {}
    if not isinstance(mapping, dict):
        return {}
    return {str(k): str(v) for k, v in mapping.items()}


def follow_reverse_map(hash_id: str, reverse_map: Dict[str, str]) -> str:
    """Walk the chain to the canonical (path-compressed locally)."""
    seen: set = set()
    cur = hash_id
    while cur in reverse_map and cur not in seen:
        seen.add(cur)
        nxt = reverse_map[cur]
        if nxt == cur:
            break
        cur = nxt
    return cur


# ============================================================================
# Acceptance handlers (overlay / collapse_basic / collapse_provenance)
# ============================================================================

ACCEPTANCE_HANDLER_OVERLAY = "overlay"
ACCEPTANCE_HANDLER_COLLAPSE_BASIC = "collapse_basic"
ACCEPTANCE_HANDLER_COLLAPSE_PROVENANCE = "collapse_provenance"
ACCEPTANCE_HANDLERS = {
    ACCEPTANCE_HANDLER_OVERLAY,
    ACCEPTANCE_HANDLER_COLLAPSE_BASIC,
    ACCEPTANCE_HANDLER_COLLAPSE_PROVENANCE,
}


def _choose_canonical(
    graph: ig.Graph, a_idx: int, b_idx: int
) -> Tuple[int, int]:
    """Return ``(canonical_idx, other_idx)`` by surface_quality_score.

    Picks the cleanest surface as canonical. Ties broken by hash_id
    string order for reproducibility (igraph vertex index is build-time
    insertion order, which is not stable across runs).
    """
    a_attrs = graph.vs[a_idx].attributes()
    b_attrs = graph.vs[b_idx].attributes()
    a_surface = a_attrs.get("content") or a_attrs.get("name") or ""
    b_surface = b_attrs.get("content") or b_attrs.get("name") or ""
    sa = surface_quality_score(a_surface)
    sb = surface_quality_score(b_surface)
    if sa > sb:
        return a_idx, b_idx
    if sb > sa:
        return b_idx, a_idx
    # Score tie → deterministic by name.
    a_name = a_attrs.get("name") or ""
    b_name = b_attrs.get("name") or ""
    if a_name <= b_name:
        return a_idx, b_idx
    return b_idx, a_idx


def _on_alias_accepted_overlay(
    graph: ig.Graph,
    new_hash_id: str,
    candidates: Sequence[AliasCandidate],
    features_list: Sequence[Dict[str, Any]],
    w_prop_list: Sequence[float],
    *,
    name_to_idx: Optional[Dict[str, int]] = None,
    **_,
) -> int:
    """Overlay handler: just add alias edges. The default reversible path."""
    return add_alias_edges(
        graph,
        new_hash_id,
        candidates,
        features_list=features_list,
        w_prop_list=w_prop_list,
        name_to_idx=name_to_idx,
    )


def _redirect_entity_passage_edges(
    graph: ig.Graph,
    other_idx: int,
    canonical_idx: int,
    *,
    carry_provenance: bool,
    alias_features: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Move every entity_passage edge incident to ``other_idx`` onto
    ``canonical_idx``; merge duplicate ``(canonical, dst)`` edges by
    summing weights. Returns the new-edge indices added.

    When ``carry_provenance`` is True, each rewritten edge carries
    ``source_member`` = old vertex name and ``alias_features_json`` =
    the alias features sidecar (collapse-provenance mode).
    """
    if "edge_type" not in graph.es.attributes():
        return []
    other_name = graph.vs[other_idx]["name"]
    incident = graph.incident(other_idx, mode="all")
    new_pairs: List[Tuple[int, int]] = []
    new_weights: List[float] = []
    new_sources: List[Optional[str]] = []
    edges_to_delete: List[int] = []
    # Pre-index existing canonical-side targets so we can merge on collision.
    existing: Dict[int, int] = {}
    for e in graph.incident(canonical_idx, mode="all"):
        edge = graph.es[e]
        if edge["edge_type"] != "entity_passage":
            continue
        target = edge.target if edge.source == canonical_idx else edge.source
        existing[target] = e
    # In provenance mode multiple members can redirect onto the same
    # (canonical, dst) edge. The sidecar must record (a) every member
    # that contributed, (b) the alias_features dict that admitted each
    # contribution, and (c) each member's individual weight — without
    # all three the sidecar is no longer self-contained for any
    # post-collision edge. GraphML has no list attr, so we store the
    # member list as ``source_member`` CSV and the per-member features
    # as a JSON-encoded array under ``alias_features_json``.
    merge_provenance: Dict[int, List[Tuple[str, float, Optional[Dict[str, Any]]]]] = {}
    for e in incident:
        edge = graph.es[e]
        if edge["edge_type"] != "entity_passage":
            edges_to_delete.append(e)
            continue
        target = edge.target if edge.source == other_idx else edge.source
        weight = float(edge["weight"]) if "weight" in graph.es.attributes() else 1.0
        if target in existing:
            existing_eid = existing[target]
            # Merge: bump the canonical-side edge's weight.
            graph.es[existing_eid]["weight"] = (
                float(graph.es[existing_eid]["weight"]) + weight
            )
            if carry_provenance:
                merge_provenance.setdefault(existing_eid, []).append(
                    (other_name, weight, alias_features)
                )
        else:
            new_pairs.append((canonical_idx, target))
            new_weights.append(weight)
            new_sources.append(other_name if carry_provenance else None)
        edges_to_delete.append(e)
    # Flush merge-side provenance BEFORE delete_edges (the eids change
    # after deletion, but ``existing`` was captured before that).
    if carry_provenance:
        for eid, contributions in merge_provenance.items():
            existing_attrs = graph.es[eid].attributes()
            prev_sm = existing_attrs.get("source_member") or ""
            existing_members = [m for m in str(prev_sm).split(",") if m]
            prev_fj = existing_attrs.get("alias_features_json") or ""
            try:
                existing_records = json.loads(prev_fj) if prev_fj else []
                if not isinstance(existing_records, list):
                    # Single-dict shape (not a list): wrap it.
                    existing_records = [existing_records]
            except (ValueError, TypeError):
                existing_records = []
            new_members = [c[0] for c in contributions]
            new_records = [
                {"source_member": m, "weight": w, "features": f}
                for m, w, f in contributions
            ]
            graph.es[eid]["source_member"] = ",".join(existing_members + new_members)
            graph.es[eid]["alias_features_json"] = json.dumps(
                existing_records + new_records, ensure_ascii=False
            )
    if edges_to_delete:
        graph.delete_edges(edges_to_delete)
    added: List[int] = []
    if new_pairs:
        start = graph.ecount()
        graph.add_edges(new_pairs)
        # Even on first redirection store the per-member record as a
        # single-element list so future collisions append uniformly.
        for offset, (w, src) in enumerate(zip(new_weights, new_sources)):
            idx = start + offset
            graph.es[idx]["weight"] = w
            graph.es[idx]["edge_type"] = "entity_passage"
            if carry_provenance and src is not None:
                graph.es[idx]["source_member"] = src
                record = {"source_member": src, "weight": w, "features": alias_features}
                graph.es[idx]["alias_features_json"] = json.dumps(
                    [record], ensure_ascii=False
                )
            added.append(idx)
    return added


def _on_alias_accepted_collapse(
    graph: ig.Graph,
    new_hash_id: str,
    candidates: Sequence[AliasCandidate],
    features_list: Sequence[Dict[str, Any]],
    w_prop_list: Sequence[float],
    *,
    reverse_map: Dict[str, str],
    carry_provenance: bool,
    name_to_idx: Optional[Dict[str, int]] = None,
    **_,
) -> int:
    """Collapse handler (basic + provenance variants).

    Per accepted candidate:

      1. Resolve both endpoints through ``reverse_map`` (chain compaction).
      2. Choose canonical by surface quality.
      3. Redirect ``other``'s entity_passage edges to canonical (with
         optional ``source_member`` + alias_features_json sidecar).
      4. Delete the other's remaining incident edges.
      5. Mark ``other`` vertex ``hidden=True``.
      6. Record ``reverse_map[other] = canonical``.

    No alias edges are created — the cluster contract under collapse is
    "one canonical absorbs the rest". Returns the count of accepted
    (collapsed) pairs.
    """
    if not candidates:
        return 0
    if name_to_idx is None:
        name_to_idx = {v["name"]: v.index for v in graph.vs if "name" in v.attributes()}
    accepted = 0
    for i, cand in enumerate(candidates):
        new_canon = follow_reverse_map(new_hash_id, reverse_map)
        old_canon = follow_reverse_map(cand.hash_id, reverse_map)
        if new_canon == old_canon:
            continue
        if new_canon not in name_to_idx or old_canon not in name_to_idx:
            continue
        a_idx = name_to_idx[new_canon]
        b_idx = name_to_idx[old_canon]
        canonical_idx, other_idx = _choose_canonical(graph, a_idx, b_idx)
        other_name = graph.vs[other_idx]["name"]
        canonical_name = graph.vs[canonical_idx]["name"]
        features = dict(features_list[i]) if features_list else {}
        # Stash the policy-derived propagation weight into the sidecar so
        # collapse_provenance edges carry a self-contained record of what
        # the overlay variant would have used as edge weight. Without this,
        # the sidecar reproduces every admission feature except the one
        # number that drove PPR behaviour.
        if w_prop_list and i < len(w_prop_list):
            features["w_prop"] = float(w_prop_list[i])
        _redirect_entity_passage_edges(
            graph,
            other_idx,
            canonical_idx,
            carry_provenance=carry_provenance,
            alias_features=features,
        )
        # Mark the absorbed vertex hidden — we do NOT physically delete
        # so reverse_map lookups still resolve and graphml round-trips
        # remain lossless.
        graph.vs[other_idx]["hidden"] = True
        reverse_map[other_name] = canonical_name
        accepted += 1
    return accepted


def on_alias_accepted(
    handler: str,
    graph: ig.Graph,
    new_hash_id: str,
    candidates: Sequence[AliasCandidate],
    features_list: Sequence[Dict[str, Any]],
    w_prop_list: Sequence[float],
    *,
    reverse_map: Optional[Dict[str, str]] = None,
    name_to_idx: Optional[Dict[str, int]] = None,
) -> int:
    """Dispatch alias acceptance to the handler named in ``handler``.

    ``reverse_map`` is mutated in place for collapse handlers (callers
    persist it after the batch); ignored for overlay. ``name_to_idx`` (name →
    vertex index) is an optional prebuilt map a batch caller passes once for
    the whole ER pass to avoid the O(V) rebuild per accepted entity. Unknown
    handlers raise ``ValueError`` — silent fallback would hide a config typo.
    """
    if handler == ACCEPTANCE_HANDLER_OVERLAY:
        return _on_alias_accepted_overlay(
            graph, new_hash_id, candidates, features_list, w_prop_list,
            name_to_idx=name_to_idx,
        )
    if handler in (ACCEPTANCE_HANDLER_COLLAPSE_BASIC, ACCEPTANCE_HANDLER_COLLAPSE_PROVENANCE):
        if reverse_map is None:
            raise ValueError(
                f"on_alias_accepted: collapse handler requires a reverse_map dict"
            )
        return _on_alias_accepted_collapse(
            graph,
            new_hash_id,
            candidates,
            features_list,
            w_prop_list,
            reverse_map=reverse_map,
            carry_provenance=(handler == ACCEPTANCE_HANDLER_COLLAPSE_PROVENANCE),
            name_to_idx=name_to_idx,
        )
    raise ValueError(f"on_alias_accepted: unknown handler {handler!r}")


# ============================================================================
# Collapse-mode clusters
# ============================================================================


def compute_clusters_for_collapse(
    graph: ig.Graph, reverse_map: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Build cluster dicts from the reverse_map.

    In collapse mode the alias subgraph is empty (every member was
    folded into a canonical), so ``compute_clusters`` returns nothing.
    Entity-lookup still needs cluster info, so we synthesise it from
    the reverse_map: every canonical with at least one absorbed
    member becomes a cluster keyed by the canonical's surface.
    """
    if not reverse_map:
        return []
    canonical_to_members: Dict[str, List[str]] = {}
    for other, canonical in reverse_map.items():
        # Follow the chain in case reverse_map carries pre-compaction hops.
        canonical_resolved = follow_reverse_map(canonical, reverse_map)
        canonical_to_members.setdefault(canonical_resolved, []).append(other)
    # Add the canonical itself as a member of its own cluster.
    name_to_idx = {v["name"]: v.index for v in graph.vs if "name" in v.attributes()}
    clusters: List[Dict[str, Any]] = []
    for cid, (canonical, others) in enumerate(sorted(canonical_to_members.items())):
        members = [canonical] + others
        canonical_text = canonical
        if canonical in name_to_idx:
            attrs = graph.vs[name_to_idx[canonical]].attributes()
            canonical_text = attrs.get("content") or canonical
        clusters.append(
            {
                "id": f"c_{cid:04d}",
                "members": members,
                "canonical": canonical_text,
            }
        )
    return clusters


# ============================================================================
# Cluster-aggregated scores
# ============================================================================


def aggregate_by_cluster(
    scores: Dict[str, float],
    clusters: Sequence[Dict[str, Any]],
    op: str = "sum",
) -> Dict[str, float]:
    """Project physical-node scores onto logical-cluster scores.

    Membership is taken from ``clusters[i]["members"]`` (list of hash_ids).
    Physical nodes not appearing in any cluster are passed through as
    singleton-clusters (key = hash_id).

    Supported ops:

    * ``sum``     — additive mass; the natural "logical entity strength".
    * ``max``     — max member score; conservative (no double-counting).
    * ``normsum`` — sum divided by sqrt(|members|); damps reward to
      large clusters so a 50-member garbage bucket doesn't outscore a
      clean 2-member cluster on a single strong hit.
    """
    if op not in {"sum", "max", "normsum"}:
        raise ValueError(f"aggregate_by_cluster: unknown op {op!r}")
    out: Dict[str, float] = {}
    seen_members: set = set()
    for c in clusters:
        members = c.get("members") or []
        cluster_id = c.get("id") or c.get("canonical") or ""
        member_scores = [float(scores.get(m, 0.0)) for m in members]
        for m in members:
            seen_members.add(m)
        if not member_scores:
            continue
        if op == "sum":
            v = float(sum(member_scores))
        elif op == "max":
            v = float(max(member_scores))
        else:  # normsum
            n = max(1, len(member_scores))
            v = float(sum(member_scores) / math.sqrt(n))
        out[str(cluster_id)] = v
    # Singleton pass-through — every physical entity that isn't in any
    # cluster contributes its own score under its hash_id key.
    for h, s in scores.items():
        if h not in seen_members:
            out[h] = float(s)
    return out
