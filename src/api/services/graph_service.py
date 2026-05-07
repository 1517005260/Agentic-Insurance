"""GraphService — web-side facade over GraphPPRChannel + igraph.

Reuses the lifespan-built ``GraphPPRChannel`` singleton so the igraph
graph + three faiss stores are mmap'd exactly once per process. We do
NOT rebuild any of those resources here — the channel already carries
``passage_store`` / ``entity_store`` / ``sentence_store`` / ``graph``
references, and the LLM-side ``GraphExploreTool`` (the agent path)
shares the same instance.

Shape contract: every endpoint that returns subgraph data uses the
G6 v5 ``{nodes:[{id,label,vertex_type,...}], edges:[{source,target,
weight,type}]}`` schema. The five public methods cover the four
frontend use cases:

* overview / seed_search / expand / node_detail — generic canvas
* ppr_subgraph                                  — RAG PPR drawer

GraphAgent live replay (use case C) does NOT need a method here:
the agent's ``graph_explore`` tool already emits ``candidate_*`` /
``paths`` directly into the SSE ``tool_result`` payload.
"""
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import igraph as ig
import numpy as np

from rag.channels.graph_ppr import GraphPPRChannel
from ingestion.index.linear_rag.disambig import (
    AliasCandidate,
    gradient_topk_candidates,
    get_clusters,
)
from config.settings import faiss_graph_dir


logger = logging.getLogger(__name__)


# --------------------------------------------------------------- defaults

_OVERVIEW_TOP_CENTRAL = 10
_SEED_DEFAULT_TOP_K = 10
_SEED_MIN_SIM = 0.4               # matches GraphExploreTool.entity_lookup floor
_EXPAND_DEFAULT_TOP_K = 50
_EXPAND_MAX_HOPS = 3
_NODE_DETAIL_NEIGHBOR_FILES = 5


def _guess_vertex_type(hash_id: str) -> str:
    """Fallback when the graph lacks a ``vertex_type`` attribute.

    Surface form depends on how the index was built. Newer ingest
    runs always set the attribute; this guards older artifacts.
    """
    if hash_id.startswith("entity-"):
        return "entity"
    if hash_id.startswith("passage-"):
        return "passage"
    if hash_id.startswith("sentence-"):
        return "sentence"
    return "unknown"


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class GraphServiceUnavailable(RuntimeError):
    """Raised when the underlying graph artifact is missing.

    Lifespan still constructs ``GraphPPRChannel`` even when the
    graphml file isn't on disk yet (a fresh-deploy state); the
    routes translate this into a 503 so the frontend can fall back
    to a "graph not built yet" state instead of crashing.
    """


class GraphService:
    """Per-process façade. Hold one of these on ``app.state.graph_service``.

    Public methods are pure read-only; thread-safe under FastAPI's
    asyncio model because the underlying igraph + EmbeddingStore are
    immutable post-ingest.
    """

    def __init__(self, channel: GraphPPRChannel) -> None:
        self._channel = channel
        self._clusters_cache_path = faiss_graph_dir() / "clusters.json"
        # Lazy caches — built on first hit; cheap to recompute on
        # rebuild because we throw the service away on lifespan reload.
        self._cluster_index: Optional[Dict[str, Dict[str, Any]]] = None
        self._passage_meta_by_hash: Optional[
            Dict[str, Tuple[Optional[str], Optional[int]]]
        ] = None

    # --------------------------------------------------- helpers ----

    @property
    def graph(self) -> ig.Graph:
        if self._channel.graph is None:
            raise GraphServiceUnavailable(
                "graph not built yet — run ingest to populate "
                "STORAGE_PATH/faiss/graph/LinearRAG.graphml"
            )
        return self._channel.graph

    def _passage_meta_lookup(
        self,
    ) -> Dict[str, Tuple[Optional[str], Optional[int]]]:
        if self._passage_meta_by_hash is not None:
            return self._passage_meta_by_hash
        store = self._channel.passage_store
        col_file_id = store.meta_column("file_id")
        col_page_n = store.meta_column("page_number")
        self._passage_meta_by_hash = {
            h: (f, _coerce_int(p))
            for h, f, p in zip(store.hash_ids, col_file_id, col_page_n)
        }
        return self._passage_meta_by_hash

    def _cluster_index_lookup(self) -> Dict[str, Dict[str, Any]]:
        """``entity_hash → {canonical, members}`` via the disk cache.

        Built lazily; missing file → empty dict (older corpora may
        not have cluster info, hover card just hides the section).
        """
        if self._cluster_index is not None:
            return self._cluster_index
        idx: Dict[str, Dict[str, Any]] = {}
        try:
            clusters = get_clusters(self.graph, self._clusters_cache_path)
        except Exception as exc:
            logger.warning("graph_service: cluster cache load failed: %s", exc)
            clusters = []
        for c in clusters:
            for member in c.get("members", []):
                idx[member] = {
                    "cluster_id": c.get("id"),
                    "canonical": c.get("canonical"),
                    "members": c.get("members", []),
                }
        self._cluster_index = idx
        return idx

    def _vertex_type(self, vertex: ig.Vertex) -> str:
        """Pull ``vertex_type`` from the attribute or guess from hash prefix."""
        if "vertex_type" in vertex.attributes() and vertex["vertex_type"]:
            return vertex["vertex_type"]
        return _guess_vertex_type(vertex["name"])

    def _vertex_label(self, vertex: ig.Vertex) -> str:
        """Surface text for ``entity`` / ``passage`` / ``sentence`` vertices.

        Entities + sentences carry their content directly via the
        embedding store text columns; passage vertices get rendered as
        ``<file_id> p<page_number>`` for legibility.
        """
        vtype = self._vertex_type(vertex)
        name = vertex["name"]
        if vtype == "entity":
            return self._channel.entity_store.hash_id_to_text.get(name, name)
        if vtype == "sentence":
            return self._channel.sentence_store.hash_id_to_text.get(name, name)
        if vtype == "passage":
            file_id, page_n = self._passage_meta_lookup().get(name, (None, None))
            if file_id and page_n is not None:
                return f"{file_id} p{page_n}"
            return name
        return name

    # -------------------------------------------------- 1. overview ----

    def overview(self) -> Dict[str, Any]:
        """Counts + most central entities — the canvas's first paint."""
        g = self.graph
        # Per-type vertex counts. Cheap because attribute access is
        # O(V) and V is in the thousands.
        type_counts: Dict[str, int] = {}
        for v in g.vs:
            t = self._vertex_type(v)
            type_counts[t] = type_counts.get(t, 0) + 1

        # Top-K entities by degree. Restrict to entity vertices —
        # passage degree is dominated by adjacent_passage edges, which
        # are not informative for "central concept" ranking.
        entity_vidx = [v.index for v in g.vs if self._vertex_type(v) == "entity"]
        if entity_vidx:
            degrees = g.degree(entity_vidx)
            ranked = sorted(
                zip(entity_vidx, degrees),
                key=lambda x: x[1],
                reverse=True,
            )[:_OVERVIEW_TOP_CENTRAL]
            top_central = [
                {
                    "id": g.vs[vidx]["name"],
                    "label": self._vertex_label(g.vs[vidx]),
                    "vertex_type": "entity",
                    "degree": int(deg),
                }
                for vidx, deg in ranked
            ]
        else:
            top_central = []

        return {
            "counts": {
                "nodes": g.vcount(),
                "edges": g.ecount(),
                **type_counts,
            },
            "top_central_entities": top_central,
        }

    # -------------------------------------------------- 2. seed search ----

    def seed_search(self, query: str, top_k: int = _SEED_DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Embedding-based fuzzy entity match (mirrors agent entity_lookup).

        Empty query → empty list (caller's job to special-case this
        for nicer UX). Reuses the disambig ``gradient_topk_candidates``
        with min_sim=0.4 — same floor as the agent tool, so admin and
        agent see the same candidate set.
        """
        q = (query or "").strip()
        if not q:
            return []
        if len(self._channel.entity_store) == 0:
            return []
        emb = self._channel.embedding_client.encode(q)
        cands: List[AliasCandidate] = gradient_topk_candidates(
            emb,
            self._channel.entity_store,
            k=max(1, min(top_k, 50)),
            min_sim=_SEED_MIN_SIM,
        )
        clusters = self._cluster_index_lookup()
        ent_text = self._channel.entity_store.hash_id_to_text
        out: List[Dict[str, Any]] = []
        for c in cands:
            hit: Dict[str, Any] = {
                "hash_id": c.hash_id,
                "surface": ent_text.get(c.hash_id, ""),
                "similarity": float(c.score),
            }
            cluster = clusters.get(c.hash_id)
            if cluster:
                hit["logical_cluster"] = {
                    "canonical": cluster.get("canonical"),
                    "members": list(cluster.get("members", [])),
                }
            out.append(hit)
        return out

    # -------------------------------------------------- 3. expand ----

    def expand(
        self,
        node_id: str,
        *,
        hops: int = 1,
        top_k: int = _EXPAND_DEFAULT_TOP_K,
        vertex_type: str = "both",            # "entity" | "passage" | "both"
        file_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Subgraph centered at ``node_id`` out to ``hops`` (BFS).

        The seed itself is hop=0; every BFS-reached vertex carries its
        hop distance + the max edge weight on its shortest path. Edges
        are returned for the induced subgraph (only edges where BOTH
        endpoints are in the kept node set).

        ``vertex_type`` filter applies to non-seed nodes only — the
        seed is always included so the canvas has an anchor. ``file_ids``
        prunes passage vertices to those carrying matching file_id meta.
        """
        g = self.graph
        if node_id not in self._channel._name_to_vidx:
            raise KeyError(f"unknown node_id: {node_id!r}")

        seed_vidx = self._channel._name_to_vidx[node_id]
        hops_clamped = max(1, min(int(hops), _EXPAND_MAX_HOPS))
        top_k_clamped = max(1, min(int(top_k), 200))
        vtype_filter = vertex_type if vertex_type in ("entity", "passage", "both") else "both"
        file_id_set = set(file_ids) if file_ids else None

        # BFS with hop tracking. We keep the SHORTEST hop to each node
        # AND the MAX edge weight along any same-hop incoming edge —
        # if a later parent at the same hop has a stronger edge, score
        # bumps up. The previous "skip if already in visited" version
        # would freeze the score at the order-of-first-visit edge,
        # which made top_k truncation depend on igraph neighbor order.
        visited: Dict[int, int] = {seed_vidx: 0}     # vidx → hop
        score_to: Dict[int, float] = {seed_vidx: 1.0}
        frontier = [seed_vidx]
        for hop in range(1, hops_clamped + 1):
            next_frontier: List[int] = []
            for v in frontier:
                for nbr_vidx in g.neighbors(v, mode="all"):
                    eid = g.get_eid(v, nbr_vidx, error=False)
                    weight = float(g.es[eid]["weight"]) if eid >= 0 else 0.0
                    if nbr_vidx in visited:
                        # Same-hop revisit → bump score if stronger.
                        # Different (deeper) hop → keep original (BFS
                        # invariant: first visit is the shortest path).
                        if visited[nbr_vidx] == hop and weight > score_to.get(nbr_vidx, 0.0):
                            score_to[nbr_vidx] = weight
                        continue
                    visited[nbr_vidx] = hop
                    score_to[nbr_vidx] = weight
                    next_frontier.append(nbr_vidx)
            frontier = next_frontier
            if not frontier:
                break

        # Apply vertex_type + file_id filters; seed always survives.
        passage_meta = self._passage_meta_lookup()
        kept: List[int] = []
        for vidx, hop in visited.items():
            if vidx == seed_vidx:
                kept.append(vidx)
                continue
            v = g.vs[vidx]
            vt = self._vertex_type(v)
            if vtype_filter != "both" and vt != vtype_filter:
                continue
            if file_id_set is not None and vt == "passage":
                file_id, _ = passage_meta.get(v["name"], (None, None))
                if file_id not in file_id_set:
                    continue
            kept.append(vidx)

        # Cap by top_k of non-seed nodes, ranked by edge weight.
        if len(kept) > top_k_clamped:
            non_seed = [v for v in kept if v != seed_vidx]
            non_seed.sort(key=lambda v: score_to.get(v, 0.0), reverse=True)
            kept = [seed_vidx] + non_seed[: top_k_clamped - 1]

        kept_set = set(kept)
        nodes_out = [
            {
                "id": g.vs[vidx]["name"],
                "label": self._vertex_label(g.vs[vidx]),
                "vertex_type": self._vertex_type(g.vs[vidx]),
                "hop": visited[vidx],
                "score": float(score_to.get(vidx, 0.0)),
            }
            for vidx in kept
        ]
        # Walk only edges incident to ``kept`` vidx instead of the full
        # edge list — O(sum_v deg(v)) ≤ O(2E_kept), avoiding a full
        # graph scan per /expand request even on large graphs.
        edges_out = self._induced_edges(g, kept_set)
        return {"nodes": nodes_out, "edges": edges_out}

    def _induced_edges(self, g: ig.Graph, kept_set: set) -> List[Dict[str, Any]]:
        """Build the induced-subgraph edge list for ``kept_set``.

        De-duped by edge id: an edge incident to two kept endpoints
        would otherwise show up twice (once per endpoint walk).
        """
        seen_eids: set = set()
        out: List[Dict[str, Any]] = []
        has_weight = "weight" in g.es.attributes()
        has_etype = "edge_type" in g.es.attributes()
        for vidx in kept_set:
            for eid in g.incident(vidx, mode="all"):
                if eid in seen_eids:
                    continue
                e = g.es[eid]
                src, tgt = e.tuple
                if src not in kept_set or tgt not in kept_set:
                    continue
                seen_eids.add(eid)
                out.append(
                    {
                        "source": g.vs[src]["name"],
                        "target": g.vs[tgt]["name"],
                        "weight": float(e["weight"]) if has_weight else 1.0,
                        "type": (e["edge_type"] if has_etype and e["edge_type"] else "unknown"),
                    }
                )
        return out

    # -------------------------------------------------- 4. node detail ----

    def node_detail(self, hash_id: str) -> Dict[str, Any]:
        """Hover-card payload — pure text, no hash_id leak.

        Returns ``surface`` / ``vertex_type`` / ``degree`` plus, for
        entities, ``logical_cluster`` (alias canonical + members) +
        ``mention_count`` + ``neighboring_files`` (top file_ids that
        cite this entity). For passages we just surface the file +
        page header. Front-end shows nothing else, so we deliberately
        OMIT the hash_id — IDs are ugly in the UI.
        """
        g = self.graph
        if hash_id not in self._channel._name_to_vidx:
            raise KeyError(f"unknown node_id: {hash_id!r}")
        vidx = self._channel._name_to_vidx[hash_id]
        v = g.vs[vidx]
        vt = self._vertex_type(v)
        out: Dict[str, Any] = {
            "surface": self._vertex_label(v),
            "vertex_type": vt,
            "degree": int(g.degree(vidx)),
        }
        if vt == "entity":
            cluster = self._cluster_index_lookup().get(hash_id)
            if cluster:
                out["logical_cluster"] = {
                    "canonical": cluster.get("canonical"),
                    "members": list(cluster.get("members", [])),
                }
            # Mention count = count of sentence-vertex neighbors
            # connected by entity_passage / mention edges. Cheap proxy
            # since we don't track per-edge mention frequency.
            neighbor_vidx = g.neighbors(vidx, mode="all")
            mention_count = 0
            file_ids_seen: Dict[str, int] = {}
            passage_meta = self._passage_meta_lookup()
            for nv in neighbor_vidx:
                ntype = self._vertex_type(g.vs[nv])
                if ntype in ("sentence", "passage"):
                    mention_count += 1
                if ntype == "passage":
                    file_id, _ = passage_meta.get(g.vs[nv]["name"], (None, None))
                    if file_id:
                        file_ids_seen[file_id] = file_ids_seen.get(file_id, 0) + 1
            out["mention_count"] = mention_count
            top_files = sorted(file_ids_seen.items(), key=lambda kv: kv[1], reverse=True)
            out["neighboring_files"] = [fid for fid, _ in top_files[:_NODE_DETAIL_NEIGHBOR_FILES]]
        elif vt == "passage":
            file_id, page_n = self._passage_meta_lookup().get(hash_id, (None, None))
            out["file_id"] = file_id
            out["page_number"] = page_n
        return out

    # -------------------------------------------------- 5. ppr subgraph ----

    def ppr_subgraph(
        self,
        query: str,
        *,
        file_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Re-run PPR for ``query`` and return seeds + actived + passages
        + induced edges — what the RAG-PPR drawer needs to render.

        We re-run rather than caching the original RAG call's PPR
        intermediate state: cheap on a warm channel (~300-500ms),
        no consistency window between the trace and the live graph,
        and the user opens this drawer rarely.

        ``mode`` in the response distinguishes:
          - ``ppr``       — normal happy path
          - ``no_seeds``  — query had no entity match (fallback included)
          - ``no_graph``  — graphml missing on disk
        """
        result = self._channel.retrieve_subgraph(
            question=query,
            file_ids=list(file_ids) if file_ids else None,
        )
        mode = result.get("mode", "ppr")
        if mode != "ppr":
            return {
                "mode": mode,
                "seeds": [],
                "actived_entities": [],
                "passages": [],
                "edges": [],
            }

        g = self.graph
        seeds_raw = result["seeds"]
        actived_raw = result["actived_entities"]   # {hash_id: {vidx, surface, score, tier}}
        passages_raw = result["passages"]          # List[ChannelHit]

        # Collect every vidx involved so we can render the induced
        # subgraph: seeds ∪ actived ∪ passages (the latter via name lookup).
        kept_vidx: Dict[int, str] = {}             # vidx → role tag for the response
        for s in seeds_raw:
            kept_vidx[s["vertex_idx"]] = "seed"
        for hid, info in actived_raw.items():
            if info["vertex_idx"] not in kept_vidx:
                kept_vidx[info["vertex_idx"]] = "actived"
        passage_payload: List[Dict[str, Any]] = []
        for hit in passages_raw:
            # ChannelHit has evidence=[{passage_hash_id: hid}]; pull it
            # back out so we can locate the vertex.
            phash = None
            if hit.evidence:
                phash = hit.evidence[0].get("passage_hash_id")
            if phash and phash in self._channel._name_to_vidx:
                kept_vidx.setdefault(self._channel._name_to_vidx[phash], "passage")
            passage_payload.append(
                {
                    "hash_id": phash,
                    "file_id": hit.file_id,
                    "page_id": hit.page_id,
                    "score": float(hit.score),
                }
            )

        seeds_payload = [
            {
                "id": s["hash_id"],
                "surface": s["surface"],
                "similarity": float(s["sim"]),
            }
            for s in seeds_raw
        ]
        actived_payload = [
            {
                "id": hid,
                "surface": info["surface"],
                "score": float(info["score"]),
                "iteration_tier": int(info["iteration_tier"]),
            }
            for hid, info in actived_raw.items()
        ]
        # Induced edges over the kept vidx set — same incident-walk
        # optimization as :meth:`expand`.
        edges_out = self._induced_edges(g, set(kept_vidx.keys()))
        return {
            "mode": "ppr",
            "seeds": seeds_payload,
            "actived_entities": actived_payload,
            "passages": passage_payload,
            "edges": edges_out,
        }
