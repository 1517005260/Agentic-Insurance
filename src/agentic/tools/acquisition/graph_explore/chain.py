"""``graph_chain`` — relational multi-hop walk over the entity graph.

Sentence-first co-occurrence beam walk + reciprocal-rank-fusion ranking.
Neighbors come from ``channel.cooccurrence_neighbors`` (the tail's
query-ranked mention sentences → their co-mentioned entities), so a hop
follows the natural-language predicate, not an alias edge.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np

from agentic.tools.acquisition._common import err, ok, parse_scope
from agentic.tools.acquisition.graph_explore.base import (
    _CHAIN_BEAM_K,
    _CHAIN_MAX_SEEDS,
    _CHAIN_TOP_L_NEIGHBORS,
    _CHAIN_TOP_M_SENTENCES,
    _CHAIN_TOP_PATHS,
    _CHAIN_TOP_S_SENTENCES,
    _DEFAULT_CHAIN_DEPTH,
    _GraphToolBase,
    _OVER_RETURN,
    _RRF_K,
    _coerce_int,
)
from ingestion.index.linear_rag.backfill import find_literal_matches
from ingestion.index.linear_rag.normalize import normalize_for_hash
from rag.channels.base import reciprocal_rank_fusion

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


logger = logging.getLogger(__name__)


class GraphChainTool(_GraphToolBase):
    """Relational multi-hop walk over the entity graph."""

    def _run_chain_walk(
        self, context, question, focus_seeds, focus_clusters, focus_audit, **kwargs
    ):
        """Sentence-first co-occurrence beam walk + RRF ranking.

        Neighbors come from ``channel.cooccurrence_neighbors`` (the tail's
        query-ranked mention sentences → their co-mentioned entities), so a
        hop follows the natural-language predicate, not an alias edge. Paths
        are ranked by reciprocal-rank fusion over interpretable criteria;
        candidate pages by RRF(endpoint-incident, bridge co-occurrence) with
        endpoints admitted regardless of edge rank.
        """
        channel = self._channel
        scope, scope_err = parse_scope(kwargs.get("file_ids"), kwargs.get("page_range"))
        if scope_err is not None:
            return err(
                "invalid_argument",
                scope_err,
                remediation="Fix `file_ids`/`page_range`, or omit for a corpus-wide walk.",
                valid_example={"file_ids": ["<file_id>"]},
            ), {"error": "invalid_argument"}
        if len(channel.sentence_store) == 0:
            return err(
                "graph_unavailable",
                "Sentence store is empty; graph_chain needs a sentence layer.",
                remediation="The sentence layer is not available in this corpus.",
            ), {"error": "graph_unavailable"}

        try:
            q_emb = channel.embedding_client.encode(question, is_query=True)
        except Exception as exc:
            logger.exception("graph_chain embedding failed: %s", exc)
            return err(
                "embed_failed",
                f"Embedding the query failed: {exc}",
                remediation="Retry once; if it persists, switch to graph_ppr.",
            ), {"error": "embed_failed"}
        if q_emb.ndim == 2:
            q_emb = q_emb[0]

        # Seeds = focus ∪ question (NER ∪ gazetteer ∪ embedding), deduped.
        # Focus (the deliberately-named anchors) goes FIRST so question seeds
        # never truncate the requested focus out of the _CHAIN_MAX_SEEDS budget
        # (true whenever len(focus) ≤ the cap — the common 1-2 anchor case;
        # passing more anchors than the cap is bounded as a resource limit).
        seeds_info = list(focus_seeds) + self._chain_seeds(question, q_emb)
        _seen: set = set()
        seeds_info = [
            s for s in seeds_info
            if not (s["hash_id"] in _seen or _seen.add(s["hash_id"]))
        ]
        seeds_info = seeds_info[:_CHAIN_MAX_SEEDS]
        if not seeds_info:
            return (
                ok(
                    "GraphExploreObservation",
                    question=question,
                    focus=[h.get("token") for h in focus_audit],
                    scope=scope.as_dict(),
                    seeds=[],
                    paths=[],
                    evidence=[],
                    more_candidates=[],
                    note="No seeds resolved from the question or focus.",
                ),
                {"retrieved_tokens": 0, "paths": 0},
            )

        # Sentence similarities computed once; shared across the whole beam.
        sent_hashes = channel.sentence_store.hash_ids
        sent_idx: Dict[str, int] = {h: i for i, h in enumerate(sent_hashes)}
        sent_sims = channel.sentence_store.all_similarities(q_emb)
        _, member_to_cluster = channel._load_clusters_cached()

        # Per-cluster neighbor memo — cooccurrence_neighbors folds the tail to
        # its cluster, so its result is a function of the cluster; keying the
        # memo by cluster means alias-variant tails don't re-expand the hub.
        nbr_memo: Dict[str, List[Dict[str, Any]]] = {}

        def neighbors(tail_hash: str) -> List[Dict[str, Any]]:
            cid = member_to_cluster.get(tail_hash, tail_hash)
            cached = nbr_memo.get(cid)
            if cached is None:
                cached = channel.cooccurrence_neighbors(
                    tail_hash, sent_sims, sent_idx,
                    top_s=_CHAIN_TOP_S_SENTENCES,
                    top_l=_CHAIN_TOP_L_NEIGHBORS,
                    max_via=_CHAIN_TOP_M_SENTENCES,
                )
                nbr_memo[cid] = cached
            return cached

        focus_cluster_set = set(focus_clusters)
        seed_clusters = {
            member_to_cluster.get(s["hash_id"], s["hash_id"]) for s in seeds_info
        }

        beam = [
            {"nodes": [s["hash_id"]], "edges": [],
             "clusters": {member_to_cluster.get(s["hash_id"], s["hash_id"])}}
            for s in seeds_info
        ]
        all_paths: List[Dict[str, Any]] = []
        for _hop in range(_DEFAULT_CHAIN_DEPTH):
            next_paths: List[Dict[str, Any]] = []
            for path in beam:
                tail = path["nodes"][-1]
                for nb in neighbors(tail):
                    head = nb["hash_id"]
                    hc = nb["cluster_id"]
                    # No physical revisit AND no LOGICAL revisit: returning to a
                    # cluster already on the path is an alias re-hop in disguise.
                    if head in path["nodes"] or hc in path["clusters"]:
                        continue
                    next_paths.append({
                        "nodes": path["nodes"] + [head],
                        "edges": path["edges"] + [{
                            "tail": tail, "head": head,
                            "max_cos": nb["max_cos"], "mean_cos": nb["mean_cos"],
                            "support": nb["support"], "via_sids": nb["via_sids"],
                            "head_cluster": hc,
                        }],
                        "clusters": path["clusters"] | {hc},
                    })
            if not next_paths:
                break
            # Beam prune by a monotone proxy (mean edge cosine); the final
            # ranking is RRF-over-criteria below, not this proxy.
            next_paths.sort(
                key=lambda p: sum(e["max_cos"] for e in p["edges"]) / len(p["edges"]),
                reverse=True,
            )
            beam = next_paths[:_CHAIN_BEAM_K]
            all_paths.extend(beam)

        # Target-aware: 2+ focus clusters → always add their direct
        # co-occurrence as depth-1 paths so a comparison bridge surfaces even
        # when the beam did not route through it.
        if len(focus_cluster_set) >= 2:
            all_paths.extend(self._focus_bridge_paths(
                focus_seeds, focus_cluster_set, sent_sims, sent_idx
            ))

        # Dedup by node tuple (keep the higher mean-edge-cos instance).
        by_sig: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for p in all_paths:
            if not p["edges"]:
                continue
            proxy = sum(e["max_cos"] for e in p["edges"]) / len(p["edges"])
            sig = tuple(p["nodes"])
            cur = by_sig.get(sig)
            if cur is None or proxy > cur.get("_proxy", -1.0):
                p["_proxy"] = proxy
                by_sig[sig] = p
        paths = self._rank_paths_rrf(
            list(by_sig.values()), seed_clusters, focus_cluster_set, member_to_cluster
        )[:_CHAIN_TOP_PATHS]

        paths_out = self._render_chain_paths(paths)
        candidate_pages = self._chain_candidate_pages(
            paths, seeds_info, focus_cluster_set, scope, limit=_OVER_RETURN
        )
        # Same asymmetric packaging as graph_ppr: top-K candidate pages carry
        # a query-centered ``window`` (``evidence``), the rest a one-line menu
        # (``more_candidates``), each with a full-read ``cost_tokens``;
        # per-run dedup collapses an already-shown/read page to a stub.
        # ``_attach_previews`` resolves each page's passage via its
        # (file_id, page_number) since chain candidates carry no per-hit
        # evidence handle.
        evidence, more_candidates = self._split_candidate_pages(
            candidate_pages, question=question, context=context
        )
        entity_audit = [
            self._compact_cluster_audit(h["cluster_id"], token=h.get("token", ""))
            for h in focus_audit if h.get("cluster_id")
        ]

        log_meta = {
            "question": question,
            "focus": [h.get("token") for h in focus_audit],
            "scope": scope.as_dict(),
            "seeds": [s["surface"] for s in seeds_info],
            "paths": len(paths_out), "candidate_pages": len(candidate_pages),
        }
        context.add_retrieval_log(tool_name="graph_chain", tokens=0, metadata=log_meta)
        return (
            ok(
                "GraphExploreObservation",
                question=question,
                focus=[h.get("token") for h in focus_audit],
                scope=scope.as_dict(),
                seeds=[{
                    "hash_id": s["hash_id"], "surface": s["surface"],
                    "sim": round(float(s["sim"]), 4), "source": s["source"],
                } for s in seeds_info],
                entity_audit=entity_audit,
                paths=paths_out,
                evidence=evidence,
                more_candidates=more_candidates,
            ),
            {
                "retrieved_tokens": 0,
                "paths": len(paths_out),
                "pages": len(candidate_pages),
            },
        )


    def _focus_bridge_paths(
        self, focus_seeds, focus_cluster_set, sent_sims, sent_idx
    ) -> List[Dict[str, Any]]:
        """Direct co-occurrence between each pair of distinct focus clusters,
        as depth-1 paths (comparison / bridge questions).

        This does NOT route through the generic beam (whose ``top_s``/``top_l``
        caps could drop a mediocre-cosine but real focus-pair bridge): it probes
        ``pair_via_sentences`` directly over the clusters' members and ranks the
        shared sentences by query cosine. Evidence is the UNION of shared
        sentences across all probed member pairs (so support/max/mean reflect
        every alias member, not just the single best pair); the displayed nodes
        come from the best-contributing member pair. Member fan-out is capped as
        a resource bound — the resolved focus-seed representative of each cluster
        is always probed first, so the guarantee is "checked within the member
        cap", not absolute.
        """
        channel = self._channel
        clusters, member_to_cluster = channel._load_clusters_cached()
        pair_via = channel.pair_via_sentences()
        seed_rep = {
            member_to_cluster.get(s["hash_id"], s["hash_id"]): s["hash_id"]
            for s in focus_seeds
        }
        _MEMBER_CAP = 20  # resource cap on |X|x|Y| pair probes per focus pair

        def members_of(cid):
            base = clusters.get(cid) or [cid]
            rep = seed_rep.get(cid)
            # Resolved anchor first (always probed), then the rest, deduped.
            ordered = ([rep] if rep else []) + [m for m in base if m != rep]
            return ordered[:_MEMBER_CAP]

        fcl = sorted(focus_cluster_set)
        out: List[Dict[str, Any]] = []
        for i in range(len(fcl)):
            for j in range(i + 1, len(fcl)):
                mi, mj = members_of(fcl[i]), members_of(fcl[j])
                via: Dict[str, float] = {}          # union of shared sentences
                best_pair, best_pair_max = None, -1.0
                for a in mi:
                    for b in mj:
                        sids = pair_via.get(frozenset((a, b)))
                        if not sids:
                            continue
                        pair_max = -1.0
                        for s in sids:
                            si = sent_idx.get(s)
                            if si is None:
                                continue
                            sim = float(sent_sims[si])
                            if sim > via.get(s, -1.0):
                                via[s] = sim
                            if sim > pair_max:
                                pair_max = sim
                        if pair_max > best_pair_max:
                            best_pair_max, best_pair = pair_max, (a, b)
                if best_pair is None:
                    continue
                a, b = best_pair
                ranked = sorted(via.items(), key=lambda t: t[1], reverse=True)
                out.append({
                    "nodes": [a, b],
                    "edges": [{
                        "tail": a, "head": b,
                        "max_cos": ranked[0][1],
                        "mean_cos": sum(v for _, v in ranked) / len(ranked),
                        "support": len(ranked),
                        "via_sids": [s for s, _ in ranked[:_CHAIN_TOP_M_SENTENCES]],
                        "head_cluster": fcl[j],
                    }],
                    "clusters": {fcl[i], fcl[j]},
                })
        return out


    def _rank_paths_rrf(
        self, paths, seed_clusters, focus_cluster_set, member_to_cluster
    ) -> List[Dict[str, Any]]:
        """Reciprocal-rank fusion over interpretable, weight-free criteria:
        focus-cluster coverage, seed-cluster coverage, weak-link (min) edge
        cosine, mean edge cosine, and endpoint specificity (low cluster df).
        No learned weights; the only constant is the published RRF ``k``.
        """
        if not paths:
            return paths
        idxs = list(range(len(paths)))
        metrics: List[Dict[str, float]] = []
        for p in paths:
            pcl = {member_to_cluster.get(h, h) for h in p["nodes"]}
            max_coses = [e["max_cos"] for e in p["edges"]]
            metrics.append({
                "focus_cov": float(len(pcl & focus_cluster_set)),
                "seed_cov": float(len(pcl & seed_clusters)),
                "min_edge": min(max_coses) if max_coses else 0.0,
                "mean_edge": (sum(max_coses) / len(max_coses)) if max_coses else 0.0,
                "endpoint_spec": -self._endpoint_df(p, member_to_cluster),
            })
        # Focus-cluster coverage is a hard tier, not just one RRF vote: a path
        # joining the named focus entities must outrank open-ended exploration
        # (target-aware comparison). Pure RRF over all five lets the tied edge
        # criteria outvote coverage, so coverage leads and RRF over the
        # remaining weight-free criteria breaks ties. With no focus, coverage
        # is uniformly 0 → this degrades to pure RRF.
        rank_lists = [
            sorted(idxs, key=lambda i: metrics[i][key], reverse=True)
            for key in ("seed_cov", "min_edge", "mean_edge", "endpoint_spec")
        ]
        fused = reciprocal_rank_fusion(rank_lists, k=_RRF_K)
        return [
            paths[i] for i in sorted(
                idxs,
                key=lambda i: (metrics[i]["focus_cov"], fused.get(i, 0.0)),
                reverse=True,
            )
        ]


    def _endpoint_df(self, path, member_to_cluster) -> float:
        """Cluster document-frequency (passage count) of the path endpoint.
        Lower = more specific. Corpus-derived, not a tuned threshold.
        """
        endpoint = path["nodes"][-1]
        cid = member_to_cluster.get(endpoint, endpoint)
        try:
            return float(self._channel.cluster_passage_count(cid))
        except Exception:
            return 0.0


    def _chain_seeds(
        self, question: str, q_emb: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Union NER ∪ gazetteer ∪ question-embedding top-k.

        Returns at most ``_CHAIN_MAX_SEEDS`` ranked by similarity (literal
        gazetteer hits pinned at sim=1.0). Hidden vertices (collapse-absorbed)
        are skipped. Each ingredient runs unconditionally so seeds are a true
        union — different question shapes lean on different ingredients.
        """
        channel = self._channel
        best: Dict[str, Tuple[float, str, str]] = {}  # hash → (sim, surface, source)

        # 1) NER-driven seeds — embedding-match each tagged surface.
        try:
            ner = channel._ensure_ner()
            raw_surfaces = ner.question_ner(question)
        except Exception as exc:
            logger.warning("graph_chain question_ner failed: %s", exc)
            raw_surfaces = []
        canonical: List[str] = []
        seen_canon: set = set()
        for raw in raw_surfaces:
            c = normalize_for_hash(
                raw,
                fold_traditional=channel.linear_config.fold_traditional,
                han_fragment_max_chars=channel.linear_config.junk_max_han_chars,
            )
            if c and c not in seen_canon:
                seen_canon.add(c)
                canonical.append(c)
        if canonical:
            embs = channel.embedding_client.encode(canonical)
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)
            for vec in embs:
                top1 = channel.entity_store.topk(vec, 1)
                if not top1:
                    continue
                hid, sc = top1[0]
                if hid not in channel._name_to_vidx:
                    continue
                if channel._is_hidden(channel._name_to_vidx[hid]):
                    continue
                surface = channel.entity_store.hash_id_to_text.get(hid, "")
                cur = best.get(hid)
                if cur is None or float(sc) > cur[0]:
                    best[hid] = (float(sc), surface, "ner")

        # 2) Gazetteer literal scan — Aho-Corasick over question text.
        try:
            gaz = channel._ensure_gazetteer()
        except Exception as exc:
            logger.warning("graph_chain gazetteer build failed: %s", exc)
            gaz = None
        if gaz is not None:
            counts = find_literal_matches(question, gaz)
            for hid in counts.keys():
                if hid not in channel._name_to_vidx:
                    continue
                if channel._is_hidden(channel._name_to_vidx[hid]):
                    continue
                surface = channel.entity_store.hash_id_to_text.get(hid, "")
                cur = best.get(hid)
                if cur is None or 1.0 > cur[0]:
                    best[hid] = (1.0, surface, "gazetteer")

        # 3) Whole-question embedding top-k — rank cap, no absolute floor (an
        # absolute cosine cutoff is a per-corpus tuned threshold; the top-k is
        # a resource bound and weak seeds are down-ranked by coverage / edge
        # cosine downstream).
        for hid, sc in channel.entity_store.topk(q_emb, 5):
            if hid not in channel._name_to_vidx:
                continue
            if channel._is_hidden(channel._name_to_vidx[hid]):
                continue
            surface = channel.entity_store.hash_id_to_text.get(hid, "")
            cur = best.get(hid)
            if cur is None or float(sc) > cur[0]:
                best[hid] = (float(sc), surface, "question_embedding")

        seeds = [
            {"hash_id": hid, "sim": sim, "surface": surf, "source": src}
            for hid, (sim, surf, src) in best.items()
        ]
        seeds.sort(key=lambda s: s["sim"], reverse=True)
        return seeds[:_CHAIN_MAX_SEEDS]


    def _render_chain_paths(
        self, paths: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Serialise paths for the agent: cluster-annotated nodes + edges with
        their query-ranked via-sentence snippets (the bridge evidence). The
        agent quotes these snippets; full text comes via the ``read`` tool.
        """
        channel = self._channel
        ent_text = channel.entity_store.hash_id_to_text
        sent_text = channel.sentence_store.hash_id_to_text
        _, m2c = channel._load_clusters_cached()
        out: List[Dict[str, Any]] = []
        for p in paths:
            nodes_out = []
            for h in p["nodes"]:
                cid = m2c.get(h, h)
                surf = ent_text.get(h, "")
                rep_surfs = channel.cluster_top_surfaces(cid, top_n=1)
                top = rep_surfs[0]["surface"] if rep_surfs else surf
                # The raw entity hash is never re-fed by the agent (it focuses
                # by surface / cluster_id); cluster_top_surface only adds signal
                # when it differs from the node's own surface.
                node = {"surface": surf, "cluster_id": cid}
                if top and top != surf:
                    node["cluster_top_surface"] = top
                nodes_out.append(node)
            edges_out = []
            for e in p["edges"]:
                # The agent quotes the via-sentence text, not the sentence hash;
                # endpoints read better as surfaces than raw entity hashes.
                via = [
                    (txt if len(txt) <= 220 else txt[:217] + "...")
                    for sid in e["via_sids"]
                    if (txt := sent_text.get(sid, ""))
                ]
                edges_out.append({
                    "from": ent_text.get(e["tail"], e["tail"]),
                    "to": ent_text.get(e["head"], e["head"]),
                    "edge_cos": round(float(e["max_cos"]), 4),
                    "support": int(e["support"]),
                    "via_sentences": via,
                })
            out.append({"nodes": nodes_out, "edges": edges_out, "hops": len(p["edges"])})
        return out


    def _chain_candidate_pages(
        self,
        paths: List[Dict[str, Any]],
        seeds_info: List[Dict[str, Any]],
        focus_cluster_set: set,
        scope,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """``RRF(endpoint-incident pages, bridge co-occurrence pages)``.

        Endpoint pages: incident passages of each path node, admitted
        regardless of edge rank — a 2-hop answer usually lives on the endpoint
        entity's own page, whose bridge sentence may have only mediocre query
        cosine. A node is admitted unless it is a pure question-seed that is
        neither the path endpoint nor a focus anchor (those seed pages are
        already known to the agent); the endpoint and every focus-cluster node
        are always admitted (so a ``focus=[X,Y]`` comparison surfaces BOTH X's
        and Y's pages). Incidence aggregates over all alias-cluster members, so
        the answer page is found even if it names a surface variant. Bridge
        pages: passages incident to both ends of a bridge edge, ranked by the
        edge's query cosine. Edge cosine ranks the path; it never gates which
        endpoint pages are admitted.
        """
        graph = self._channel.graph
        name_to_vidx = self._channel._name_to_vidx
        clusters, member_to_cluster = self._channel._load_clusters_cached()
        passage_meta = self._passage_meta_lookup()
        has_weight = "weight" in graph.es.attributes()
        seed_hashes = {s["hash_id"] for s in seeds_info}
        # Cluster-keyed incidence: union the incident passages of every member
        # of a node's logical entity (alias fold on the page-attribution side).
        inc_cache: Dict[str, List[Tuple[str, float]]] = {}

        def _node_incident(node_hash: str) -> List[Tuple[str, float]]:
            res: List[Tuple[str, float]] = []
            vidx = name_to_vidx.get(node_hash)
            if vidx is None:
                return res
            v = graph.vs[vidx]
            vtype = v.attributes().get("vertex_type") or self._guess_vertex_type(v["name"])
            if vtype != "entity":
                return res
            for e in graph.incident(vidx, mode="all"):
                edge = graph.es[e]
                w = float(edge["weight"]) if has_weight else 1.0
                tgt = edge.target if edge.source == vidx else edge.source
                tv = graph.vs[tgt]
                tvtype = tv.attributes().get("vertex_type") or self._guess_vertex_type(tv["name"])
                if tvtype == "passage":
                    res.append((tv["name"], w))
            return res

        def incident_passages(node_hash: str) -> List[Tuple[str, float]]:
            cid = member_to_cluster.get(node_hash, node_hash)
            cached = inc_cache.get(cid)
            if cached is not None:
                return cached
            agg: Dict[str, float] = defaultdict(float)
            for m in (clusters.get(cid) or [node_hash]):
                for ph, w in _node_incident(m):
                    agg[ph] += w
            res = list(agg.items())
            inc_cache[cid] = res
            return res

        endpoint_score: Dict[str, float] = defaultdict(float)
        for rank, p in enumerate(paths):
            base = 1.0 / (1.0 + rank)
            nodes = p["nodes"]
            last = len(nodes) - 1
            for pos, node in enumerate(nodes):
                cid = member_to_cluster.get(node, node)
                # Skip a pure question-seed only when it is neither the endpoint
                # nor a focus anchor.
                if node in seed_hashes and pos != last and cid not in focus_cluster_set:
                    continue
                for ph, w in incident_passages(node):
                    endpoint_score[ph] += base * w
        endpoint_rank = [
            ph for ph, _ in sorted(endpoint_score.items(), key=lambda kv: kv[1], reverse=True)
        ]

        bridge_score: Dict[str, float] = defaultdict(float)
        for p in paths:
            for e in p["edges"]:
                tail_p = {ph for ph, _ in incident_passages(e["tail"])}
                for ph, _w in incident_passages(e["head"]):
                    if ph in tail_p and e["max_cos"] > bridge_score[ph]:
                        bridge_score[ph] = e["max_cos"]
        bridge_rank = [
            ph for ph, _ in sorted(bridge_score.items(), key=lambda kv: kv[1], reverse=True)
        ]

        fused = reciprocal_rank_fusion([endpoint_rank, bridge_rank], k=_RRF_K)
        out: List[Dict[str, Any]] = []
        for ph, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            meta = passage_meta.get(ph)
            if meta is None:
                continue
            file_id, page_n = meta
            pn_int = _coerce_int(page_n)
            if not scope.contains(file_id, pn_int):
                continue
            out.append({
                "file_id": file_id,
                "page_id": f"p_{pn_int:04d}" if pn_int is not None else None,
                "page_number": pn_int,
                "score": round(float(score), 6),
            })
            if len(out) >= limit:
                break
        return out



    @property
    def name(self) -> str:
        return "graph_chain"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "graph_chain",
                "description": (
                    "Relational multi-hop. Give `question` to follow "
                    "relations across entities (system finds bridge + answer "
                    "pages). Optional `focus`=anchor entity name(s)/"
                    "cluster_id(s); 2+ = bridge/compare."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Relational / multi-hop natural-language question.",
                        },
                        "focus": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional anchor entity name(s) or cluster_id(s). Two or more = comparison/bridge between them.",
                        },
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional file id allow-list.",
                        },
                        "page_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional [start, end] inclusive page filter.",
                        },
                    },
                    "required": ["question"],
                },
            },
        }

    def execute(self, context: "AgentContext", **kwargs):
        question = (kwargs.get("question") or "").strip()
        if not question:
            return err(
                "invalid_argument",
                "graph_chain requires `question`.",
                remediation="Give a relational/multi-hop `question`; add `focus` to anchor on a known entity.",
                valid_example={"question": "Who is the spouse of the director of Inception?"},
            ), {"error": "invalid_argument"}
        channel = self._channel
        if channel.graph is None or len(channel.entity_store) == 0:
            return err(
                "graph_unavailable",
                "Entity store is empty; index the corpus first.",
                remediation="The entity layer is not available in this corpus.",
            ), {"error": "graph_unavailable"}

        focus_raw = kwargs.get("focus") or []
        if isinstance(focus_raw, str):
            focus_raw = [focus_raw]
        focus = [f.strip() for f in focus_raw if isinstance(f, str) and f.strip()]
        focus_seeds, focus_clusters, focus_audit = self._resolve_focus(focus)
        return self._run_chain_walk(
            context, question, focus_seeds, focus_clusters, focus_audit,
            file_ids=kwargs.get("file_ids"), page_range=kwargs.get("page_range"),
        )

