"""Entity-graph retrieval over the LinearRAG-style graph.

Three modes:

* ``ppr`` — question-driven personalized PageRank over the entity ↔
  passage graph, identical to the standalone-RAG ``graph_ppr`` channel.
  We delegate to :class:`rag.channels.GraphPPRChannel` so the two paths
  cannot drift apart in subtle ways (NER → seed entities → entity-score
  BFS → passage scoring → PPR).
* ``neighbors`` — k-hop BFS over the in-memory igraph from a list of
  seed entities and/or page references. Returns the set of reachable
  entities, the set of reachable passages, and a small bundle of
  representative paths. Edge weights determine traversal priority.
* ``entity_lookup`` — embedding-match a surface form to *physical*
  entity nodes, and surface each hit's *logical* (alias-cluster)
  referent. Same gradient-topk + min-sim machinery used by the
  build-time disambiguator (`gradient_topk_candidates`), so the
  semantics are stable across query / index time.

Physical vs logical entity:
  Each surface form is indexed as its own node ("AXA", "AXA Hong Kong",
  "安盛"). Alias edges between near-synonyms partition the entity layer
  into logical clusters; the cluster is the LOGICAL entity. Use
  ``entity_lookup`` to map a question term to physical hits + logical
  cluster, then feed the physical hash into ``neighbors`` for
  hard-edge traversal.

Pre-warming:
  spaCy NER and igraph are heavy to load. The agent's ``warm_up()``
  invokes :meth:`warm_up` on this tool to absorb that one-time cost
  before the user's first turn.

Lifetime assumption:
  The graph and passage stores are treated as immutable for the life
  of the agent process. ``_passage_meta_lookup`` / ``_passage_hash_by_meta``
  / ``_cluster_index`` are cached on the tool instance after first use.
  If the corpus is re-ingested mid-session, construct a fresh tool
  instance — the caches are not invalidated automatically.
"""

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from agentic.tools.acquisition._common import err, ok, parse_scope
from agentic.tools.base import BaseTool
from config.settings import faiss_graph_dir
from ingestion.index.linear_rag.disambig import gradient_topk_candidates, get_clusters
from ingestion.index.linear_rag.normalize import normalize_for_hash
from rag.channels.graph_ppr import GraphPPRChannel
from rag.preprocess import QueryContext
from storage.inventory_store import InventoryStore

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


logger = logging.getLogger(__name__)


_VALID_MODES = {"ppr", "neighbors", "entity_lookup"}
_DEFAULT_TOP_K = 20
_DEFAULT_HOPS = 1
_MAX_HOPS = 3
_MAX_TOP_K = 50


class GraphExploreTool(BaseTool):
    def __init__(
        self,
        channel: Optional[GraphPPRChannel] = None,
        inventory: Optional[InventoryStore] = None,
        *,
        entity_lookup_min_sim: float = 0.6,
        entity_lookup_gradient: float = 0.5,
    ):
        # The channel owns graph + entity/passage/sentence stores + NER
        # state. We hold a reference instead of re-loading everything,
        # which both saves memory and guarantees the two retrieval paths
        # operate on the same data.
        self._channel = channel or GraphPPRChannel()
        self.inventory = inventory
        # Entity lookup tunables — admin config injects these via the
        # factory (build_graph_agent → ConfigStore). The defaults here
        # reproduce the pre-Phase-6 hardcoded constants
        # (_ENTITY_LOOKUP_MIN_SIM=0.6, gradient g=0.5) so call sites
        # that construct the tool directly (experiment scripts) keep
        # bytewise behaviour.
        self._entity_lookup_min_sim = float(entity_lookup_min_sim)
        self._entity_lookup_gradient = float(entity_lookup_gradient)
        self._clusters_cache_path: Path = faiss_graph_dir() / "clusters.json"
        self._cluster_index: Optional[Dict[str, Dict[str, Any]]] = None
        # Passage meta is stable for the life of the agent (graph is
        # built once at ingest). Cache the hash<->meta maps so neighbors
        # mode doesn't rebuild them per node render.
        self._passage_meta_by_hash: Optional[Dict[str, Tuple[str, Optional[int]]]] = None
        self._passage_hash_by_meta: Optional[Dict[Tuple[str, Optional[int]], str]] = None

    @property
    def name(self) -> str:
        return "graph_explore"

    # ---------------------------------------------------------- invalidate

    def invalidate_caches(self) -> None:
        """Drop the per-instance derived caches.

        ``_passage_meta_by_hash`` / ``_passage_hash_by_meta`` reflect a
        snapshot of the channel's passage_store at first use; after a
        reingest or delete the store has different rows and the cached
        maps return stale (file_id, page_number) pairs. Same story for
        ``_cluster_index`` — it's loaded from
        ``faiss/graph/clusters.json`` which is rewritten on ingest.

        The wrapping ``GraphPPRChannel.reload()`` rebuilds the underlying
        stores; this method is the second half of "make a stale tool
        instance look fresh again" and should be called from the
        lifespan refresh hook.
        """
        self._passage_meta_by_hash = None
        self._passage_hash_by_meta = None
        self._cluster_index = None

    # ----------------------------------------------------------- warm-up

    def warm_up(self) -> None:
        """Pre-load spaCy NER and force igraph to be in memory.

        The PPR mode needs spaCy on the first call, which is the
        single biggest source of cold-start latency in the agent loop
        (~10–15 s for transformer-based en + zh models). Run this
        before the agent's first user turn.
        """
        if self._channel.graph is None:
            return  # nothing to warm — graph not built
        try:
            self._channel._ensure_spacy()
        except Exception as exc:
            logger.warning("graph_explore: NER warm-up failed: %s", exc)

    # ----------------------------------------------------------- schema

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "graph_explore",
                "description": (
                    "Entity-graph retrieval. Three modes:\n"
                    "- mode='ppr': personalized PageRank from a free-text "
                    "question. Best for fuzzy semantic neighborhoods.\n"
                    "- mode='neighbors': k-hop BFS from explicit seed "
                    "entities and/or page references. Best for tracing "
                    "hard relations once you already have an anchor.\n"
                    "- mode='entity_lookup': embedding-match a surface "
                    "form to physical entity nodes and report each hit's "
                    "logical (alias) cluster.\n\n"
                    "Physical vs logical entity:\n"
                    "Each surface form indexes as its own node ('AXA', "
                    "'AXA Hong Kong', '安盛'). Alias edges fuse near-"
                    "synonyms into logical clusters — the cluster is the "
                    "logical entity. Look up first, then traverse.\n\n"
                    "Required arguments depend on the mode:\n"
                    "- ppr: question, optional file_ids / page_range / "
                    "section_ids, top_k.\n"
                    "- neighbors: seeds (list), optional hops (1-3), "
                    "optional file_ids, top_k.\n"
                    "- entity_lookup: surface, optional top_k (max 10)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": sorted(_VALID_MODES),
                            "description": "ppr | neighbors | entity_lookup.",
                        },
                        "question": {
                            "type": "string",
                            "description": "Free-text question (mode=ppr).",
                        },
                        "seeds": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Seed list (mode=neighbors). Each item is "
                                "either an entity surface form or a page "
                                "reference 'file_id/page_id'."
                            ),
                        },
                        "surface": {
                            "type": "string",
                            "description": "Entity surface form (mode=entity_lookup).",
                        },
                        "hops": {
                            "type": "integer",
                            "description": "BFS depth for mode=neighbors; default 1, max 3.",
                        },
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional file id allow-list. Honored by "
                                "ppr (full scope) and neighbors (page-side "
                                "only)."
                            ),
                        },
                        "page_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "Optional [start, end] inclusive page-number "
                                "filter (mode=ppr only)."
                            ),
                        },
                        "section_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional hard filter (mode=ppr only). "
                                "Section ids of the form "
                                "'<file_id>:sec_NNN' from `toc`. A "
                                "candidate page must lie inside at "
                                "least one listed section to qualify; "
                                "ANDed with file_ids and page_range."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max items to return; default 20, max 50.",
                        },
                    },
                    "required": ["mode"],
                },
            },
        }

    # ----------------------------------------------------------- execute

    def execute(self, context: "AgentContext", **kwargs):
        mode = str(kwargs.get("mode") or "").strip().lower()
        if mode not in _VALID_MODES:
            return (
                err(
                    "invalid_argument",
                    f"`mode` must be one of {sorted(_VALID_MODES)}.",
                    remediation="Set `mode` to 'ppr' (free-text), 'neighbors' (BFS from seeds), or 'entity_lookup' (surface-form to entity).",
                    valid_example={"mode": "ppr"},
                ),
                {"error": "invalid_argument"},
            )
        if self._channel.graph is None:
            return (
                err(
                    "graph_unavailable",
                    "LinearRAG graph is not built; ingest the corpus first.",
                    remediation="Fall back to semantic_search / bm25_search / pattern_search for this query — the entity graph is not available in this corpus.",
                ),
                {"error": "graph_unavailable"},
            )

        if mode == "ppr":
            return self._run_ppr(context, **kwargs)
        if mode == "neighbors":
            return self._run_neighbors(context, **kwargs)
        return self._run_entity_lookup(context, **kwargs)

    # ----------------------------------------------------------- PPR mode

    def _run_ppr(self, context: "AgentContext", **kwargs):
        question = (kwargs.get("question") or "").strip()
        if not question:
            return err(
                "invalid_argument",
                "mode='ppr' requires `question`.",
                remediation="Add `question` (free-text natural-language query) to the call; PPR uses NER to seed the random walk.",
                valid_example={"mode": "ppr", "question": "Which sections discuss premium rebates?"},
            ), {"error": "invalid_argument"}
        scope, scope_err = parse_scope(
            kwargs.get("file_ids"),
            kwargs.get("page_range"),
            kwargs.get("section_ids"),
            inventory=self.inventory,
        )
        if scope_err is not None:
            return err(
                "invalid_argument",
                scope_err,
                remediation="Fix the scope arguments per the message: file_ids must come from list_files; page_range must be [start, end]; section_ids must come from toc.",
                valid_example={"file_ids": ["<file_id>"]},
            ), {"error": "invalid_argument"}

        try:
            top_k_int = int(kwargs.get("top_k", _DEFAULT_TOP_K))
        except (TypeError, ValueError):
            return err(
                "invalid_argument",
                "`top_k` must be an integer.",
                remediation="Pass `top_k` as a positive integer (default 20, max 50).",
                valid_example={"top_k": 20},
            ), {"error": "invalid_argument"}
        if top_k_int < 1:
            return err(
                "invalid_argument",
                "`top_k` must be >= 1.",
                remediation="Set `top_k` to a positive integer (default 20, max 50).",
                valid_example={"top_k": 20},
            ), {"error": "invalid_argument"}
        limit = min(top_k_int, _MAX_TOP_K)

        ctx = QueryContext(
            query=question,
            hyde="",
            rewrite="",
            lang="",
            regexes=[],
            file_ids=list(scope.file_ids) if scope.file_ids else None,
            # Agent path: PPR is the only signal here, so let the channel
            # rescue NER misses with the gazetteer + question-embedding
            # fallbacks. The 4-channel RAG path leaves this off.
            enable_ppr_seed_fallback=True,
        )
        try:
            # Atomic ``hits + debug snapshot`` — the channel singleton
            # is shared across all 3 agent kinds AND the RAG pipeline,
            # so reading ``self._channel.last_debug`` after the call is
            # racy: another concurrent retrieve() could overwrite it.
            # ``retrieve_with_debug`` holds the channel's RLock for both
            # the call and the snapshot.
            channel_hits, debug_snapshot = self._channel.retrieve_with_debug(ctx)
        except Exception as exc:
            logger.exception("graph_explore[ppr] failed: %s", exc)
            return err(
                "ppr_failed",
                f"PPR raised: {exc}",
                remediation="Retry with a simpler question or fall back to semantic_search / bm25_search; the PPR channel hit an internal error.",
            ), {"error": "ppr_failed"}

        # Apply the page_range gate post-hoc (PPR channel honors file_ids
        # but not page_range — the channel was designed for the global RAG
        # path which doesn't expose ranges). Reuse the cached meta map.
        meta_by_hash = self._passage_meta_lookup()
        results: List[Dict[str, Any]] = []

        for hit in channel_hits:
            page_number = self._first_page_number(hit, meta_by_hash)
            if not scope.contains(hit.file_id, page_number):
                continue
            results.append(
                {
                    "file_id": hit.file_id,
                    "page_id": hit.page_id,
                    "page_number": page_number,
                    "score": round(float(hit.score), 6),
                }
            )
            if len(results) >= limit:
                break

        seeds_dbg = debug_snapshot.get("seeds", [])
        log_meta = {
            "mode": "ppr",
            "question": question,
            "scope": scope.as_dict(),
            "seeds": [s["surface"] for s in seeds_dbg] if isinstance(seeds_dbg, list) else [],
            "hits": len(results),
        }
        context.add_retrieval_log(tool_name="graph_explore", tokens=0, metadata=log_meta)

        return (
            ok(
                "GraphExploreObservation",
                mode="ppr",
                question=question,
                scope=scope.as_dict(),
                seeds=[
                    {"surface": s.get("surface"), "sim": s.get("sim")}
                    for s in (seeds_dbg if isinstance(seeds_dbg, list) else [])
                ],
                candidate_pages=results,
            ),
            {"retrieved_tokens": 0, "hits": len(results)},
        )

    @staticmethod
    def _first_page_number(hit, meta_by_hash) -> Optional[int]:
        for ev in hit.evidence or []:
            h = ev.get("passage_hash_id")
            if h and h in meta_by_hash:
                _, pn = meta_by_hash[h]
                try:
                    return int(pn) if pn is not None else None
                except (TypeError, ValueError):
                    return None
        return None

    # ----------------------------------------------------------- neighbors mode

    def _run_neighbors(self, context: "AgentContext", **kwargs):
        seeds = kwargs.get("seeds") or []
        if not isinstance(seeds, (list, tuple)) or not seeds:
            return err(
                "invalid_argument",
                "mode='neighbors' requires a non-empty `seeds` list.",
                remediation="Pass `seeds` as a list of entity surface forms (e.g. 'AXA') and/or page references (e.g. 'fileA_xxx/p_0001'). Use mode='entity_lookup' first to discover canonical surface forms.",
                valid_example={"mode": "neighbors", "seeds": ["AXA", "<file_id>/p_0001"]},
            ), {"error": "invalid_argument"}

        try:
            hops = int(kwargs.get("hops", _DEFAULT_HOPS))
        except (TypeError, ValueError):
            return err(
                "invalid_argument",
                "`hops` must be an integer.",
                remediation="Pass `hops` as an integer in [1, 3] (default 1).",
                valid_example={"hops": 1},
            ), {"error": "invalid_argument"}
        if not 1 <= hops <= _MAX_HOPS:
            return (
                err(
                    "invalid_argument",
                    f"`hops` must be 1..{_MAX_HOPS}.",
                    remediation=f"Set `hops` to an integer in [1, {_MAX_HOPS}] (default 1).",
                    valid_example={"hops": 1},
                ),
                {"error": "invalid_argument"},
            )
        try:
            top_k_int = int(kwargs.get("top_k", _DEFAULT_TOP_K))
        except (TypeError, ValueError):
            return err(
                "invalid_argument",
                "`top_k` must be an integer.",
                remediation="Pass `top_k` as a positive integer (default 20, max 50).",
                valid_example={"top_k": 20},
            ), {"error": "invalid_argument"}
        if top_k_int < 1:
            return err(
                "invalid_argument",
                "`top_k` must be >= 1.",
                remediation="Set `top_k` to a positive integer (default 20, max 50).",
                valid_example={"top_k": 20},
            ), {"error": "invalid_argument"}
        limit = min(top_k_int, _MAX_TOP_K)

        scope, scope_err = parse_scope(kwargs.get("file_ids"), None)
        if scope_err is not None:
            return err(
                "invalid_argument",
                scope_err,
                remediation="Fix `file_ids`: pass a list of ids returned by list_files, or omit the field for corpus-wide neighbors.",
                valid_example={"file_ids": ["<file_id>"]},
            ), {"error": "invalid_argument"}

        resolved = self._resolve_seeds(seeds)
        if not resolved["found"]:
            return (
                err(
                    "no_seeds_resolved",
                    "None of the seeds could be matched to a graph node.",
                    remediation="Call mode='entity_lookup' with each surface form first to find the canonical entity, then pass its hash_id (or the matched surface) as a seed; for page seeds use 'file_id/p_NNNN' from any retrieval observation.",
                    seeds=list(seeds),
                    unresolved=resolved["unresolved"],
                ),
                {"error": "no_seeds_resolved"},
            )

        graph = self._channel.graph
        seed_vidx = [item["vertex_idx"] for item in resolved["found"]]
        bfs, parents = self._bfs(seed_vidx, hops=hops)

        # Materialize: split into entities and passages.
        passage_meta = self._passage_meta_lookup()
        candidate_entities: List[Dict[str, Any]] = []
        candidate_pages: List[Dict[str, Any]] = []
        for vidx, score, hop in bfs:
            v = graph.vs[vidx]
            vtype = v.attributes().get("vertex_type") or self._guess_vertex_type(v["name"])
            if vtype == "entity":
                candidate_entities.append(
                    {
                        "hash_id": v["name"],
                        "surface": v.attributes().get("content") or "",
                        "score": round(float(score), 6),
                        "hop": hop,
                    }
                )
            elif vtype == "passage":
                meta = passage_meta.get(v["name"])
                if meta is None:
                    continue
                file_id, page_n = meta
                if scope.file_ids is not None and file_id not in scope.file_ids:
                    continue
                page_id = f"p_{int(page_n):04d}" if page_n is not None else None
                candidate_pages.append(
                    {
                        "file_id": file_id,
                        "page_id": page_id,
                        "page_number": int(page_n) if page_n is not None else None,
                        "score": round(float(score), 6),
                        "hop": hop,
                    }
                )

        candidate_entities.sort(key=lambda r: r["score"], reverse=True)
        candidate_pages.sort(key=lambda r: r["score"], reverse=True)
        candidate_entities = candidate_entities[:limit]
        candidate_pages = candidate_pages[:limit]

        # Build a small "why" trail so the agent can see HOW each top hit
        # is connected to the seeds. Cap to the union of the top entities
        # and pages so the path bundle stays bounded.
        path_targets = [(r["hash_id"], "entity") for r in candidate_entities[:5]] + [
            ((r["file_id"], r.get("page_number")), "passage")
            for r in candidate_pages[:5]
        ]
        paths = self._materialize_paths(parents, resolved["found"], path_targets)

        log_meta = {
            "mode": "neighbors",
            "seeds_count": len(seeds),
            "seeds_found": len(resolved["found"]),
            "hops": hops,
            "entities": len(candidate_entities),
            "pages": len(candidate_pages),
            "paths": len(paths),
        }
        context.add_retrieval_log(tool_name="graph_explore", tokens=0, metadata=log_meta)

        return (
            ok(
                "GraphExploreObservation",
                mode="neighbors",
                hops=hops,
                seeds_resolved=resolved["found"],
                seeds_unresolved=resolved["unresolved"],
                candidate_entities=candidate_entities,
                candidate_pages=candidate_pages,
                paths=paths,
            ),
            {
                "retrieved_tokens": 0,
                "entities": len(candidate_entities),
                "pages": len(candidate_pages),
                "paths": len(paths),
            },
        )

    def _resolve_seeds(self, seeds: Sequence[str]) -> Dict[str, Any]:
        """Resolve heterogeneous seed strings to graph vertices.

        Each seed is one of:

        * page reference ``"file_id/page_id"`` — looked up via the
          passage store's meta columns.
        * entity surface — normalized then matched to entity_store
          ``text_to_hash_id``; falls back to a normalized text match
          when the canonical form differs.
        """
        graph = self._channel.graph
        name_to_vidx = self._channel._name_to_vidx
        passage_lookup = self._passage_meta_to_hash()
        entity_text_to_hash = self._channel.entity_store.text_to_hash_id

        found: List[Dict[str, Any]] = []
        unresolved: List[str] = []
        for raw in seeds:
            s = str(raw).strip()
            if not s:
                continue
            hash_id = self._resolve_one(s, passage_lookup, entity_text_to_hash)
            if hash_id is None or hash_id not in name_to_vidx:
                unresolved.append(s)
                continue
            vidx = name_to_vidx[hash_id]
            v = graph.vs[vidx]
            vtype = v.attributes().get("vertex_type") or self._guess_vertex_type(hash_id)
            found.append(
                {
                    "input": s,
                    "hash_id": hash_id,
                    "vertex_type": vtype,
                    "vertex_idx": vidx,
                    "surface": v.attributes().get("content") or "",
                }
            )
        return {"found": found, "unresolved": unresolved}

    @staticmethod
    def _resolve_one(
        s: str,
        passage_lookup: Dict[Tuple[str, Optional[int]], str],
        entity_text_to_hash: Dict[str, str],
    ) -> Optional[str]:
        # Page reference?
        if "/" in s:
            file_id, _, page_id = s.partition("/")
            if page_id.startswith("p_"):
                try:
                    page_n: Optional[int] = int(page_id[2:])
                except ValueError:
                    page_n = None
            else:
                page_n = None
            if page_n is not None:
                hit = passage_lookup.get((file_id, page_n))
                if hit:
                    return hit
        # Entity surface — normalize to canonical hash key.
        canon = normalize_for_hash(s, fold_traditional=True)
        if canon and canon in entity_text_to_hash:
            return entity_text_to_hash[canon]
        # Last resort: raw text match (already-canonical input).
        if s in entity_text_to_hash:
            return entity_text_to_hash[s]
        return None

    def _bfs(
        self,
        seed_vidx: List[int],
        hops: int,
    ) -> Tuple[List[Tuple[int, float, int]], Dict[int, int]]:
        """Breadth-first expansion. Returns ``(visited, parents)``.

        ``visited`` is ``[(vidx, score, hop), ...]`` with seed vertices
        excluded so callers see only *neighbors*.

        ``parents`` maps every visited vertex to the predecessor along
        a SHORTEST-HOP path (BFS first-discovery — never overwritten on
        re-visit). Seeds map to themselves. Used by
        :meth:`_materialize_paths` to render the breadcrumb trail.

        Score is the *maximum* sum-of-edge-weights seen along any
        explored path (treating absent weights as 1.0). Score and
        parent are deliberately decoupled: parent always points along
        the BFS tree (no cycles), score reflects the strongest
        connection found during exploration.
        """
        graph = self._channel.graph
        has_weight = "weight" in graph.es.attributes()
        # ``best[v] = (score, hop)`` — score may improve on revisits;
        # hop never does (BFS guarantees shortest hop on first visit).
        best: Dict[int, Tuple[float, int]] = {v: (0.0, 0) for v in seed_vidx}
        # ``parents`` is set ONCE per vertex on first discovery so the
        # parent graph is acyclic and bounded-depth.
        parents: Dict[int, int] = {v: v for v in seed_vidx}
        frontier: deque = deque((v, 0.0, 0) for v in seed_vidx)
        while frontier:
            v, score, hop = frontier.popleft()
            if hop >= hops:
                continue
            for e in graph.incident(v, mode="all"):
                edge = graph.es[e]
                w = float(edge["weight"]) if has_weight else 1.0
                target = edge.target if edge.source == v else edge.source
                new_score = score + w
                if target not in best:
                    best[target] = (new_score, hop + 1)
                    parents[target] = v
                    frontier.append((target, new_score, hop + 1))
                else:
                    # Already discovered — only update score, keep the
                    # original BFS-tree parent intact.
                    cur_score, cur_hop = best[target]
                    if new_score > cur_score:
                        best[target] = (new_score, cur_hop)
        visited = [
            (vidx, sc, hp)
            for vidx, (sc, hp) in best.items()
            if hp > 0
        ]
        return visited, parents

    def _materialize_paths(
        self,
        parents: Dict[int, int],
        seeds: List[Dict[str, Any]],
        targets: List[Tuple[Any, str]],
    ) -> List[Dict[str, Any]]:
        """Walk parent pointers from a target back to its seed.

        Returns a list of ``{seed, target, hops, intermediates}`` where
        ``intermediates`` lists ``{hash_id, surface, vertex_type}`` for
        each hop strictly between seed and target. Targets that resolve
        to no vertex (e.g. a passage we filtered out by file_ids) are
        silently skipped.
        """
        graph = self._channel.graph
        passage_meta = self._passage_meta_to_hash()
        seed_idx_set = {s["vertex_idx"] for s in seeds}
        out: List[Dict[str, Any]] = []
        seen_targets: set = set()
        for target_key, kind in targets:
            target_vidx = self._target_vidx(target_key, kind, passage_meta)
            if target_vidx is None or target_vidx in seen_targets:
                continue
            # Skip targets that ARE seeds — a hops=0 path is degenerate
            # (we already ship seed info in ``seeds_resolved``).
            if target_vidx in seed_idx_set:
                continue
            seen_targets.add(target_vidx)
            chain = self._walk_to_seed(parents, target_vidx, seed_idx_set)
            if chain is None or len(chain) < 2:
                continue
            seed_v = chain[0]
            target_v = chain[-1]
            intermediates = [
                self._render_node(graph.vs[v]) for v in chain[1:-1]
            ]
            out.append(
                {
                    "seed": self._render_node(graph.vs[seed_v]),
                    "target": self._render_node(graph.vs[target_v]),
                    "hops": len(chain) - 1,
                    "intermediates": intermediates,
                }
            )
        return out

    def _target_vidx(
        self,
        target_key: Any,
        kind: str,
        passage_meta: Dict[Tuple[str, Optional[int]], str],
    ) -> Optional[int]:
        if kind == "entity":
            return self._channel._name_to_vidx.get(str(target_key))
        if kind == "passage":
            file_id, page_n = target_key
            hash_id = passage_meta.get((file_id, _coerce_int(page_n)))
            return self._channel._name_to_vidx.get(hash_id) if hash_id else None
        return None

    @staticmethod
    def _walk_to_seed(
        parents: Dict[int, int], start: int, seed_idx_set: set
    ) -> Optional[List[int]]:
        chain: List[int] = [start]
        cur = start
        # Bound the walk by the number of vertices to avoid pathological
        # cycles in the parent map (shouldn't happen, but cheap to guard).
        for _ in range(len(parents) + 1):
            if cur in seed_idx_set:
                chain.reverse()
                return chain
            nxt = parents.get(cur)
            if nxt is None or nxt == cur:
                return None
            chain.append(nxt)
            cur = nxt
        return None

    def _render_node(self, vertex) -> Dict[str, Any]:
        attrs = vertex.attributes()
        vtype = attrs.get("vertex_type") or self._guess_vertex_type(vertex["name"])
        node: Dict[str, Any] = {
            "hash_id": vertex["name"],
            "vertex_type": vtype,
        }
        surface = attrs.get("content")
        if surface:
            node["surface"] = surface
        if vtype == "passage":
            meta = self._passage_meta_lookup().get(vertex["name"])
            if meta:
                file_id, page_n = meta
                pn_int = _coerce_int(page_n)
                node["file_id"] = file_id
                if pn_int is not None:
                    node["page_id"] = f"p_{pn_int:04d}"
                    node["page_number"] = pn_int
        return node

    # ----------------------------------------------------------- entity_lookup

    def _run_entity_lookup(self, context: "AgentContext", **kwargs):
        surface = (kwargs.get("surface") or "").strip()
        if not surface:
            return err(
                "invalid_argument",
                "mode='entity_lookup' requires `surface`.",
                remediation="Pass `surface` as the entity name to look up (e.g. 'AXA' or 'AXA Hong Kong').",
                valid_example={"mode": "entity_lookup", "surface": "AXA"},
            ), {"error": "invalid_argument"}
        try:
            top_k_int = int(kwargs.get("top_k", 5))
        except (TypeError, ValueError):
            return err(
                "invalid_argument",
                "`top_k` must be an integer.",
                remediation="Pass `top_k` as a positive integer (default 5, max 10).",
                valid_example={"top_k": 5},
            ), {"error": "invalid_argument"}
        if top_k_int < 1:
            return err(
                "invalid_argument",
                "`top_k` must be >= 1.",
                remediation="Set `top_k` to a positive integer (default 5, max 10).",
                valid_example={"top_k": 5},
            ), {"error": "invalid_argument"}
        limit = min(top_k_int, 10)

        store = self._channel.entity_store
        if len(store) == 0:
            return (
                err(
                    "graph_unavailable",
                    "Entity store is empty; index the corpus first.",
                    remediation="Fall back to semantic_search / bm25_search / pattern_search — the entity layer is not built for this corpus.",
                ),
                {"error": "graph_unavailable"},
            )

        try:
            emb = self._channel.embedding_client.encode(surface)
        except Exception as exc:
            return (
                err(
                    "embed_failed",
                    f"Embedding the surface failed: {exc}",
                    remediation="Retry once; if the failure repeats, fall back to bm25_search or pattern_search using the surface form as the query.",
                ),
                {"error": "embed_failed"},
            )
        emb_arr = np.asarray(emb)
        if emb_arr.ndim == 2:
            emb_arr = emb_arr[0]
        # The disambiguator's 0.85 threshold is tuned for *adding* alias
        # edges (high precision, low recall); query-time lookup wants
        # something looser, but 0.4 surfaces too much noise (e.g. a
        # surface like "Premium Refund" matched a sim=0.4 product name).
        # 0.6 is the empirical sweet spot on this corpus; admin config
        # `graph_explore.entity_lookup_min_sim` overrides it per
        # corpus (see schema.py).
        cands = gradient_topk_candidates(
            emb_arr,
            store,
            k=limit,
            g=self._entity_lookup_gradient,
            min_sim=self._entity_lookup_min_sim,
        )
        if not cands:
            return (
                ok(
                    "GraphExploreObservation",
                    mode="entity_lookup",
                    surface=surface,
                    physical=[],
                    note=(
                        f"No entity above min similarity {self._entity_lookup_min_sim} "
                        f"— try a different surface form."
                    ),
                ),
                {"retrieved_tokens": 0, "physical_hits": 0},
            )

        cluster_idx = self._cluster_index_lookup()
        physical: List[Dict[str, Any]] = []
        for cand in cands:
            row = store.get_meta_row(cand.hash_id)
            phys_text = (row.get("text") or "").strip()
            cluster_info = cluster_idx.get(cand.hash_id)
            entry = {
                "hash_id": cand.hash_id,
                "surface": phys_text,
                "similarity": round(float(cand.score), 4),
            }
            if cluster_info is not None:
                entry["logical_cluster"] = cluster_info
            physical.append(entry)

        log_meta = {"mode": "entity_lookup", "surface": surface, "physical_hits": len(physical)}
        context.add_retrieval_log(tool_name="graph_explore", tokens=0, metadata=log_meta)

        return (
            ok(
                "GraphExploreObservation",
                mode="entity_lookup",
                surface=surface,
                physical=physical,
            ),
            {"retrieved_tokens": 0, "physical_hits": len(physical)},
        )

    # ----------------------------------------------------------- shared helpers

    def _passage_meta_lookup(self) -> Dict[str, Tuple[str, Optional[int]]]:
        if self._passage_meta_by_hash is None:
            store = self._channel.passage_store
            col_file_id = store.meta_column("file_id")
            col_page_n = store.meta_column("page_number")
            self._passage_meta_by_hash = {
                h: (f, p) for h, f, p in zip(store.hash_ids, col_file_id, col_page_n)
            }
        return self._passage_meta_by_hash

    def _passage_meta_to_hash(self) -> Dict[Tuple[str, Optional[int]], str]:
        if self._passage_hash_by_meta is None:
            self._passage_hash_by_meta = {
                (file_id, _coerce_int(page_n)): h
                for h, (file_id, page_n) in self._passage_meta_lookup().items()
            }
        return self._passage_hash_by_meta

    @staticmethod
    def _guess_vertex_type(hash_id: str) -> str:
        # When `vertex_type` is missing, derive it from the hash-id
        # namespace prefix (`entity-…`, `passage-…`, `sentence-…`).
        if hash_id.startswith("entity-"):
            return "entity"
        if hash_id.startswith("passage-"):
            return "passage"
        if hash_id.startswith("sentence-"):
            return "sentence"
        return "unknown"

    def _cluster_index_lookup(self) -> Dict[str, Dict[str, Any]]:
        """Map ``entity_hash → {cluster_id, canonical, members}``.

        Built lazily because the cache file may not exist on a corpus
        that was indexed before clusters were enabled.
        """
        if self._cluster_index is not None:
            return self._cluster_index
        idx: Dict[str, Dict[str, Any]] = {}
        try:
            clusters = get_clusters(self._channel.graph, self._clusters_cache_path)
        except Exception as exc:
            logger.warning("graph_explore: cluster cache load failed: %s", exc)
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


def _coerce_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
