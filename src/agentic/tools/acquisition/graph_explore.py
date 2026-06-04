"""Entity-graph retrieval over the LinearRAG-style graph.

Two declarative modes — the agent states intent in natural language
plus an optional anchor; it never fills numeric parameters.

* ``ppr`` — associative page retrieval. A free-text ``question`` drives
  personalized PageRank over the entity ↔ passage graph, delegated to
  :class:`rag.channels.GraphPPRChannel` so the agent path and the
  standalone-RAG path cannot drift (NER → seed entities → entity-score
  BFS → passage scoring → PPR). Returns ranked candidate pages with a
  question-conditioned snippet each.

* ``chain_entity`` — relational multi-hop + entity disambiguation,
  dispatched by argument shape:
    - ``question`` (± ``focus``) → query-time typed-edge beam search.
      Hops expand **sentence-first**: a tail entity's mention sentences
      are ranked by ``cos(question, sentence)`` and the entities they
      co-mention become the next hop (channel.cooccurrence_neighbors).
      The natural-language sentence IS the predicate, so the relation
      type is materialised per query without any build-time LLM. Paths
      are ranked by reciprocal-rank fusion over interpretable criteria
      (seed/focus coverage, weak-link and mean edge cosine, endpoint
      specificity) — no learned weights, no per-corpus tuned thresholds.
      Candidate pages = RRF(endpoint-incident pages, bridge co-occurrence
      pages); endpoint pages are admitted regardless of edge rank, since
      a 2-hop answer usually lives on the endpoint entity's own page.
    - ``focus`` only (no ``question``) → compact disambiguation audit of
      the anchor: the cluster(s) a surface maps to (or a ``cluster_id``
      from PPR's ``clusters_touched``), member surfaces, and top
      cross-doc pages. Full member-level alias-quality / repair audit is
      the maintenance API's job, not the QA agent's.
  ``focus`` accepts either a surface string or a ``cluster_id``. When two
  or more focus clusters resolve, the walk is target-aware: it always
  checks their direct co-occurrence and ranks paths that connect distinct
  focus clusters above open-ended exploration (comparison questions).

Physical vs logical entity:
  Each surface form is indexed as its own node ("AXA", "AXA Hong Kong",
  "安盛"). Alias edges between near-synonyms partition the entity layer
  into logical clusters; the cluster is the LOGICAL entity. Aliases fold
  surface variants into one logical hop — they are never walked as
  relations.

Pre-warming:
  GLiNER NER and igraph are heavy to load. The agent's
  ``warm_up()`` invokes :meth:`warm_up` on this tool to absorb that
  one-time cost before the user's first turn.

Lifetime assumption:
  The graph and passage stores are treated as immutable for the life
  of the agent process. ``_passage_meta_lookup`` is cached on the tool
  instance after first use. If the corpus is re-ingested mid-session,
  construct a fresh tool instance — the cache is not invalidated
  automatically.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from agentic.tools.acquisition._common import err, ok, parse_scope
from agentic.tools.base import BaseTool
from ingestion.index.linear_rag.backfill import find_literal_matches
from ingestion.index.linear_rag.normalize import normalize_for_hash
from rag.channels.base import RawHit, aggregate_per_doc, reciprocal_rank_fusion
from rag.channels.graph_ppr import GraphPPRChannel
from rag.preprocess import QueryContext
from storage.inventory_store import InventoryStore

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


logger = logging.getLogger(__name__)


_VALID_MODES = {"ppr", "chain_entity"}

# Agent observations over-return a fixed candidate pool (not a tuned
# top-k): a wide-enough slate so that when the agent calls both ``ppr``
# and ``chain_entity`` and reads, the union still contains the pages a
# deterministic rank fusion would have rescued — truncating to ~5 would
# destroy that. This is a token-budget cap, not a quality threshold.
_OVER_RETURN = 30
# How many chars of passage text to expose per candidate as a preview.
# Just enough for the agent to discriminate "yes this is the right
# page, read it" from "wrong page, skip". The full page comes via
# the ``read`` tool.
_PPR_PREVIEW_CHARS = 240

# Chain beam resource caps — latency/token budget, NOT tuned knobs.
_DEFAULT_CHAIN_DEPTH = 2      # 2Wiki-style two-hop; fixed, never adaptive
_CHAIN_TOP_S_SENTENCES = 32   # query-ranked tail sentences scanned / hop
_CHAIN_TOP_L_NEIGHBORS = 20   # co-occurrence neighbors kept / tail / hop
_CHAIN_TOP_M_SENTENCES = 8    # via-sentences rendered per edge
_CHAIN_BEAM_K = 32            # beam width (paths kept between hops)
_CHAIN_MAX_SEEDS = 8          # union-seed fan-out cap
_CHAIN_TOP_PATHS = 10         # paths surfaced to the agent

# Reciprocal-rank-fusion constant (Cormack et al., SIGIR 2009): published,
# unsupervised, corpus-independent. The ONLY constant in path/page
# scoring — there are no learned weights and no per-corpus tuned
# thresholds anywhere in chain ranking.
_RRF_K = 60


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
        # (_ENTITY_LOOKUP_MIN_SIM=0.6, gradient g=0.5) keep call sites
        # that construct the tool directly (experiment scripts) working
        # without any config plumbing.
        self._entity_lookup_min_sim = float(entity_lookup_min_sim)
        # Reserved for config compatibility: unused by the current
        # entity-lookup modes. Kept so the admin config key still binds
        # without a factory / schema change.
        self._entity_lookup_gradient = float(entity_lookup_gradient)
        # Passage meta is stable for the life of the agent (graph is built
        # once at ingest). Cache hash → (file_id, page_number) so chain / ppr
        # candidate-page rendering doesn't rebuild it each call.
        self._passage_meta_by_hash: Optional[Dict[str, Tuple[str, Optional[int]]]] = None

    @property
    def name(self) -> str:
        return "graph_explore"

    # ---------------------------------------------------------- invalidate

    def invalidate_caches(self) -> None:
        """Drop the per-instance derived cache.

        ``_passage_meta_by_hash`` reflects a snapshot of the channel's
        passage_store at first use; after a reingest or delete the store has
        different rows and the cached map returns stale (file_id,
        page_number) pairs.

        The wrapping ``GraphPPRChannel.reload()`` rebuilds the underlying
        stores; this method is the second half of "make a stale tool
        instance look fresh again" and should be called from the
        lifespan refresh hook.
        """
        self._passage_meta_by_hash = None

    # ----------------------------------------------------------- warm-up

    def warm_up(self) -> None:
        """Pre-load GLiNER NER and force igraph to be in memory.

        Both the PPR and chain modes need the NER model on the first
        call, which is the single biggest source of cold-start latency
        in the agent loop (~3-10 s for GLiNER multi-v2.1 depending on
        HF cache warmth). Run this before the agent's first user turn.
        """
        if self._channel.graph is None:
            return  # nothing to warm — graph not built
        try:
            self._channel._ensure_ner()
        except Exception as exc:
            logger.warning("graph_explore: NER warm-up failed: %s", exc)

    # ----------------------------------------------------------- schema

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "graph_explore",
                "description": (
                    "Navigate the entity knowledge graph. Two modes:\n"
                    "- ppr: associative retrieval. Give `question` "
                    "(free text) -> ranked candidate pages, each with a "
                    "question-conditioned snippet. Use for 'which pages "
                    "are about X'.\n"
                    "- chain_entity: relational multi-hop + entity "
                    "disambiguation. Give `question` to follow relations "
                    "across entities (the system finds the bridge and "
                    "answer pages itself). Optionally add `focus` (entity "
                    "name(s) or a cluster_id) to anchor; for a comparison "
                    "put BOTH entities in `focus`. Give `focus` alone (no "
                    "`question`) to look up what an entity is + where it "
                    "appears. The system chooses all search depth/breadth."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": sorted(_VALID_MODES),
                        },
                        "question": {
                            "type": "string",
                            "description": "Free-text natural-language query (ppr; or chain_entity for a relational/multi-hop question).",
                        },
                        "focus": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional anchor entity name(s) or cluster_id(s) (chain_entity). Two or more entries = comparison/bridge between them.",
                        },
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional file id allow-list.",
                        },
                        "page_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional [start, end] inclusive page filter (mode=ppr).",
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
                    remediation="Set `mode` to 'ppr' (associative page retrieval) or 'chain_entity' (relational multi-hop + entity disambiguation).",
                    valid_example={"mode": "chain_entity", "question": "..."},
                ),
                {"error": "invalid_argument"},
            )
        if self._channel.graph is None:
            return (
                err(
                    "graph_unavailable",
                    "LinearRAG graph is not built; ingest the corpus first.",
                    remediation="The entity graph is not available in this corpus.",
                ),
                {"error": "graph_unavailable"},
            )

        if mode == "ppr":
            return self._run_ppr(context, **kwargs)
        return self._run_chain_entity(context, **kwargs)

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
            inventory=self.inventory,
        )
        if scope_err is not None:
            return err(
                "invalid_argument",
                scope_err,
                remediation="Fix the scope arguments per the message: file_ids must come from list_files; page_range must be [start, end].",
                valid_example={"file_ids": ["<file_id>"]},
            ), {"error": "invalid_argument"}

        # No agent-facing top_k: over-return a fixed candidate pool so the
        # agent (and any downstream rank fusion) sees a wide enough slate.
        limit = _OVER_RETURN

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
                remediation="Retry with a simpler question; if the PPR channel keeps failing, switch to mode=chain_entity (with `focus` set to a known entity name).",
            ), {"error": "ppr_failed"}

        # Apply the page_range gate post-hoc (PPR channel honors file_ids
        # but not page_range — the channel was designed for the global RAG
        # path which doesn't expose ranges). Reuse the cached meta map.
        # IMPORTANT: keep ``kept_hits`` (the ChannelHit slice that
        # passed scope) in lock-step with ``results``, because
        # _attach_previews / per-page provenance below ZIP these two
        # together to look up evidence.passage_hash_id per hit.
        # If these two drift apart (e.g. zipping unfiltered hits with
        # the filtered ``results``), scoped queries silently shift
        # preview/clusters_touched onto the wrong page.
        meta_by_hash = self._passage_meta_lookup()

        # Pass 1: scope-filter the channel hits without truncating to
        # ``limit`` yet — the doc-aware rerank below needs the full
        # eligible slate so that a doc with several lower-ranked pages
        # can still outrank a doc with one stronger page that would
        # otherwise be the only one to fit in the top-K.
        eligible: List[Tuple[Any, Optional[int]]] = []
        for hit in channel_hits:
            page_number = self._first_page_number(hit, meta_by_hash)
            if scope.contains(hit.file_id, page_number):
                eligible.append((hit, page_number))

        # Cross-doc duplicate guard: when several pages from the same
        # doc each score independently, aggregate them via
        # ``Σ s_i / sqrt(N + 1)`` (rag.channels.base.aggregate_per_doc)
        # and re-sort pages by (doc_score, original_page_score). Pages
        # from the highest-aggregated doc rise to the top; pages within
        # one doc keep their PPR order. Targets the failure mode where
        # a near-duplicate from a wrong doc outranks the gold doc's
        # several supporting pages on raw PPR score alone.
        eligible = self._doc_aware_rerank(eligible, limit)

        # Per-doc page roll-up over the FULL eligible slate (pre-truncation
        # so docs with several lower-ranked pages still surface in the
        # rollup). Powers the ``docs_summary`` field — the agent's main
        # batch-read signal: "doc X has K pages in the top-K, range
        # p_lo..p_hi" lets it issue one ``read(unit_ids=[…])`` for all
        # siblings in one shot, instead of reading only 1-2 of several
        # gold pages on the same doc.
        doc_pages_full: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for hit, page_number in eligible:
            if page_number is None:
                continue
            doc_pages_full[hit.file_id].append(
                (int(page_number), float(hit.score))
            )

        results: List[Dict[str, Any]] = []
        kept_hits: List[Any] = []
        for hit, page_number in eligible:
            results.append(
                {
                    "file_id": hit.file_id,
                    "page_id": hit.page_id,
                    "page_number": page_number,
                    "score": round(float(hit.score), 6),
                }
            )
            kept_hits.append(hit)
            if len(results) >= limit:
                break

        # Cap doc_pages_full to the doc set that actually appears in
        # ``results`` (i.e. survives the top-K cut). Including docs the
        # agent can't reach via the visible candidate list is noise.
        visible_docs = {r["file_id"] for r in results}
        doc_pages_full = {
            fid: ps for fid, ps in doc_pages_full.items() if fid in visible_docs
        }

        seeds_dbg = debug_snapshot.get("seeds", [])
        # Attach question-conditioned previews — single highest
        # cos(q_emb, sent_emb) sentence per candidate page. Decisive
        # for the agent's first read selection versus a first-N-chars
        # preview, which surfaces table headers or references
        # regardless of the question.
        results_with_preview = self._attach_previews(results, kept_hits, question=question)

        # Seed surfaces (lowercased once) — used for per-page
        # ``supported_by`` and the global ``seeds_unsupported`` diagnostic.
        # Substring check, same heuristic ``_calculate_passage_scores``
        # uses internally — cheap on top-K * |seeds| ~= 5 * 8.
        seed_surfaces: List[str] = []
        for s in (seeds_dbg if isinstance(seeds_dbg, list) else []):
            surf = (s.get("surface") or "").strip()
            if surf:
                seed_surfaces.append(surf)
        seeds_with_any_support: set = set()

        # L3 + L6: per-page logical-cluster provenance + corpus breadth
        # (pages_in_cluster) so the agent can distinguish a specific
        # anchor entity (≤ 5 pages) from a hub cluster spanning the
        # corpus (≥ 100 pages) that PPR drifted into.
        for r, hit in zip(results_with_preview, kept_hits):
            ph = None
            for ev in (getattr(hit, "evidence", None) or []):
                if isinstance(ev, dict) and ev.get("passage_hash_id"):
                    ph = ev["passage_hash_id"]
                    break
            # ``supported_by`` — which seed surfaces actually appear on
            # this passage's text. A page with high PPR score but no
            # seed support was brought in by graph diffusion through a
            # hub, not by direct match — a strong demote signal that
            # the agent today can only guess from the preview.
            page_text = ""
            if ph is not None:
                page_text = self._channel.passage_store.hash_id_to_text.get(ph, "") or ""
            page_text_lower = page_text.lower()
            supported: List[str] = []
            for surf in seed_surfaces:
                if surf.lower() in page_text_lower:
                    supported.append(surf)
                    seeds_with_any_support.add(surf)
            # Cap to top 3 (most informative; whole list bloats the obs).
            r["supported_by"] = supported[:3]

            if ph is None:
                r["clusters_touched"] = []
                continue
            try:
                clusters = self._channel.passage_top_clusters(ph, top_n=3)
                for c in clusters:
                    cid = c.get("cluster_id")
                    if cid:
                        c["pages_in_cluster"] = self._channel.cluster_passage_count(cid)
                r["clusters_touched"] = clusters
            except Exception:
                r["clusters_touched"] = []

        # ``seeds_unsupported`` — seeds that activated PPR but whose
        # surface never appears in any top-K candidate's text. Telegraphs
        # "your seed `Saint-Galmier` never landed in the surfaced pages"
        # so the agent knows to switch tactic (chain_entity with a `focus`)
        # instead of reformulating the same PPR query.
        seeds_unsupported = [
            s for s in seed_surfaces if s not in seeds_with_any_support
        ]

        # Per-doc rollup. Each row carries:
        #  * ``pages_in_topk`` + ``page_numbers`` + ``page_span``
        #    — the visible-from-PPR subset
        #  * ``total_pages_in_doc`` + ``pages_not_in_topk_sample`` —
        #    so the agent sees "PPR surfaced 2 of 11 pages this doc
        #    has; 9 more exist (e.g. p_3, p_4, p_8, ...)" and can
        #    batch-read the missing siblings without a second PPR
        #  * ``dominant_cluster`` — the cluster_id occurring most
        #    often across the doc's visible pages, plus its surface;
        #    distinguishes "Gignac bio" from "Florida Panthers bio"
        #    without reading p_1 of each
        per_page_clusters: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = {
            (r["file_id"], r.get("page_number")): r.get("clusters_touched") or []
            for r in results_with_preview
        }
        channel = self._channel
        docs_summary: List[Dict[str, Any]] = []
        for fid, ps in doc_pages_full.items():
            ps_sorted = sorted(ps, key=lambda t: t[0])
            page_numbers = [pn for pn, _ in ps_sorted]
            total = round(sum(s for _, s in ps), 6)
            total_pages = channel.doc_page_count(fid)
            not_in_topk = (
                [pn for pn in range(1, total_pages + 1) if pn not in page_numbers]
                if total_pages > 0 else []
            )
            # Cap the missing-pages list to keep token cost in check
            # on very long docs; the first 15 are the most informative
            # ("contiguous near the surfaced range" or "all of them").
            not_in_topk_short = not_in_topk[:15]
            cluster_counts: Dict[str, int] = defaultdict(int)
            cluster_surface: Dict[str, str] = {}
            for pn in page_numbers:
                for c in per_page_clusters.get((fid, pn), []):
                    cid = c.get("cluster_id")
                    if not cid:
                        continue
                    cluster_counts[cid] += 1
                    if cid not in cluster_surface:
                        cluster_surface[cid] = c.get("top_surface") or ""
            dominant = None
            if cluster_counts:
                cid_best, n_pages = max(cluster_counts.items(), key=lambda kv: kv[1])
                dominant = {
                    "cluster_id": cid_best,
                    "top_surface": cluster_surface.get(cid_best, ""),
                    "pages_anchored": n_pages,
                }
            row: Dict[str, Any] = {
                "file_id": fid,
                "pages_in_topk": len(page_numbers),
                "total_pages_in_doc": total_pages,
                "page_numbers": page_numbers,
                "page_span": [page_numbers[0], page_numbers[-1]] if page_numbers else [],
                "total_score": total,
            }
            if not_in_topk_short:
                row["pages_not_in_topk_sample"] = not_in_topk_short
            if dominant is not None:
                row["dominant_cluster"] = dominant
            docs_summary.append(row)
        docs_summary.sort(key=lambda d: d["total_score"], reverse=True)
        docs_summary = docs_summary[:5]  # token cap
        # L2: top logical clusters by PPR mass.  Surfaced as DIAGNOSTIC
        # CONTEXT only — LLMs navigate poorly via opaque cluster IDs
        # (exposing them as the primary navigation signal regresses
        # answer quality). The five entries here let the agent notice
        # when PPR drifts onto a hub cluster off-topic without being
        # expected to follow `cluster_id` back.
        cluster_scores = debug_snapshot.get("cluster_scores", {}) or {}
        top_logical = []
        for cid, mass in sorted(
            cluster_scores.items(), key=lambda kv: kv[1], reverse=True
        )[:5]:
            top_surf = self._channel.cluster_top_surfaces(cid, top_n=1)
            members = self._channel._cluster_cache.get(cid) if self._channel._cluster_cache else None
            top_logical.append({
                "cluster_id": cid,
                "top_surface": top_surf[0]["surface"] if top_surf
                               else self._channel.entity_store.hash_id_to_text.get(cid, cid),
                "mass": round(float(mass), 6),
                "members_n": len(members) if members else 1,
            })
        log_meta = {
            "mode": "ppr",
            "question": question,
            "scope": scope.as_dict(),
            "seeds": [s["surface"] for s in seeds_dbg] if isinstance(seeds_dbg, list) else [],
            "hits": len(results_with_preview),
        }
        context.add_retrieval_log(tool_name="graph_explore", tokens=0, metadata=log_meta)

        # Surface compact cluster info (L2/L3): the agent's whole task
        # IS logical-entity navigation. ``top_logical_clusters`` is the
        # decompressed form, ``clusters_touched`` per page localizes
        # which logical entities live on each candidate.
        # ``unresolved`` — telegraphs "this PPR call did not anchor on
        # any specific entity from the question". Fires when every
        # activated seed surface was unsupported in the top-K AND the
        # leading logical cluster's mass is below a low floor. Signals
        # to the agent: do NOT answer-from-noise on this PPR slate;
        # either pivot to chain_entity with `focus` set to a more
        # specific entity, or stop and ask for disambiguation. False when no
        # seeds at all (different failure mode — "no_seeds_skip").
        _top_cluster_mass = max(
            (float(c.get("mass") or 0.0) for c in top_logical), default=0.0
        )
        unresolved = (
            len(seed_surfaces) > 0
            and len(seeds_unsupported) == len(seed_surfaces)
            and _top_cluster_mass < 0.02
        )

        return (
            ok(
                "GraphExploreObservation",
                mode="ppr",
                question=question,
                scope=scope.as_dict(),
                seeds=[
                    {"surface": s.get("surface"), "sim": s.get("sim")}
                    for s in (seeds_dbg if isinstance(seeds_dbg, list) else [])
                ][:8],  # cap seed list — long NER seeds also bloat
                seeds_unsupported=seeds_unsupported,
                unresolved=unresolved,
                top_logical_clusters=top_logical,
                docs_summary=docs_summary,
                candidate_pages=results_with_preview,
            ),
            {"retrieved_tokens": 0, "hits": len(results_with_preview)},
        )

    def _attach_previews(
        self,
        results: List[Dict[str, Any]],
        channel_hits: Optional[List[Any]] = None,
        *,
        question: str = "",
    ) -> List[Dict[str, Any]]:
        """Add a question-conditioned ``preview`` field per candidate
        page — the single sentence on that page that maximises
        cos(q_emb, sent_emb). The agent uses it for fine-grained page
        selection before a ``read`` call; question-conditioned snippets
        flip "this candidate is about my topic" from a guess to a
        direct signal, unlike a first-N-chars preview.

        Channel artifact path: each page's sentences + their
        embeddings come from
        :meth:`GraphPPRChannel.passage_sentence_embs`, built once from
        the ingest-persisted ``passage_hash_id_to_sentences`` map and
        the sentence_store. The query embedding is encoded once per
        call and reused across every candidate page.

        Prefers ``channel_hits[i].evidence[0]["passage_hash_id"]`` for
        an exact passage match. Falls back to a one-time-built
        ``(file_id, page_number) → passage_hash`` map when evidence
        is unavailable (older trace formats / non-PPR modes).
        """
        from agentic.tools.acquisition._preview import query_snippet

        if not results:
            return results
        out: List[Dict[str, Any]] = []
        channel = self._channel
        page_store = getattr(channel, "passage_store", None)
        embed_client = getattr(channel, "embedding_client", None)

        # Encode the query once and reuse across every candidate page.
        q_emb = None
        if question and embed_client is not None:
            try:
                q_emb = embed_client.encode(question, is_query=True)
                if q_emb.ndim == 2:
                    q_emb = q_emb[0]
            except Exception:
                q_emb = None

        # Build the reverse meta map lazily — only if we end up needing
        # the fallback path (i.e. some hit lacks evidence).
        meta_map: Optional[Dict[Tuple[str, Optional[int]], str]] = None
        for i, hit_meta in enumerate(results):
            if page_store is None or not hasattr(page_store, "hash_id_to_text"):
                out.append({**hit_meta, "preview": ""})
                continue
            # Path 1: exact passage_hash_id from ChannelHit.evidence.
            passage_hash: Optional[str] = None
            if channel_hits and i < len(channel_hits):
                ev = getattr(channel_hits[i], "evidence", None) or []
                for e in ev:
                    h = e.get("passage_hash_id") if isinstance(e, dict) else None
                    if h:
                        passage_hash = h
                        break
            # Path 2: fallback via (file_id, page_number).  When a
            # page is split into multiple passages, a plain dict-comp
            # would keep the LAST passage's hash; use ``setdefault``
            # so the FIRST passage on each page wins — that's
            # typically the page heading / lead paragraph, which is
            # a better disambiguation preview than a mid-page span.
            if passage_hash is None:
                if meta_map is None:
                    meta_map = {}
                    try:
                        col_file_id = page_store.meta_column("file_id")
                        col_page_number = page_store.meta_column("page_number")
                        for h, fid, pn in zip(
                            page_store.hash_ids, col_file_id, col_page_number
                        ):
                            key_pn = int(pn) if pn is not None else None
                            meta_map.setdefault((str(fid), key_pn), h)
                    except Exception:
                        meta_map = {}
                key = (str(hit_meta["file_id"]),
                       int(hit_meta["page_number"]) if hit_meta.get("page_number") is not None else None)
                passage_hash = meta_map.get(key)

            page_text = ""
            cached_sentences = None
            if passage_hash is not None:
                page_text = page_store.hash_id_to_text.get(passage_hash, "") or ""
                # passage_sentence_embs is a one-time disk read + dict
                # translation; any failure (missing ner_results.json,
                # sentence_store schema drift) should degrade to the
                # slow-path snippet, not lose the whole observation.
                try:
                    cached_sentences = channel.passage_sentence_embs(passage_hash)
                except Exception:
                    cached_sentences = None
            try:
                preview = query_snippet(
                    page_text,
                    question,
                    embed_client,
                    max_chars=_PPR_PREVIEW_CHARS,
                    cached_query_emb=q_emb,
                    cached_sentences=cached_sentences,
                )
            except Exception:
                preview = (" ".join(page_text.split())[:_PPR_PREVIEW_CHARS]
                           if page_text else "")
            out.append({**hit_meta, "preview": preview})
        return out

    @staticmethod
    def _doc_aware_rerank(
        eligible: List[Tuple[Any, Optional[int]]],
        limit: int,
    ) -> List[Tuple[Any, Optional[int]]]:
        """Reorder scope-filtered PPR hits so pages from the strongest
        document (in the ``Σ s_i / sqrt(N+1)`` sense) come first.

        No-op when at most one doc is present or the eligible slate
        already fits in ``limit`` from a single doc; the reordering
        only matters when several pages from one or more docs compete
        for the top slots. Within a doc, the page order from PPR is
        preserved (secondary sort by original score).
        """
        if not eligible:
            return eligible
        distinct_docs = {h.file_id for h, _ in eligible}
        if len(distinct_docs) <= 1:
            return eligible
        raw = [RawHit(file_id=h.file_id, page_id=h.page_id, score=float(h.score))
               for h, _ in eligible]
        doc_hits = aggregate_per_doc(raw, top_k=len(distinct_docs))
        doc_score = {d.file_id: d.score for d in doc_hits}
        return sorted(
            eligible,
            key=lambda hp: (doc_score.get(hp[0].file_id, 0.0), float(hp[0].score)),
            reverse=True,
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

    # ----------------------------------------------------------- chain mode

    def _run_chain_entity(self, context: "AgentContext", **kwargs):
        """Relational multi-hop + entity disambiguation (one mode, by shape).

        ``question`` (± ``focus``) → sentence-first co-occurrence beam walk.
        ``focus`` only (no ``question``) → compact disambiguation audit of
        the anchor. ``focus`` accepts surface strings or ``cluster_id``\\ s;
        two or more focus clusters make the walk target-aware.
        """
        question = (kwargs.get("question") or "").strip()
        focus_raw = kwargs.get("focus") or []
        if isinstance(focus_raw, str):
            focus_raw = [focus_raw]
        focus = [f.strip() for f in focus_raw if isinstance(f, str) and f.strip()]
        if not question and not focus:
            return err(
                "invalid_argument",
                "mode='chain_entity' needs `question` and/or `focus`.",
                remediation=(
                    "Give `question` for a relational/multi-hop query, "
                    "`focus=[entity]` to look up one entity, or "
                    "`focus=[A, B]` (+ `question`) to bridge/compare two."
                ),
                valid_example={
                    "mode": "chain_entity",
                    "question": "Who is the spouse of the director of Inception?",
                },
            ), {"error": "invalid_argument"}
        channel = self._channel
        if len(channel.entity_store) == 0:
            return err(
                "graph_unavailable",
                "Entity store is empty; index the corpus first.",
                remediation="The entity layer is not available in this corpus.",
            ), {"error": "graph_unavailable"}

        focus_seeds, focus_clusters, focus_audit = self._resolve_focus(focus)

        if not question:
            return self._run_focus_audit(context, focus, focus_audit)
        return self._run_chain_walk(
            context, question, focus_seeds, focus_clusters, focus_audit,
            file_ids=kwargs.get("file_ids"), page_range=kwargs.get("page_range"),
        )

    def _resolve_focus(
        self, focus: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        """Resolve focus tokens (surface | cluster_id) to seeds + clusters.

        Returns ``(seeds, cluster_ids, audit)``:
          - ``seeds`` — one representative chain seed per token
          - ``cluster_ids`` — distinct focus clusters (target-aware ranking)
          - ``audit`` — compact handle ``{token, kind, cluster_id, surface}``
            per token, for the focus-only audit path
        """
        channel = self._channel
        clusters, member_to_cluster = channel._load_clusters_cached()
        text = channel.entity_store.hash_id_to_text
        seeds: List[Dict[str, Any]] = []
        cluster_ids: List[str] = []
        audit: List[Dict[str, Any]] = []
        surface_tokens: List[str] = []
        for tok in focus:
            if tok in clusters:
                members = clusters.get(tok) or []
                rep = members[0] if members else None
                if rep is not None and rep in channel._name_to_vidx:
                    seeds.append({
                        "hash_id": rep, "sim": 1.0,
                        "surface": text.get(rep, ""), "source": "focus_cluster",
                    })
                if tok not in cluster_ids:
                    cluster_ids.append(tok)
                audit.append({
                    "token": tok, "kind": "cluster_id", "cluster_id": tok,
                    "surface": text.get(rep, "") if rep else "",
                })
            elif tok in channel._name_to_vidx:
                # Singleton cluster_id: PPR's clusters_touched surfaces a
                # singleton logical entity as its own entity-hash (the locked
                # A condition — agent passes that hash straight back as focus).
                cid = member_to_cluster.get(tok, tok)
                seeds.append({
                    "hash_id": tok, "sim": 1.0,
                    "surface": text.get(tok, ""), "source": "focus_cluster",
                })
                if cid not in cluster_ids:
                    cluster_ids.append(cid)
                audit.append({
                    "token": tok, "kind": "cluster_id", "cluster_id": cid,
                    "surface": text.get(tok, ""),
                })
            else:
                surface_tokens.append(tok)
        if surface_tokens:
            for s in self._chain_seeds_from_surfaces(surface_tokens, source_tag="focus"):
                # Disclosed sanity floor for an explicitly-named anchor: a
                # deliberate focus surface that only weakly matches any entity
                # should not anchor the walk (admin-injectable
                # `entity_lookup_min_sim`). Weak matches are surfaced as
                # low-confidence in the audit but not seeded.
                if s["sim"] < self._entity_lookup_min_sim:
                    audit.append({
                        "token": s["surface"], "kind": "surface",
                        "cluster_id": None, "surface": s["surface"],
                        "note": "low_confidence_match", "sim": round(float(s["sim"]), 4),
                    })
                    continue
                seeds.append(s)
                cid = member_to_cluster.get(s["hash_id"], s["hash_id"])
                if cid not in cluster_ids:
                    cluster_ids.append(cid)
                audit.append({
                    "token": s["surface"], "kind": "surface",
                    "cluster_id": cid, "surface": s["surface"],
                })
        return seeds, cluster_ids, audit

    def _run_focus_audit(self, context: "AgentContext", focus, focus_audit):
        """Focus-only path: compact disambiguation audit per token."""
        entries: List[Dict[str, Any]] = []
        for h in focus_audit:
            cid = h.get("cluster_id")
            if not cid:
                entries.append({"token": h.get("token"), "note": "unresolved"})
                continue
            entries.append(self._compact_cluster_audit(cid, token=h.get("token", "")))
        log_meta = {"mode": "chain_entity", "focus": focus, "audit": len(entries)}
        context.add_retrieval_log(tool_name="graph_explore", tokens=0, metadata=log_meta)
        return (
            ok(
                "GraphExploreObservation",
                mode="chain_entity",
                focus=focus,
                entity_audit=entries,
                paths=[],
                candidate_pages=[],
            ),
            {"retrieved_tokens": 0, "audit": len(entries)},
        )

    def _compact_cluster_audit(self, cluster_id: str, *, token: str = "") -> Dict[str, Any]:
        """Compact disambiguation audit: cluster identity + member surfaces +
        top cross-doc pages + co-occurring clusters. Enough to disambiguate
        and to answer safely; the full member-level alias-quality / repair /
        reversibility audit is the maintenance API's job, not the QA agent's.
        """
        ch = self._channel
        clusters, _ = ch._load_clusters_cached()
        members = clusters.get(cluster_id) or [cluster_id]
        canonical = ch.entity_store.hash_id_to_text.get(members[0], "") if members else ""
        try:
            top_members = ch.cluster_top_surfaces(cluster_id, top_n=8)
        except Exception:
            top_members = []
        try:
            top_pages = ch.cluster_top_passages(cluster_id, top_n=5) if len(members) <= 50 else []
        except Exception:
            top_pages = []
        try:
            cooccurring = ch.cluster_cooccurrences(cluster_id, top_n=6)
        except Exception:
            cooccurring = []
        return {
            "token": token,
            "cluster_id": cluster_id,
            "canonical": canonical,
            "cluster_size": len(members),
            "top_members": top_members,
            "top_pages": top_pages,
            "cooccurring": cooccurring,
            "audit_recommended": len(members) > 50,
        }

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
                "Sentence store is empty; chain_entity needs a sentence layer.",
                remediation="The sentence layer is not available in this corpus.",
            ), {"error": "graph_unavailable"}

        try:
            q_emb = channel.embedding_client.encode(question, is_query=True)
        except Exception as exc:
            logger.exception("graph_explore[chain_entity] embedding failed: %s", exc)
            return err(
                "embed_failed",
                f"Embedding the query failed: {exc}",
                remediation="Retry once; if it persists, switch to mode=ppr.",
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
                    mode="chain_entity",
                    question=question,
                    focus=[h.get("token") for h in focus_audit],
                    scope=scope.as_dict(),
                    seeds=[],
                    paths=[],
                    candidate_pages=[],
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
        entity_audit = [
            self._compact_cluster_audit(h["cluster_id"], token=h.get("token", ""))
            for h in focus_audit if h.get("cluster_id")
        ]

        log_meta = {
            "mode": "chain_entity", "question": question,
            "focus": [h.get("token") for h in focus_audit],
            "scope": scope.as_dict(),
            "seeds": [s["surface"] for s in seeds_info],
            "paths": len(paths_out), "candidate_pages": len(candidate_pages),
        }
        context.add_retrieval_log(tool_name="graph_explore", tokens=0, metadata=log_meta)
        return (
            ok(
                "GraphExploreObservation",
                mode="chain_entity",
                question=question,
                focus=[h.get("token") for h in focus_audit],
                scope=scope.as_dict(),
                seeds=[{
                    "hash_id": s["hash_id"], "surface": s["surface"],
                    "sim": round(float(s["sim"]), 4), "source": s["source"],
                } for s in seeds_info],
                entity_audit=entity_audit,
                paths=paths_out,
                candidate_pages=candidate_pages,
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
            logger.warning("graph_explore[chain_entity] question_ner failed: %s", exc)
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
            logger.warning("graph_explore[chain_entity] gazetteer build failed: %s", exc)
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

    def _chain_seeds_from_surfaces(
        self,
        surfaces: List[str],
        *,
        source_tag: str,
    ) -> List[Dict[str, Any]]:
        """Resolve free-text surfaces to entity hashes by embedding top-1.
        Used for ``focus`` anchor resolution. No absolute floor — the best
        match is taken and its similarity is reported as confidence.
        """
        if not surfaces:
            return []
        channel = self._channel
        text = channel.entity_store.hash_id_to_text
        out: List[Dict[str, Any]] = []
        for surf in surfaces:
            surf = (surf or "").strip()
            if not surf:
                continue
            try:
                vec = channel.embedding_client.encode(surf)
                if vec.ndim == 2:
                    vec = vec[0]
                top = channel.entity_store.topk(vec, 1)
            except Exception:
                continue
            if not top:
                continue
            hid, score = top[0]
            if hid not in channel._name_to_vidx:
                continue
            out.append({
                "hash_id": hid,
                "sim": float(score),
                "surface": text.get(hid, surf),
                "source": source_tag,
            })
        return out

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
                rep_surfs = channel.cluster_top_surfaces(cid, top_n=1)
                nodes_out.append({
                    "hash_id": h,
                    "surface": ent_text.get(h, ""),
                    "cluster_id": cid,
                    "cluster_top_surface": (
                        rep_surfs[0]["surface"] if rep_surfs else ent_text.get(h, "")
                    ),
                })
            edges_out = []
            for e in p["edges"]:
                via = []
                for sid in e["via_sids"]:
                    txt = sent_text.get(sid, "")
                    if not txt:
                        continue
                    via.append({
                        "sentence_id": sid,
                        "text": txt if len(txt) <= 220 else txt[:217] + "...",
                    })
                edges_out.append({
                    "from": e["tail"],
                    "to": e["head"],
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

    # ----------------------------------------------------------- helpers

def _coerce_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
