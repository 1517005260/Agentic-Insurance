"""``graph_ppr`` — associative page retrieval over the entity graph (PPR)."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from agentic.tools.acquisition._common import err, ok, parse_scope
from agentic.tools.acquisition.graph_explore.base import _GraphToolBase, _OVER_RETURN
from rag.preprocess import QueryContext

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


logger = logging.getLogger(__name__)


class GraphPprTool(_GraphToolBase):
    """Associative page retrieval over the entity graph (PPR)."""

    # ----------------------------------------------------------- PPR mode

    def _run_ppr(self, context: "AgentContext", **kwargs):
        question = (kwargs.get("question") or "").strip()
        if not question:
            return err(
                "invalid_argument",
                "graph_ppr requires `question`.",
                remediation="Add `question` (free-text natural-language query) to the call; PPR uses NER to seed the random walk.",
                valid_example={"question": "Which sections discuss premium rebates?"},
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
            logger.exception("graph_ppr failed: %s", exc)
            return err(
                "ppr_failed",
                f"PPR raised: {exc}",
                remediation="Retry with a simpler question; if the PPR channel keeps failing, switch to graph_chain (with `focus` set to a known entity name).",
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
        # Attach question-conditioned evidence — a query-centered sentence
        # window for the top rows, a single best sentence for the menu, each
        # with a full-read ``cost_tokens``. Decisive for the agent's first
        # read selection versus a first-N-chars preview, which surfaces table
        # headers or references regardless of the question. Per-run dedup via
        # ``context`` collapses an already-shown/read page to a stub.
        results_with_preview = self._attach_previews(
            results, kept_hits, question=question, context=context
        )

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
        # so the agent knows to switch tactic (graph_chain with a `focus`)
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
            # Single-page docs (MuSiQue/2wiki: file == passage) gain nothing
            # from a per-doc rollup — the page already rides in evidence /
            # more_candidates. Emit docs_summary only for genuine multi-page
            # docs (Double-Bench PDFs), where the missing-sibling-page list and
            # span are what drive in-doc navigation.
            if total_pages <= 1:
                continue
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
            "question": question,
            "scope": scope.as_dict(),
            "seeds": [s["surface"] for s in seeds_dbg] if isinstance(seeds_dbg, list) else [],
            "hits": len(results_with_preview),
        }
        context.add_retrieval_log(tool_name="graph_ppr", tokens=0, metadata=log_meta)

        # Surface compact cluster info (L2/L3): the agent's whole task
        # IS logical-entity navigation. ``top_logical_clusters`` is the
        # decompressed form, ``clusters_touched`` per page localizes
        # which logical entities live on each candidate.
        # ``unresolved`` — telegraphs "this PPR call did not anchor on
        # any specific entity from the question". Fires when every
        # activated seed surface was unsupported in the top-K AND the
        # leading logical cluster's mass is below a low floor. Signals
        # to the agent: do NOT answer-from-noise on this PPR slate;
        # either pivot to graph_chain with `focus` set to a more
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

        # Asymmetric observation: the top rows carry a query-centered
        # sentence ``window`` (``evidence``) so the strongest pages are
        # answerable in place, with ``cost_tokens`` telling what a full
        # ``read`` would add; the rest are a one-line menu
        # (``more_candidates``) whose ``top_entities`` are the human-readable
        # surface names of the logical entities on the page — a navigation
        # handle, never a raw cluster_id. A page already shown/read this run
        # rides in ``evidence`` as a ``seen`` stub (scroll up, don't re-emit).
        evidence: List[Dict[str, Any]] = []
        more_candidates: List[Dict[str, Any]] = []
        for r in results_with_preview:
            if r.get("seen"):
                evidence.append({
                    "file_id": r["file_id"],
                    "page_id": r.get("page_id"),
                    "page_number": r.get("page_number"),
                    "score": r["score"],
                    "seen": True,
                    "cost_tokens": r.get("cost_tokens", 0),
                    "supported_by": r.get("supported_by", []),
                    "clusters_touched": r.get("clusters_touched", []),
                })
            elif "window" in r:
                evidence.append({
                    "file_id": r["file_id"],
                    "page_id": r.get("page_id"),
                    "page_number": r.get("page_number"),
                    "score": r["score"],
                    "window": r["window"],
                    "cost_tokens": r.get("cost_tokens", 0),
                    "supported_by": r.get("supported_by", []),
                    "clusters_touched": r.get("clusters_touched", []),
                })
            else:
                top_entities = [
                    c.get("top_surface")
                    for c in (r.get("clusters_touched") or [])
                    if c.get("top_surface")
                ][:3]
                more_candidates.append({
                    "file_id": r["file_id"],
                    "page_number": r.get("page_number"),
                    "score": r["score"],
                    "preview": r.get("preview", ""),
                    "cost_tokens": r.get("cost_tokens", 0),
                    "top_entities": top_entities,
                })

        # Explicit relation stream: the chain tool emits per-edge via-sentences,
        # but PPR (the workhorse) returns only passages. Surface the strongest
        # query-relevant `A —[sentence]→ B` hops off the activated entities so a
        # multi-hop bridge is one observation away without a tool switch.
        relations = self._build_ppr_relations(question, seeds_dbg, top_logical)

        ok_kwargs: Dict[str, Any] = dict(
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
            evidence=evidence,
            more_candidates=more_candidates,
        )
        if relations:
            ok_kwargs["relations"] = relations

        return (
            ok("GraphExploreObservation", **ok_kwargs),
            {"retrieved_tokens": 0, "hits": len(results_with_preview)},
        )


    def _build_ppr_relations(
        self,
        question: str,
        seeds_dbg: List[Dict[str, Any]],
        top_logical: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Top query-relevant `A —[evidence sentence]→ B` hops off the
        activated entities, the same sentence-first co-occurrence the chain
        tool uses, but bolted onto PPR's observation so a multi-hop bridge is
        reachable without a tool switch.

        Tails = the resolved seed hashes plus the entity-hash representatives
        of the top logical clusters (cap ~3). For each tail, rank its mention
        sentences by query cosine (``cooccurrence_neighbors``) and emit the
        co-mentioned entity with its bridge sentence. Bounded and wrapped in
        try/except so any failure degrades to no ``relations`` field rather
        than breaking PPR.
        """
        try:
            channel = self._channel
            clusters, _ = channel._load_clusters_cached()
            ent_text = channel.entity_store.hash_id_to_text
            sent_text = channel.sentence_store.hash_id_to_text

            # Tail entity hashes: seeds first, then a member-hash of each top
            # cluster (a c_XXXX cluster id is not a tail; fold to a member).
            tails: List[str] = []
            seen_tail: set = set()
            for s in (seeds_dbg if isinstance(seeds_dbg, list) else []):
                hid = s.get("hash_id")
                if hid and hid in channel._name_to_vidx and hid not in seen_tail:
                    seen_tail.add(hid)
                    tails.append(hid)
            for c in top_logical:
                cid = c.get("cluster_id")
                if not cid:
                    continue
                rep = cid if cid in channel._name_to_vidx else None
                if rep is None:
                    members = clusters.get(cid) or []
                    rep = next((m for m in members if m in channel._name_to_vidx), None)
                if rep and rep not in seen_tail:
                    seen_tail.add(rep)
                    tails.append(rep)
            tails = tails[:3]
            if not tails:
                return []

            qemb = channel.embedding_client.encode(question, is_query=True)
            if qemb.ndim == 2:
                qemb = qemb[0]
            sids = channel.sentence_store.hash_ids
            sidx = {h: i for i, h in enumerate(sids)}
            ssims = channel.sentence_store.all_similarities(qemb)

            rels: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for tail in tails:
                from_surf = ent_text.get(tail, "")
                for nb in channel.cooccurrence_neighbors(tail, ssims, sidx, top_l=8):
                    to_surf = ent_text.get(nb["hash_id"], "")
                    if not from_surf or not to_surf:
                        continue
                    via_sids = nb.get("via_sids") or []
                    via = sent_text.get(via_sids[0], "") if via_sids else ""
                    if len(via) > 220:
                        via = via[:217] + "..."
                    key = (from_surf, to_surf)
                    cur = rels.get(key)
                    if cur is None or nb["max_cos"] > cur["cos"]:
                        rels[key] = {
                            "from": from_surf,
                            "to": to_surf,
                            "via_sentence": via,
                            "cos": round(float(nb["max_cos"]), 4),
                            "support": int(nb["support"]),
                        }
            return sorted(rels.values(), key=lambda r: r["cos"], reverse=True)[:8]
        except Exception as exc:
            logger.warning("graph_ppr relation stream failed: %s", exc)
            return []



    @property
    def name(self) -> str:
        return "graph_ppr"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "graph_ppr",
                "description": (
                    "Associative page retrieval over the entity graph. Give "
                    "a free-text `question` -> ranked pages: the top few with "
                    "a query-relevant `window` excerpt (`evidence`), the rest "
                    "as a one-line menu (`more_candidates`). Each shows "
                    "`cost_tokens` for a full `read`."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Free-text natural-language query; NER seeds the random walk.",
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
        if self._channel.graph is None:
            return (
                err(
                    "graph_unavailable",
                    "LinearRAG graph is not built; ingest the corpus first.",
                    remediation="The entity graph is not available in this corpus.",
                ),
                {"error": "graph_unavailable"},
            )
        return self._run_ppr(context, **kwargs)

