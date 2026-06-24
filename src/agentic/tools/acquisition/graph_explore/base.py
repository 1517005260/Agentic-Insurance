"""Entity-graph retrieval over the LinearRAG-style graph.

Three named tools share one base (:class:`_GraphToolBase`); each states
its intent in natural language plus an optional anchor and never fills
numeric parameters.

* ``graph_ppr`` — associative page retrieval. A free-text ``question``
  drives personalized PageRank over the entity ↔ passage graph,
  delegated to :class:`rag.channels.GraphPPRChannel` so the agent path
  and the standalone-RAG path cannot drift (NER → seed entities →
  entity-score BFS → passage scoring → PPR). The observation is
  asymmetric: the top pages carry a query-centered sentence ``window``
  (``evidence``), the rest a one-line menu (``more_candidates``); full
  text is reached only via ``read``.

* ``graph_chain`` — relational multi-hop. A ``question`` (± ``focus``)
  drives a query-time typed-edge beam search. Hops expand
  **sentence-first**: a tail entity's mention sentences are ranked by
  ``cos(question, sentence)`` and the entities they co-mention become
  the next hop (channel.cooccurrence_neighbors). The natural-language
  sentence IS the predicate, so the relation type is materialised per
  query without any build-time LLM.

* ``entity_inspect`` — entity disambiguation / neighborhood expansion.
  ``focus`` (surface string(s) or ``cluster_id``\\ s) → compact
  disambiguation audit.

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
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from agentic.tools.acquisition._common import ok
from agentic.tools.base import BaseTool
from ingestion.index.linear_rag.disambig import (
    build_surface_idf,
    idf_weighted_overlap,
    tokenize_surface,
)
from rag.channels.base import RawHit, aggregate_per_doc
from rag.channels.graph_ppr import GraphPPRChannel
from storage.inventory_store import InventoryStore

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


logger = logging.getLogger(__name__)


# Agent observations over-return a fixed candidate pool (not a tuned
# top-k): a wide-enough slate so that when the agent calls both
# ``graph_ppr`` and ``graph_chain`` and reads, the union still contains
# the pages a deterministic rank fusion would have rescued — truncating
# to ~5 would destroy that. This is a token-budget cap, not a quality
# threshold.
_OVER_RETURN = 30
# How many chars of passage text to expose per menu candidate as a
# preview. Just enough for the agent to discriminate "yes this is the
# right page, read it" from "wrong page, skip". The full page comes via
# the ``read`` tool.
_PPR_PREVIEW_CHARS = 240

# Evidence-tier policy. The top-K rows carry a query-centered sentence
# WINDOW (richer than one sentence so a small model can answer in place
# without a ``read``); lower-ranked rows keep the cheap 1-sentence menu
# preview so deep candidates (the agent reads rank up to ~21) stay
# discoverable. The full passage text is available only via ``read``.
# Token-budget caps, not quality thresholds.
# TODO admin panel: expose evidence-window k / sentences / max_chars.
_EVIDENCE_WINDOW_TOP_K = 5
_EVIDENCE_WINDOW_SENTENCES = 1
_EVIDENCE_WINDOW_MAX_CHARS = 400

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


def _hybrid_select(
    surf: str,
    candidates: List[Tuple[str, str, float]],
    idf: Dict[str, float],
    overlap_min: float,
    syn_sim: float,
) -> Optional[Dict[str, Any]]:
    """Resolve a surface against embedding-NN candidates by two signals.

    ``candidates`` is ``(hash_id, surface, emb_sim)`` sorted by ``emb_sim``
    descending. Embedding recall alone produces high-confidence WRONG
    matches ("Warner Bros. Records" → "hollywood records" at 0.901); the
    IDF-weighted lexical overlap is a precision gate of a different signal
    class (record-linkage), so a candidate that shares distinctive head
    tokens with the surface wins over a merely embedding-near one.

    Resolution order:

    * lexical — any candidate clears ``overlap_min``: take the one with
      the highest ``(overlap, emb_sim)``.
    * synonym escape — no lexical overlap but the top NN is at/above
      ``syn_sim``: trust the embedding (true synonym with no shared
      surface tokens, e.g. an acronym ↔ expansion).
    * abstain — neither holds: return None so the caller does not seed a
      confidently-wrong match.

    Pure (no channel state) so it is unit-testable on synthetic candidates.
    """
    if not candidates:
        return None
    toks_surf = tokenize_surface(surf.lower())
    scored: List[Tuple[float, str, str, float]] = []
    for hid, cand_surf, emb_sim in candidates:
        ov = idf_weighted_overlap(
            toks_surf, tokenize_surface((cand_surf or "").lower()), idf
        )
        scored.append((ov, hid, cand_surf, emb_sim))
    best_lexical = max(scored, key=lambda t: (t[0], t[3]))
    if best_lexical[0] > overlap_min:
        ov, hid, cand_surf, emb_sim = best_lexical
        return {
            "hash_id": hid,
            "surface": cand_surf,
            "sim": emb_sim,
            "overlap": ov,
            "resolved_by": "lexical",
        }
    top_hid, top_surf, top_sim = candidates[0]
    if top_sim >= syn_sim:
        top_ov = next(ov for ov, hid, _, _ in scored if hid == top_hid)
        return {
            "hash_id": top_hid,
            "surface": top_surf,
            "sim": top_sim,
            "overlap": top_ov,
            "resolved_by": "synonym",
        }
    return None


class _GraphToolBase(BaseTool):
    """Shared graph machinery for the three named graph tools.

    Holds the channel reference, caches, NER warm-up, and the focus-audit /
    seed-resolution / passage-meta helpers shared by ``GraphPprTool``,
    ``GraphChainTool`` and ``EntityInspectTool``. Each concrete tool supplies
    its own ``name`` / ``get_schema`` / ``execute`` plus its mode-specific
    machinery (PPR in ``ppr.py``, the chain walk in ``chain.py``).
    """

    def __init__(
        self,
        channel: Optional[GraphPPRChannel] = None,
        inventory: Optional[InventoryStore] = None,
        *,
        entity_lookup_min_sim: float = 0.6,
        entity_lookup_gradient: float = 0.5,
        er_seed_overlap_min: float = 0.15,
        er_seed_synonym_sim: float = 0.90,
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
        # Hybrid seed-resolution tunables — embedding recall is paired with an
        # IDF lexical-overlap precision gate (``er_seed_overlap_min``) plus a
        # synonym escape (``er_seed_synonym_sim``) so a confidently-wrong NN
        # match cannot anchor a chain.
        # TODO admin panel: expose er_seed_overlap_min / er_seed_synonym_sim.
        self._er_seed_overlap_min = float(er_seed_overlap_min)
        self._er_seed_synonym_sim = float(er_seed_synonym_sim)
        # Corpus surface-IDF for the lexical gate, built lazily from the entity
        # store on first surface resolution.
        self._surface_idf: Optional[Dict[str, float]] = None
        # Passage meta is stable for the life of the agent (graph is built
        # once at ingest). Cache hash → (file_id, page_number) so chain / ppr
        # candidate-page rendering doesn't rebuild it each call.
        self._passage_meta_by_hash: Optional[Dict[str, Tuple[str, Optional[int]]]] = None


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
        self._surface_idf = None


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
            logger.warning("graph tool: NER warm-up failed: %s", exc)


    def _attach_previews(
        self,
        results: List[Dict[str, Any]],
        channel_hits: Optional[List[Any]] = None,
        *,
        question: str = "",
        window_top_k: int = _EVIDENCE_WINDOW_TOP_K,
        context: Optional["AgentContext"] = None,
    ) -> List[Dict[str, Any]]:
        """Attach per-candidate evidence, tiered by rank, deduped per run.

        The full passage text is never inlined whole — it is reachable
        only via the ``read`` tool. Each candidate instead carries a
        ``cost_tokens`` estimate (what a full ``read`` would cost) plus,
        by rank:

        * rank ``< window_top_k`` — a ``window``: the query-centered
          sentence window (``query_window``, ~3 sentences) so a small
          model can answer in place without a ``read``.
        * rank ``>= window_top_k`` — a single-sentence ``preview``
          (``query_snippet``) so deeper candidates stay discoverable
          cheaply.

        Per-run dedup: when ``context`` is supplied, a page already shown
        (its window/preview emitted earlier) or already ``read`` is not
        re-serialized — it collapses to a ``"seen": true`` stub keeping
        only its metadata + ``cost_tokens``. The dedup key is
        :func:`agentic.core.context.page_key`, identical to ``read``'s,
        so windows and full reads share one "already delivered" view.

        Channel artifact path: each page's sentences + their embeddings
        come from :meth:`GraphPPRChannel.passage_sentence_embs`, built
        once from the ingest-persisted map and the sentence_store. The
        query embedding is encoded once per call and reused across pages.

        Prefers ``channel_hits[i].evidence[0]["passage_hash_id"]`` for an
        exact passage match. Falls back to a one-time-built
        ``(file_id, page_number) → passage_hash`` map when evidence is
        unavailable (older trace formats).
        """
        from agentic.core.context import page_key
        from agentic.tools.acquisition._preview import query_snippet, query_window

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
            window_tier = i < window_top_k
            field = "window" if window_tier else "preview"
            if page_store is None or not hasattr(page_store, "hash_id_to_text"):
                out.append({**hit_meta, field: "", "cost_tokens": 0})
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
            if passage_hash is not None:
                page_text = page_store.hash_id_to_text.get(passage_hash, "") or ""

            # Cost a full ``read`` would incur (~4 chars/token over the
            # whole passage, not the capped excerpt) — always emitted so
            # the agent can weigh a window against the read it replaces.
            cost_tokens = len(page_text) // 4

            # Dedup first: a page already shown (its window/preview
            # delivered earlier this run) or already ``read`` collapses to
            # a stub. Same key as ``read`` so the two views agree.
            pkey = page_key(
                str(hit_meta["file_id"]),
                hit_meta.get("page_id"),
                hit_meta.get("page_number"),
            )
            if context is not None and (
                context.is_page_shown(pkey) or context.is_page_read(pkey)
            ):
                out.append({**hit_meta, "seen": True, "cost_tokens": cost_tokens})
                continue
            if context is not None:
                context.mark_page_as_shown(pkey)

            if window_tier:
                cached_sentences = None
                if passage_hash is not None:
                    # passage_sentence_embs is a one-time disk read + dict
                    # translation; any failure (missing ner_results.json,
                    # sentence_store schema drift) degrades to the slow-path
                    # window split, not a lost observation.
                    try:
                        cached_sentences = channel.passage_sentence_embs(passage_hash)
                    except Exception:
                        cached_sentences = None
                try:
                    window = query_window(
                        page_text,
                        question,
                        embed_client,
                        window_sentences=_EVIDENCE_WINDOW_SENTENCES,
                        max_chars=_EVIDENCE_WINDOW_MAX_CHARS,
                        cached_query_emb=q_emb,
                        cached_sentences=cached_sentences,
                    )
                except Exception:
                    window = (" ".join(page_text.split())[:_EVIDENCE_WINDOW_MAX_CHARS]
                              if page_text else "")
                out.append({**hit_meta, "window": window, "cost_tokens": cost_tokens})
                continue

            cached_sentences = None
            if passage_hash is not None:
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
            out.append({**hit_meta, "preview": preview, "cost_tokens": cost_tokens})
        return out


    def _split_candidate_pages(
        self,
        candidate_pages: List[Dict[str, Any]],
        *,
        question: str,
        context: Optional["AgentContext"] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Enrich + split bare candidate-page rows into evidence / menu.

        Reuses :meth:`_attach_previews` so the top-K rows carry a query
        ``window`` and the rest a one-line ``preview`` (each with
        ``cost_tokens``) — identical to the graph_ppr observation shape.
        A row the renderer marked ``seen`` (already shown / read this run)
        rides in ``evidence`` as a stub. ``top_entities`` comes from any
        ``clusters_touched`` already on a row (PPR enriches them; chain
        candidates carry none, so the menu shows the preview alone).
        """
        enriched = self._attach_previews(
            candidate_pages, question=question, context=context
        )
        evidence: List[Dict[str, Any]] = []
        more_candidates: List[Dict[str, Any]] = []
        for r in enriched:
            if r.get("seen"):
                evidence.append({
                    "file_id": r["file_id"],
                    "page_id": r.get("page_id"),
                    "page_number": r.get("page_number"),
                    "score": r.get("score"),
                    "seen": True,
                    "cost_tokens": r.get("cost_tokens", 0),
                })
            elif "window" in r:
                evidence.append({
                    "file_id": r["file_id"],
                    "page_id": r.get("page_id"),
                    "page_number": r.get("page_number"),
                    "score": r.get("score"),
                    "window": r["window"],
                    "cost_tokens": r.get("cost_tokens", 0),
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
                    "score": r.get("score"),
                    "preview": r.get("preview", ""),
                    "cost_tokens": r.get("cost_tokens", 0),
                    "top_entities": top_entities,
                })
        return evidence, more_candidates


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
                # Hybrid resolution (embedding recall + IDF lexical precision)
                # abstains on a confidently-wrong NN match; an abstained surface
                # carries ``accepted=False`` and is surfaced as low-confidence in
                # the audit but not seeded. The disclosed ``entity_lookup_min_sim``
                # floor stays as an additional guard on accepted entries.
                if not s.get("accepted") or s["sim"] < self._entity_lookup_min_sim:
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
        log_meta = {"focus": focus, "audit": len(entries)}
        context.add_retrieval_log(tool_name="entity_inspect", tokens=0, metadata=log_meta)
        return (
            ok(
                "GraphExploreObservation",
                focus=focus,
                entity_audit=entries,
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


    def _chain_seeds_from_surfaces(
        self,
        surfaces: List[str],
        *,
        source_tag: str,
    ) -> List[Dict[str, Any]]:
        """Resolve free-text surfaces to entity hashes by hybrid matching.

        Embedding top-10 gives recall; :func:`_hybrid_select` re-ranks by
        IDF lexical overlap (precision) with a synonym escape, so a
        confidently-wrong NN ("Warner Bros. Records" → "hollywood records")
        is rejected. A surface that ``_hybrid_select`` abstains on is still
        returned, flagged ``accepted=False`` with the bare top-NN similarity,
        so :meth:`_resolve_focus` can audit it as a low-confidence match;
        accepted entries carry ``accepted=True`` plus ``overlap`` /
        ``resolved_by``.
        """
        if not surfaces:
            return []
        channel = self._channel
        text = channel.entity_store.hash_id_to_text
        idf = self._get_surface_idf()
        out: List[Dict[str, Any]] = []
        for surf in surfaces:
            surf = (surf or "").strip()
            if not surf:
                continue
            try:
                vec = channel.embedding_client.encode(surf)
                if vec.ndim == 2:
                    vec = vec[0]
                top = channel.entity_store.topk(vec, 10)
            except Exception:
                continue
            candidates = [
                (hid, text.get(hid, ""), float(score))
                for hid, score in top
                if hid in channel._name_to_vidx
            ]
            if not candidates:
                continue
            picked = _hybrid_select(
                surf, candidates, idf,
                self._er_seed_overlap_min, self._er_seed_synonym_sim,
            )
            if picked is None:
                out.append({
                    "hash_id": candidates[0][0],
                    "sim": candidates[0][2],
                    "surface": surf,
                    "source": source_tag,
                    "accepted": False,
                })
                continue
            out.append({
                "hash_id": picked["hash_id"],
                "sim": picked["sim"],
                "surface": text.get(picked["hash_id"], surf),
                "source": source_tag,
                "overlap": picked["overlap"],
                "resolved_by": picked["resolved_by"],
                "accepted": True,
            })
        return out


    # ----------------------------------------------------------- shared helpers

    def _get_surface_idf(self) -> Dict[str, float]:
        """Corpus surface-IDF over the entity store, built once per instance.

        Each entity surface is one "document"; rare head tokens (the entity's
        distinctive name) carry weight, corpus-frequent template tokens carry
        almost none. ``tokenize_surface`` is case-sensitive, so surfaces are
        lowercased before the IDF table is built (the resolver lowercases the
        query surface too).
        """
        if self._surface_idf is None:
            self._surface_idf = build_surface_idf(
                [
                    (t or "").lower()
                    for t in self._channel.entity_store.hash_id_to_text.values()
                ]
            )
        return self._surface_idf

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


def _coerce_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
