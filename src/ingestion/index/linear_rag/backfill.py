"""Literal-substring entity-mention backfill (KAG-style "domain mount").

Why this exists
---------------

spaCy NER decides per-context whether a token span is an entity, so the
same surface (e.g. "Heritage Protector Option") can be tagged on the
page that introduces it but missed on later pages that merely refer
back to it. On a 39-page closed insurance corpus this leaves a 52% gap
between (entity, page) pairs where the surface literally appears and
the entity↔passage edges actually written into the LinearRAG graph.

This module closes that gap by treating every entity surface NER ever
produced anywhere in the corpus as an authoritative gazetteer, then
sweeping every passage with an Aho-Corasick automaton and emitting
``entity_passage`` edges for the literal misses.

Why this is safe
----------------

* The gazetteer is built from what spaCy already validated as entities.
  We are not inventing new entities — only finding additional mentions
  of entities the pipeline already accepted.
* Word-boundary checks reject "us" matching "user", "axa" matching
  "axanet", etc. The automaton is case-insensitive (target text is
  lowercased before iteration).
* The default surface filter ``min_surface_chars=4 +
  multi_word_only=True`` drops short / single-token surfaces ("us",
  "irs", "axa", "company") that would over-fire even with word
  boundaries. Both knobs are caller-overridable and meant to be
  exposed in the admin tunables panel.

References: KAG (OpenSPG) "domain knowledge mount" v0.6 — the analogous
pattern we mirror for closed corpora; HippoRAG 2 query-side fallback
for the same recall problem on questions.
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List, Mapping, Tuple

import ahocorasick
import igraph as ig

logger = logging.getLogger(__name__)


def build_gazetteer_automaton(
    entity_surfaces_by_hash: Mapping[str, str],
    *,
    min_surface_chars: int = 4,
    multi_word_only: bool = True,
) -> Tuple[ahocorasick.Automaton, int]:
    """Compile entity surfaces into a case-insensitive AC automaton.

    Returns ``(automaton, n_kept_after_filters)``. The automaton's
    payload is ``(entity_hash_id, lowercased_surface)``; surfaces are
    stored lowercased so callers must lowercase the target text before
    iterating with :func:`find_literal_matches`.
    """
    A = ahocorasick.Automaton()
    n_kept = 0
    for hash_id, surface in entity_surfaces_by_hash.items():
        s = (surface or "").strip().lower()
        if not s or len(s) < min_surface_chars:
            continue
        if multi_word_only and " " not in s:
            continue
        # Two distinct hash_ids could collapse to the same lowercased
        # surface; first wins (the existing graph already merges them
        # by alias edges anyway).
        if A.exists(s):
            continue
        A.add_word(s, (hash_id, s))
        n_kept += 1
    A.make_automaton()
    return A, n_kept


def find_literal_matches(
    text: str,
    automaton: ahocorasick.Automaton,
) -> Dict[str, int]:
    """Word-boundary substring matches → ``{entity_hash: mention_count}``.

    Word boundary = the character just outside each end is not in
    ``[a-z0-9_]``. This catches "AXA's" → "axa" while rejecting
    "user" → "us". Empty when no hits.
    """
    text_l = text.lower()
    counts: Dict[str, int] = defaultdict(int)
    for end_idx, (hash_id, surf) in automaton.iter(text_l):
        start = end_idx - len(surf) + 1
        left = text_l[start - 1] if start > 0 else " "
        right = text_l[end_idx + 1] if end_idx + 1 < len(text_l) else " "
        if left.isalnum() or left == "_":
            continue
        if right.isalnum() or right == "_":
            continue
        counts[hash_id] += 1
    return counts


def literal_backfill_graph(
    graph: ig.Graph,
    entity_surfaces_by_hash: Mapping[str, str],
    passage_text_by_hash: Mapping[str, str],
    *,
    min_surface_chars: int = 4,
    multi_word_only: bool = True,
) -> int:
    """In-place backfill: add missing entity↔passage edges from literal hits.

    Returns the number of edges added. Caller is responsible for
    persisting (``graph.write_graphml(...)``).
    """
    automaton, n_kept = build_gazetteer_automaton(
        entity_surfaces_by_hash,
        min_surface_chars=min_surface_chars,
        multi_word_only=multi_word_only,
    )
    if n_kept == 0:
        logger.info("literal_backfill: gazetteer empty after filters; skipped")
        return 0

    name_to_vidx = {v["name"]: v.index for v in graph.vs}

    # Snapshot existing entity_passage pairs so we never double-add. We
    # use frozenset(pair) because the graph is undirected — edge
    # direction is irrelevant.
    existing = set()
    if "edge_type" in graph.es.attributes():
        for e in graph.es:
            if e["edge_type"] != "entity_passage":
                continue
            a = graph.vs[e.source]["name"]
            b = graph.vs[e.target]["name"]
            existing.add(frozenset((a, b)))

    new_edge_pairs: List[Tuple[int, int]] = []
    new_edge_weights: List[float] = []
    n_skipped_existing = 0

    for phash, ptext in passage_text_by_hash.items():
        if phash not in name_to_vidx:
            continue
        pvidx = name_to_vidx[phash]
        counts = find_literal_matches(ptext or "", automaton)
        for ehash, count in counts.items():
            if ehash not in name_to_vidx:
                continue
            if frozenset((ehash, phash)) in existing:
                n_skipped_existing += 1
                continue
            evidx = name_to_vidx[ehash]
            new_edge_pairs.append((evidx, pvidx))
            new_edge_weights.append(math.log(1 + count))
            existing.add(frozenset((ehash, phash)))

    if not new_edge_pairs:
        logger.info(
            "literal_backfill: no new edges (gazetteer=%d, all literal hits already covered, skipped=%d)",
            n_kept, n_skipped_existing,
        )
        return 0

    start_eidx = graph.ecount()
    graph.add_edges(new_edge_pairs)
    for offset, weight in enumerate(new_edge_weights):
        graph.es[start_eidx + offset]["weight"] = weight
        graph.es[start_eidx + offset]["edge_type"] = "entity_passage"

    logger.info(
        "literal_backfill: gazetteer=%d added=%d edges (skipped_existing=%d)",
        n_kept, len(new_edge_pairs), n_skipped_existing,
    )
    return len(new_edge_pairs)
