"""Literal-substring entity-mention gazetteer (KAG-style "domain mount").

Why this exists
---------------

NER decides per-context whether a token span is an entity, so the
same surface (e.g. "Heritage Protector Option") can be tagged on the
page that introduces it but missed on later pages that merely refer
back to it. The query-time PPR channel (``GraphPPRChannel``) closes
that recall gap by compiling every entity surface NER ever produced
into an Aho-Corasick gazetteer and sweeping the question's candidate
passages for literal misses.

This module provides the gazetteer primitives the channel uses:
:func:`build_gazetteer_automaton` and :func:`find_literal_matches`
(plus :func:`_passage_canon`, a per-passage canonical-text memo).

Why this is safe
----------------

* The gazetteer is built from what NER already validated as entities.
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

from collections import defaultdict
from typing import Dict, Mapping, Tuple

import ahocorasick
import regex

from ingestion.index.linear_rag.normalize import canonical_form


_HAS_HAN_RE = regex.compile(r"\p{Han}")

# Per-passage canonical-text cache. ``canonical_form`` (NFKC + OpenCC
# t2s + lower) is invariant for a given passage hash, but
# ``literal_backfill_graph`` re-canonicalises every passage in the
# corpus on every ingest → O(N²) wall time (the data-confirmed
# literal-backfill cost). Passage hashes are content-addressed, so a
# process-lifetime memo keyed by (passage_hash, fold_traditional) is
# safe and collapses the per-doc recompute to one-per-passage-ever.
_PASSAGE_CANON_CACHE: Dict[Tuple[str, bool], str] = {}


def _passage_canon(phash: str, ptext: str, fold_traditional: bool) -> str:
    """Cached canonical passage text — byte-identical to what
    :func:`find_literal_matches` computes internally for the same args."""
    key = (phash, fold_traditional)
    v = _PASSAGE_CANON_CACHE.get(key)
    if v is None:
        v = (
            canonical_form(ptext, fold_traditional=fold_traditional)
            if fold_traditional
            else (ptext or "").lower()
        )
        _PASSAGE_CANON_CACHE[key] = v
    return v


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

    ``multi_word_only`` is interpreted **per script**: it requires an
    ASCII space for surfaces that look Latin-script (e.g. "axa" /
    "company" — drop, "Heritage Protector Option" — keep), but is
    waived for CJK surfaces ("萬通保險" — keep) since Chinese has no
    word boundary character. Without this carve-out the gazetteer
    comes out empty on Chinese-heavy corpora and silently produces
    zero backfill edges.
    """
    A = ahocorasick.Automaton()
    n_kept = 0
    for hash_id, surface in entity_surfaces_by_hash.items():
        s = (surface or "").strip().lower()
        if not s or len(s) < min_surface_chars:
            continue
        if multi_word_only and " " not in s and not _HAS_HAN_RE.search(s):
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
    *,
    fold_traditional: bool = True,
    precanonicalized: bool = False,
) -> Dict[str, int]:
    """Word-boundary substring matches → ``{entity_hash: mention_count}``.

    Word boundary = the character just outside each end is not in
    ``[a-z0-9_]``. This catches "AXA's" → "axa" while rejecting
    "user" → "us". For CJK we deliberately do NOT require a boundary —
    Chinese tokens are already character-level dense, and a "boundary"
    requirement would make every entity inside a longer noun phrase
    miss (e.g. "復歸紅利" inside "非保證復歸紅利"). Empty when no hits.

    ``fold_traditional=True`` runs the target text through the same
    ``canonical_form`` (lowercase + NFKC + OpenCC t2s) the entity
    surfaces went through at ingest, so a Traditional-Chinese passage
    looks the same to the automaton as the Simplified entity surface
    it should match.

    ``precanonicalized=True`` means ``text`` is already the exact
    transform this function would otherwise apply — the caller has
    cached it (passage canonical text is invariant once ingested, so
    recomputing OpenCC/NFKC for every passage on every ingest is the
    O(N²) literal-backfill cost). Byte-identical to the default path.
    """
    if precanonicalized:
        text_l = text
    else:
        text_l = canonical_form(text, fold_traditional=fold_traditional) if fold_traditional else text.lower()
    counts: Dict[str, int] = defaultdict(int)
    for end_idx, (hash_id, surf) in automaton.iter(text_l):
        start = end_idx - len(surf) + 1
        # Latin / digit boundary check applies only when both edges of
        # the matched span are in the ASCII alnum class — the case the
        # original "us"/"user" carve-out targeted. CJK matches skip the
        # check (no script-level boundary character exists).
        left = text_l[start - 1] if start > 0 else " "
        right = text_l[end_idx + 1] if end_idx + 1 < len(text_l) else " "
        surf_is_cjk = bool(_HAS_HAN_RE.search(surf))
        if not surf_is_cjk:
            if left.isalnum() or left == "_":
                continue
            if right.isalnum() or right == "_":
                continue
        counts[hash_id] += 1
    return counts
