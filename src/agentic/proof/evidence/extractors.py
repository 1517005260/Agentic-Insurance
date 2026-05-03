"""Auto-extractors keyed by ObservationType.

Auto-extraction is the bridge from acquisition tool outputs to claim
candidates the plant can ingest. The registry is small on purpose —
only ObservationTypes whose semantics we trust to map mechanically to
typed claims have an extractor; everything else stays an Observation
that the LLM may choose to convert via ``evidence_ingest``.

Currently registered:

* ``PAGE_HITS_EXHAUSTIVE`` — pattern_search with exhaustive=True.
  Aggregates page-level hits up to file or section level (when
  section confidence + page-exclusivity allow). Produces a ScanClaim
  whose positive/negative units cover the full inventory domain.

The registry is keyed by ObservationType (not tool name). New tools
that want auto-extraction declare a new ObservationType and register
an extractor here.
"""
from typing import Any, Callable, Dict, List, Optional

from agentic.proof.types import (
    Citation,
    Claim,
    ClaimType,
    Observation,
    ObservationType,
    PredicateSpec,
    ScopeRef,
)
from agentic.proof.predicate import build_spec
from storage.inventory_store import InventoryStore


# fn signature: (observation, inventory) -> List[Claim] (without ids set)
ExtractorFn = Callable[[Observation, InventoryStore], List[Claim]]


def _is_literal_pattern(pattern: str) -> bool:
    """A pattern is literal iff it carries no regex metacharacters and
    therefore is equivalent to a plain substring search.

    When the agent declares a ``contains_string`` obligation (the most
    common shape) the auto-extractor must emit a ``contains_string``
    claim, not ``regex_match`` — entailment is syntactic only, so the
    two would otherwise fail to bind even though they describe the
    same query semantically.

    Implementation: ``re.escape(pattern) == pattern`` iff the pattern
    contains no characters the regex compiler would treat specially.
    This delegates the ground truth to the standard library so newly
    introduced metacharacters (any future Python regex extension) are
    picked up automatically.
    """
    import re
    return bool(pattern) and re.escape(pattern) == pattern


def _build_predicate_from_payload(payload: Dict[str, Any]) -> Optional[PredicateSpec]:
    """Translate pattern_search's recorded pattern into the predicate
    spec that will entail-match the obligation it should close.

    Literal patterns (no regex metacharacters) become ``contains_string``;
    proper regex patterns become ``regex_match``. ``flags`` defaults to
    ``"i"`` because pattern_search compiles its regex with
    ``regex.IGNORECASE`` unconditionally — see
    src/agentic/tools/acquisition/pattern_search.py.

    Returns ``None`` when the payload carries no pattern or the
    primitive registration fails (e.g., universal pattern blacklist).
    """
    pattern = payload.get("pattern")
    if not pattern:
        return None
    flags = str(payload.get("flags") or "i")
    case_sensitive = "i" not in flags
    pattern_str = str(pattern)
    if _is_literal_pattern(pattern_str):
        try:
            return build_spec(
                "contains_string",
                {"pattern": pattern_str, "case_sensitive": case_sensitive},
            )
        except Exception:
            pass
    try:
        return build_spec("regex_match", {"pattern": pattern_str, "flags": flags})
    except Exception:
        return None


def _aggregate_page_hits_to_units(
    positive_pages: List[str],
    negative_pages: List[str],
    scanned_pages: List[str],
    inventory: InventoryStore,
    target_unit_type: str,
) -> Optional[Dict[str, Any]]:
    """Map page-level positive / negative sets to file or section level.

    ``positive_pages`` and ``negative_pages`` come straight from
    pattern_search's exhaustive partition (page global ids of the form
    ``<file_id>/<page_id>``). Returns
    ``{"positive_units": [...], "negative_units": [...], "scope": ScopeRef}``
    or ``None`` when section-level aggregation is unsafe (low confidence,
    non-exclusive sections, mixed scope).
    """
    if not scanned_pages:
        return None
    files: set[str] = set()
    pages_by_file: Dict[str, set[str]] = {}
    for pgid in scanned_pages:
        fid, sep, _ = pgid.partition("/")
        if not sep:
            continue
        files.add(fid)
        pages_by_file.setdefault(fid, set()).add(pgid)

    hit_pages = set(positive_pages)

    if target_unit_type == "file":
        positive = sorted({
            fid for fid in files
            if any(p in hit_pages for p in pages_by_file.get(fid, set()))
        })
        negative = sorted(files - set(positive))
        scope = ScopeRef(
            file_ids=frozenset(files),
            section_ids=None,
            sealed=True,        # exhaustive scan over all files in scope
        )
        return {"positive_units": positive, "negative_units": negative, "scope": scope}

    if target_unit_type == "section":
        # Aggregate only when the file's sections satisfy the
        # confidence + exclusivity guard. Otherwise we'd overclaim a
        # boundary the inventory can't certify.
        section_ids: List[str] = []
        for fid in files:
            for sec in inventory.sections_for_file(fid):
                section_ids.append(sec.section_id)
        if not section_ids:
            return None
        for sid in section_ids:
            sec = inventory.get(sid)
            if sec is None:
                return None
            if sec.confidence == "low" or not sec.is_page_exclusive:
                return None
        # Bucket pages → sections.
        positive: set[str] = set()
        negative: set[str] = set()
        section_pages: Dict[str, List[str]] = {sid: [] for sid in section_ids}
        for fid, pages in pages_by_file.items():
            for pgid in pages:
                sid = inventory.section_for_page(pgid)
                if sid is None or sid not in section_pages:
                    # A page outside any section disqualifies section
                    # aggregation — we'd have a coverage hole.
                    return None
                section_pages[sid].append(pgid)
        for sid, pgs in section_pages.items():
            if not pgs:
                # Section had zero pages scanned; ScanClaim cannot
                # certify completeness without it.
                return None
            if any(p in hit_pages for p in pgs):
                positive.add(sid)
            else:
                negative.add(sid)
        scope = ScopeRef(
            file_ids=frozenset(files),
            section_ids=frozenset(section_ids),
            sealed=True,
        )
        return {
            "positive_units": sorted(positive),
            "negative_units": sorted(negative),
            "scope": scope,
        }
    return None


def _extract_page_hits_exhaustive(
    observation: Observation,
    inventory: InventoryStore,
) -> List[Claim]:
    """Emit ScanClaims from a fully-scoped exhaustive pattern_search.

    Two soundness gates beyond the basic payload check:

    * If the pattern_search call narrowed scope (page_range or section_ids),
      we cannot emit a file-level ScanClaim — the partition does not
      cover the whole file. Section-level may still be safe iff every
      page of each candidate section was scanned.
    * Even when scope is "whole file", section-level only fires when
      every page of every section in the file actually appeared in
      ``scanned_units`` — partial scans must not certify section
      coverage.
    """
    payload = observation.payload or {}
    if not payload.get("exhaustive"):
        return []
    positive_pages = list(payload.get("positive_units") or [])
    negative_pages = list(payload.get("negative_units") or [])
    scanned_pages = list(payload.get("scanned_units") or [])
    if not scanned_pages:
        return []
    scanned_set = set(scanned_pages)
    scope = payload.get("scope") or {}
    scope_narrowed = bool(
        scope.get("page_range") is not None or scope.get("section_ids")
    )

    predicate = _build_predicate_from_payload(payload)
    if predicate is None:
        return []

    out: List[Claim] = []

    files_in_scan: set[str] = set()
    for gid in scanned_pages:
        fid, sep, _ = gid.partition("/")
        if sep:
            files_in_scan.add(fid)

    # File-level claim: requires the scan to have covered every indexed
    # page of every file under consideration. Narrowed scope disqualifies
    # this path outright.
    if not scope_narrowed:
        full_files = []
        for fid in files_in_scan:
            indexed_pages = {
                gid for gid in inventory.page_store.ids() if gid.startswith(fid + "/")
            }
            if not indexed_pages:
                continue
            if indexed_pages.issubset(scanned_set):
                full_files.append(fid)
        if len(full_files) == len(files_in_scan):
            file_agg = _aggregate_page_hits_to_units(
                positive_pages, negative_pages, scanned_pages, inventory, "file"
            )
            if file_agg is not None:
                out.append(_make_scan_claim(observation, predicate, "file", file_agg))

    # Section-level claim: every section in the file's universe must
    # have all its pages scanned. The aggregator already enforces this
    # by returning None if any section is short-scanned, but we add a
    # belt-and-suspenders pre-check so the failure is surfaced clearly.
    section_universe_ok = True
    for fid in files_in_scan:
        for sec in inventory.sections_for_file(fid):
            for page_no in range(sec.page_start, sec.page_end + 1):
                gid = f"{fid}/p_{page_no:04d}"
                if gid not in scanned_set:
                    section_universe_ok = False
                    break
            if not section_universe_ok:
                break
        if not section_universe_ok:
            break
    if section_universe_ok:
        section_agg = _aggregate_page_hits_to_units(
            positive_pages, negative_pages, scanned_pages, inventory, "section"
        )
        if section_agg is not None:
            out.append(_make_scan_claim(observation, predicate, "section", section_agg))
    return out


def _make_scan_claim(
    observation: Observation,
    predicate: PredicateSpec,
    unit_type: str,
    agg: Dict[str, Any],
) -> Claim:
    citations = list(observation.citations) if observation.citations else []
    return Claim(
        id="",
        observation_id=observation.id,
        claim_type=ClaimType.SCAN,
        scope=agg["scope"],
        unit_type=unit_type,                 # type: ignore[arg-type]
        predicate=predicate,
        score=None,
        positive_units=list(agg["positive_units"]),
        negative_units=list(agg["negative_units"]),
        value_map={},
        citations=citations,
        derivation="auto_extract",
    )


_REGISTRY: Dict[ObservationType, ExtractorFn] = {
    ObservationType.PAGE_HITS_EXHAUSTIVE: _extract_page_hits_exhaustive,
}


def auto_extract(
    observation: Observation,
    inventory: InventoryStore,
) -> List[Claim]:
    """Run the registered extractor for ``observation.observation_type``.

    Returns ``[]`` for ObservationTypes without an extractor or when
    the extractor declines to emit a claim (e.g., section guard fails).
    The plant treats an empty list as "no auto-claim — LLM must
    propose if it wants this observation as evidence".
    """
    fn = _REGISTRY.get(observation.observation_type)
    if fn is None:
        return []
    return fn(observation, inventory)
