"""Observation payload → :class:`NormalisedObservation` chokepoint.

Each acquisition tool emits a slightly different payload shape
(``page_global_id`` vs ``global_id``, ``text`` vs ``text_markdown`` vs
``snippet``, hits/results/candidates lists). Plant code MUST consume
the canonical view emitted by :func:`normalise`; the raw payload is
retained on :class:`NormalisedObservation.raw_payload` only for
debugging / tracer dumps.

``ScanMeta`` is the typed view of pattern_search metadata. Its
``narrowing_sources`` set explicitly records which axes (if any)
narrowed the scan below the full corpus / file / section domain. A
non-empty set disqualifies a file-level or section-level ScanClaim
sourced from this observation — no matter how much the LLM trims its
``positive_units`` / ``negative_units``, a narrowed scan can never
certify completeness over a wider universe.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from agentic.proof.types import (
    Citation,
    Observation,
    ObservationType,
)


# ----------------------------------------------------------- types


@dataclass(frozen=True)
class EntryPayload:
    """Per-page entry distilled from heterogeneous payload shapes.

    ``text`` is the longest text snapshot the observation recorded for
    this page (Markdown when available); empty string when the
    observation only knows about the page id (snippet-only search hit).
    ``fields`` is a structured-field bag — anything the tool put on the
    entry beyond its text + identity. Predicates that target structured
    fields read this (``field_equals``, ``numeric_compare``, etc.).
    """

    global_id: str
    file_id: str
    page_id: str
    text: str
    fields: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScanMeta:
    """Typed pattern_search metadata.

    Populated only for ``PAGE_HITS_EXHAUSTIVE`` / ``PAGE_HITS_PARTIAL``
    observations. Other observation types leave this None.

    ``narrowing_sources`` ⊆ {"page_range", "section_ids", "file_ids",
    "result_limit"} records which axes narrowed the scan below the
    full inventory domain. Any non-empty set disqualifies a file-level
    or section-level ScanClaim that would otherwise certify the
    full domain.
    """

    pattern: str
    flags: str
    exhaustive: bool
    scanned_units: FrozenSet[str]
    positive_units: FrozenSet[str]
    negative_units: FrozenSet[str]
    scope_file_ids: FrozenSet[str]
    page_range_used: Optional[Tuple[int, int]]
    section_ids_used: Optional[FrozenSet[str]]
    file_ids_used: Optional[FrozenSet[str]]
    result_limit_used: Optional[int]
    narrowing_sources: FrozenSet[str]


@dataclass(frozen=True)
class NormalisedObservation:
    """Canonical observation view.

    ``entries`` keys are page global_ids; values are typed
    :class:`EntryPayload`. ``referenced_units`` is the set of global
    ids the observation can ground a citation against (a superset of
    ``entries`` for hit-list-only payloads where no text is recorded).
    ``raw_payload`` is preserved for debugging / tracer output but
    must NOT be read by validators — those go through this view.
    """

    id: str
    observation_type: ObservationType
    entries: Dict[str, EntryPayload]
    referenced_units: FrozenSet[str]
    citations: List[Citation]
    scan_meta: Optional[ScanMeta] = None
    raw_payload: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------- helpers


def _entry_global_id(entry: Dict[str, Any]) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    gid = entry.get("page_global_id") or entry.get("global_id")
    if isinstance(gid, str) and gid:
        return gid
    fid = entry.get("file_id")
    pid = entry.get("page_id")
    if isinstance(fid, str) and isinstance(pid, str):
        return f"{fid}/{pid}"
    return None


def _entry_text(entry: Dict[str, Any]) -> str:
    """Pick the longest text snapshot the entry recorded. The ranking
    by length is deliberate: a per-result ``text_markdown`` (full page
    body) beats a 200-char ``snippet`` whenever both are present, so
    span-membership checks aren't tripped by truncation."""
    candidates: List[str] = []
    for tk in ("text_markdown", "text", "markdown", "body", "snippet"):
        v = entry.get(tk)
        if isinstance(v, str) and v:
            candidates.append(v)
    if not candidates:
        return ""
    return max(candidates, key=len)


def _split_global(gid: str) -> Tuple[str, str]:
    file_id, _, page_id = gid.partition("/")
    return file_id, page_id


def _build_entry(gid: str, *, text: str, raw: Optional[Dict[str, Any]] = None) -> EntryPayload:
    file_id, page_id = _split_global(gid)
    fields: Dict[str, Any] = {}
    if isinstance(raw, dict):
        # Anything not a known shape-key / identity-key goes into fields.
        # Predicates can address them via field_path lookups.
        skip_keys = {
            "page_global_id", "global_id", "file_id", "page_id",
            "text", "text_markdown", "markdown", "body", "snippet",
        }
        for k, v in raw.items():
            if k not in skip_keys:
                fields[k] = v
    return EntryPayload(
        global_id=gid, file_id=file_id, page_id=page_id,
        text=text, fields=fields,
    )


def _read_narrowing_field(payload: Dict[str, Any], scope: Dict[str, Any], key: str) -> Any:
    """pattern_search emits narrowing fields under ``payload['scope']``
    via ``Scope.as_dict()``; tests / synthetic fixtures use a top-level
    flat shape. Read both, preferring the scope-nested form (real tool
    output) and falling back to top-level."""
    if isinstance(scope, dict) and scope.get(key) is not None:
        return scope.get(key)
    return payload.get(key)


def _scan_meta_from_payload(payload: Dict[str, Any]) -> Optional[ScanMeta]:
    """Build typed scan metadata if the payload carries pattern-search
    fields; else None. Detects narrowing axes from
    ``page_range`` / ``section_ids`` / ``file_ids`` (strict subset of
    the scope's corpus) / ``result_limit`` — read either nested under
    ``payload['scope']`` (real ``pattern_search`` output) or at top
    level (synthetic test fixtures)."""
    if not isinstance(payload, dict):
        return None
    if "pattern" not in payload:
        return None
    scope = payload.get("scope") or {}
    scope_files_raw = scope.get("file_ids") if isinstance(scope, dict) else None
    page_range_raw = _read_narrowing_field(payload, scope, "page_range")
    page_range_used: Optional[Tuple[int, int]] = None
    if (
        isinstance(page_range_raw, (list, tuple))
        and len(page_range_raw) == 2
        and all(isinstance(p, int) for p in page_range_raw)
    ):
        page_range_used = (int(page_range_raw[0]), int(page_range_raw[1]))
    section_ids_raw = _read_narrowing_field(payload, scope, "section_ids")
    section_ids_used = (
        frozenset(str(s) for s in section_ids_raw)
        if isinstance(section_ids_raw, (list, tuple)) and section_ids_raw
        else None
    )
    # file_ids: top-level or scope-nested. ``scope_files_raw`` is the
    # scope's file_ids; the narrowing axis is "the LLM passed an
    # explicit file_ids filter", regardless of whether it equals the
    # scope's universe — pattern_search always populates scope, so the
    # presence of an explicit file_ids hint at top-level is rare. We
    # still detect the strict-subset case via top-level ``file_ids``
    # for synthetic fixtures.
    file_ids_raw_top = payload.get("file_ids")
    file_ids_used = (
        frozenset(str(f) for f in file_ids_raw_top)
        if isinstance(file_ids_raw_top, (list, tuple)) and file_ids_raw_top
        else None
    )
    result_limit_used: Optional[int] = None
    for key in ("result_limit", "max_results", "top_k", "limit"):
        v = _read_narrowing_field(payload, scope, key)
        if isinstance(v, int) and v > 0:
            result_limit_used = v
            break
    narrowing: set[str] = set()
    if page_range_used is not None:
        narrowing.add("page_range")
    if section_ids_used is not None:
        narrowing.add("section_ids")
    if file_ids_used is not None and isinstance(scope_files_raw, (list, tuple)):
        scope_set = frozenset(str(f) for f in scope_files_raw)
        if scope_set and file_ids_used != scope_set and file_ids_used.issubset(scope_set):
            narrowing.add("file_ids")
    if result_limit_used is not None:
        narrowing.add("result_limit")
    return ScanMeta(
        pattern=str(payload.get("pattern") or ""),
        flags=str(payload.get("flags") or ""),
        exhaustive=bool(payload.get("exhaustive", False)),
        scanned_units=frozenset(payload.get("scanned_units") or payload.get("scanned_pages") or []),
        positive_units=frozenset(payload.get("positive_units") or []),
        negative_units=frozenset(payload.get("negative_units") or []),
        scope_file_ids=frozenset(scope_files_raw or []),
        page_range_used=page_range_used,
        section_ids_used=section_ids_used,
        file_ids_used=file_ids_used,
        result_limit_used=result_limit_used,
        narrowing_sources=frozenset(narrowing),
    )


# ----------------------------------------------------------- public API


def normalise(observation: Observation) -> NormalisedObservation:
    """Convert an :class:`Observation` into the canonical view.

    Idempotent and side-effect-free: callers that need to re-derive the
    view get the same result. ``observation.payload`` is left
    untouched.
    """
    payload = observation.payload or {}
    entries: Dict[str, EntryPayload] = {}
    referenced: set[str] = set()

    # Single-page payload (``page_global_id`` + ``text``) — read_page
    # is the canonical example.
    pl_gid = payload.get("page_global_id") or payload.get("global_id")
    if isinstance(pl_gid, str) and pl_gid:
        text = ""
        for tk in ("text_markdown", "text", "markdown", "body"):
            v = payload.get(tk)
            if isinstance(v, str) and v:
                text = v
                break
        entries[pl_gid] = _build_entry(pl_gid, text=text, raw=payload)
        referenced.add(pl_gid)

    # Multi-entry shapes: results / hits / candidate_pages / citations.
    for key in ("results", "hits", "candidate_pages"):
        for raw_entry in payload.get(key) or []:
            gid = _entry_global_id(raw_entry)
            if not gid:
                continue
            text = _entry_text(raw_entry) if isinstance(raw_entry, dict) else ""
            existing = entries.get(gid)
            if existing is None or len(text) > len(existing.text):
                entries[gid] = _build_entry(
                    gid, text=text,
                    raw=raw_entry if isinstance(raw_entry, dict) else None,
                )
            referenced.add(gid)

    # Pattern-search lists every page in scanned_units — those are
    # *referenced* (the LLM may cite them) but typically have no text
    # snapshot on the entry itself.
    for key in ("scanned_units", "positive_units", "negative_units",
                "scanned_pages", "page_global_ids"):
        for gid in payload.get(key) or []:
            if isinstance(gid, str) and gid:
                referenced.add(gid)
                if gid not in entries:
                    entries[gid] = _build_entry(gid, text="", raw=None)

    # Observation citations: their (file_id, page_id) is also referenced.
    for cite in observation.citations or []:
        gid = f"{cite.file_id}/{cite.page_id}"
        referenced.add(gid)
        if gid not in entries:
            entries[gid] = _build_entry(gid, text=cite.span or "", raw=None)

    return NormalisedObservation(
        id=observation.id,
        observation_type=observation.observation_type,
        entries=entries,
        referenced_units=frozenset(referenced),
        citations=list(observation.citations or []),
        scan_meta=_scan_meta_from_payload(payload),
        raw_payload=payload,
    )


def citation_in(citation: Citation, normalised: NormalisedObservation, page_store) -> bool:
    """Validate a citation against ``normalised``.

    Two-stage:
    1. The cited (file_id, page_id) must appear in ``referenced_units``.
    2. If a span is supplied, it must appear in *some* recorded text
       snapshot for that page (entry.text or an observation.citations
       span). A whole-page witness (no span) only requires the page to
       resolve via the live ``page_store`` so an unmaterialised page id
       cannot pose as a witness.
    """
    gid = f"{citation.file_id}/{citation.page_id}"
    if gid not in normalised.referenced_units:
        return False
    if citation.span is None:
        return page_store.get(gid) is not None
    for body in snapshots_for(citation, normalised):
        if body and citation.span in body:
            return True
    return False


def snapshots_for(citation: Citation, normalised: NormalisedObservation) -> List[str]:
    """Every text snapshot recorded for ``(file_id, page_id)``."""
    gid = f"{citation.file_id}/{citation.page_id}"
    out: List[str] = []
    entry = normalised.entries.get(gid)
    if entry is not None and entry.text:
        out.append(entry.text)
    for cite in normalised.citations:
        if (cite.file_id, cite.page_id) == (citation.file_id, citation.page_id):
            if cite.span:
                out.append(cite.span)
    return out


def fetch_span_text(citation: Citation, normalised: NormalisedObservation) -> str:
    if citation.span is not None:
        return citation.span
    for body in snapshots_for(citation, normalised):
        if body:
            return body
    return ""
