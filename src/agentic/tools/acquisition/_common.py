"""Shared helpers for the acquisition tool family.

These utilities exist so each tool stays small and focused on its own
backend. The rules they encode (scope semantics, error envelope, snippet
shape) are agreed across all retrieval tools and *must not* drift
per-tool — the agent reads many tool results in a single trajectory and
inconsistent shapes burn token budget on schema friction.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from storage.inventory_store import InventoryStore, Section
from storage.page_store import PageAsset, PageStore


# ---------------------------------------------------------------- envelopes


def ok(observation_type: str, **fields: Any) -> str:
    """Serialize a successful observation. Every retrieval tool returns
    this shape, with `observation_type` matching docs/engineering.md §5.

    JSON is compact (no indent) so the agent's context window is not
    spent on whitespace — readability is achieved through stable field
    naming, not pretty-printing.
    """
    payload = {"observation_type": observation_type, "ok": True}
    payload.update(fields)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def err(
    code: str,
    message: str,
    *,
    remediation: Optional[str] = None,
    valid_example: Optional[Any] = None,
    affected_fields: Optional[list] = None,
    **context: Any,
) -> str:
    """Serialize a structured error envelope for the LLM.

    Every error to the LLM should answer three questions: what failed
    (``code``), why (``message``), and what to do next (``remediation``
    and optionally ``valid_example``). When multiple fields fail at
    once (typically pydantic validation), ``affected_fields`` lists
    every distinct top-level field with errors so the LLM repairs all
    of them in one turn instead of looping on the first.
    """
    error: Dict[str, Any] = {"code": code, "message": message}
    if remediation is not None:
        error["remediation"] = remediation
    if valid_example is not None:
        error["valid_example"] = valid_example
    if affected_fields:
        error["affected_fields"] = list(affected_fields)
    if context:
        error["context"] = context
    payload: Dict[str, Any] = {
        "observation_type": "ToolError",
        "ok": False,
        "error": error,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------- scope


@dataclass(frozen=True)
class Scope:
    """Resolved scope across three orthogonal filters.

    All three filters compose as **AND** — a page passes the scope only
    if it satisfies every non-empty filter.

    * ``file_ids``       — restrict to a set of file ids.
    * ``page_range``     — global ``[start, end]`` 1-based page-number gate.
    * ``section_ranges`` — page must lie inside *at least one* of these
      ``(file_id, page_start, page_end)`` triples (resolved from
      ``section_ids``). Multiple sections combine as a UNION.
    """

    file_ids: Optional[frozenset[str]]
    page_range: Optional[Tuple[int, int]]
    section_ranges: Optional[Tuple[Tuple[str, int, int], ...]]
    section_ids: Optional[Tuple[str, ...]]  # for echoing back in `as_dict`

    def contains(self, file_id: str, page_number: Optional[int]) -> bool:
        if self.file_ids is not None and file_id not in self.file_ids:
            return False
        if self.page_range is not None:
            if page_number is None:
                return False
            lo, hi = self.page_range
            if page_number < lo or page_number > hi:
                return False
        if self.section_ranges is not None:
            if page_number is None:
                return False
            for sf, slo, shi in self.section_ranges:
                if file_id == sf and slo <= page_number <= shi:
                    break
            else:
                return False
        return True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "file_ids": sorted(self.file_ids) if self.file_ids else None,
            "page_range": list(self.page_range) if self.page_range else None,
            "section_ids": list(self.section_ids) if self.section_ids else None,
        }


def parse_scope(
    file_ids: Optional[Sequence[str]],
    page_range: Optional[Sequence[int]],
    section_ids: Optional[Sequence[str]] = None,
    *,
    inventory: Optional[InventoryStore] = None,
) -> Tuple[Optional[Scope], Optional[str]]:
    """Validate scope arguments. Returns ``(scope, error_message)``.

    Empty list ``file_ids=[]`` / ``section_ids=[]`` is treated as
    ``None``: the agent passing an empty filter is almost certainly a
    mistake, and silently returning zero results would just confuse the
    next turn.

    ``section_ids`` resolution requires an :class:`InventoryStore`; the
    function returns ``"unknown_section"`` when an id can't be found
    (typo, stale id from before re-ingest) and ``"misconfigured"`` when
    section_ids are supplied without an inventory.
    """
    fids: Optional[frozenset[str]] = None
    if file_ids:
        cleaned = [str(f).strip() for f in file_ids if str(f).strip()]
        if cleaned:
            fids = frozenset(cleaned)

    rng: Optional[Tuple[int, int]] = None
    if page_range is not None:
        if (
            not isinstance(page_range, (list, tuple))
            or len(page_range) != 2
        ):
            return None, "page_range must be a 2-element list [start, end]."
        try:
            lo, hi = int(page_range[0]), int(page_range[1])
        except (TypeError, ValueError):
            return None, "page_range entries must be integers."
        if lo < 1 or hi < lo:
            return None, f"page_range must satisfy 1 <= start <= end, got [{lo}, {hi}]."
        rng = (lo, hi)

    sec_tuples: Optional[Tuple[Tuple[str, int, int], ...]] = None
    sec_ids_norm: Optional[Tuple[str, ...]] = None
    if section_ids:
        cleaned_ids = [str(s).strip() for s in section_ids if str(s).strip()]
        if cleaned_ids:
            if inventory is None:
                return None, "section_ids supplied but no InventoryStore is wired into this tool."
            resolved: List[Tuple[str, int, int]] = []
            unknown: List[str] = []
            for sid in cleaned_ids:
                section = inventory.get(sid)
                if section is None:
                    unknown.append(sid)
                    continue
                resolved.append((section.file_id, section.page_start, section.page_end))
            if unknown:
                return None, f"unknown section_ids: {unknown}. Call toc to refresh."
            sec_tuples = tuple(resolved)
            sec_ids_norm = tuple(cleaned_ids)

    return (
        Scope(
            file_ids=fids,
            page_range=rng,
            section_ranges=sec_tuples,
            section_ids=sec_ids_norm,
        ),
        None,
    )


def filter_pages(pages: Iterable[PageAsset], scope: Scope) -> List[PageAsset]:
    return [p for p in pages if scope.contains(p.file_id, p.page_number)]


def all_pages(store: PageStore) -> List[PageAsset]:
    """Iterate every page asset in stable global-id order."""
    out: List[PageAsset] = []
    for gid in store.ids():
        page = store.get(gid)
        if page is not None:
            out.append(page)
    return out


# ---------------------------------------------------------------- snippets


def make_snippet(text: str, max_chars: int = 240) -> str:
    """First non-empty line, trimmed and tail-elided.

    Acquisition tools should emit *abbreviated* snippets only — the agent
    is told to call read_page before citing — so a single representative
    line per hit is enough to triage candidates without flooding context.
    """
    if not text:
        return ""
    for line in text.splitlines():
        s = line.strip()
        if s:
            return (s[: max_chars - 1] + "…") if len(s) > max_chars else s
    return ""


def keyword_snippet(text: str, needles: Sequence[str], max_chars: int = 240) -> str:
    """First line that contains any of the given needles (case-insensitive)."""
    if not text or not needles:
        return make_snippet(text, max_chars)
    lowered = [n.lower() for n in needles if n]
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        sl = s.lower()
        if any(n in sl for n in lowered):
            return (s[: max_chars - 1] + "…") if len(s) > max_chars else s
    return make_snippet(text, max_chars)


# ---------------------------------------------------------------- io guards


def safe_resolve_path(base: Path, candidate: str) -> Optional[Path]:
    """Resolve ``candidate`` under ``base`` and reject path-traversal.

    Returns ``None`` if the resolved path escapes ``base`` or doesn't
    exist. Used by VLM image lookup to keep the page_image_path field
    from sneaking us out of STORAGE_PATH.
    """
    try:
        resolved = (base / candidate).resolve()
        base_resolved = base.resolve()
    except (OSError, RuntimeError):
        return None
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        return None
    return resolved if resolved.exists() else None
