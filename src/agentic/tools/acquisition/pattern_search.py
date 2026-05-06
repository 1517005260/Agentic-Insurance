"""Exhaustive regex scan over a chosen unit type.

Default unit is ``page`` — every in-scope page is classified positive
(regex matched at least once) or negative. Setting ``unit_type`` to
``passage`` or ``table_row`` runs the same scan over the corresponding
atom store so an obligation that ranges over feature-level items
(list entries, table rows) can produce a complete partition without
collapsing to page granularity.

Scope is **file_ids + section_ids only** (no page_range). The kernel's
ScopeRef recognises only those two; allowing page_range here would
let the agent produce a scan whose domain ≠ inventory.units(scope,
unit_type), causing scan_coverage_mismatch at ingest. For obligation-
aligned scans (predicate canonical id guaranteed to match), use
``proof_scan(obligation_id)``.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import regex as ureg

from agentic.tools.acquisition._common import (
    Scope,
    all_pages,
    err,
    filter_pages,
    ok,
    parse_scope,
)
from agentic.tools.base import BaseTool
from storage.inventory_store import InventoryStore
from storage.page_store import PageStore

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


logger = logging.getLogger(__name__)


_MAX_CITATIONS_PER_UNIT = 3
_MAX_TOTAL_CITATIONS = 200
_VALID_UNIT_TYPES = {"page", "passage", "table_row"}


class PatternSearchTool(BaseTool):
    def __init__(self, page_store: PageStore, inventory: Optional[InventoryStore] = None):
        self.page_store = page_store
        self.inventory = inventory

    @property
    def name(self) -> str:
        return "pattern_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "pattern_search",
                "description": (
                    "Generic regex scan over a chosen unit type — useful "
                    "for discovery (does this corpus mention X anywhere?). "
                    "Exhaustive over the requested scope; every in-scope "
                    "unit is classified positive (pattern matched ≥1 time) "
                    "or negative.\n\n"
                    "For closing set/count/forall/negation obligations, "
                    "prefer `proof_scan(obligation_id=…)` instead — that "
                    "guarantees the produced ScanClaim matches the "
                    "obligation's predicate canonical_id and scope.\n\n"
                    "Pattern is case-insensitive Unicode. Anchor with "
                    "literal terms — bare `.+` / `.*` / `\\d+` will match "
                    "almost every unit and waste budget.\n\n"
                    "Scope: `file_ids` and `section_ids` (atom-grounded, "
                    "match closure scope). Section ids come from `toc` "
                    "and look like '<file_id>:sec_NNN'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Python regex (`regex` module flavour, supports \\p{Han} etc.).",
                        },
                        "unit_type": {
                            "type": "string",
                            "enum": sorted(_VALID_UNIT_TYPES),
                            "default": "page",
                            "description": "Unit granularity for the partition.",
                        },
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional file id allow-list.",
                        },
                        "section_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of section ids "
                                "(e.g. '<file_id>:sec_003') from `toc`."
                            ),
                        },
                    },
                    "required": ["pattern"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        pattern: str,
        unit_type: str = "page",
        file_ids: Optional[List[str]] = None,
        section_ids: Optional[List[str]] = None,
    ):
        if not pattern or not str(pattern).strip():
            return err(
                "invalid_argument",
                "`pattern` must be a non-empty string.",
                remediation="Pass `pattern` as a non-empty Python regex; anchor it with literal terms.",
                valid_example={"pattern": r"AFYP\s+rebate"},
            ), {"error": "invalid_argument"}

        if unit_type not in _VALID_UNIT_TYPES:
            return err(
                "invalid_argument",
                f"`unit_type` must be one of {sorted(_VALID_UNIT_TYPES)}.",
                valid_example={"unit_type": "page"},
            ), {"error": "invalid_argument"}

        scope, scope_err = parse_scope(
            file_ids, None, section_ids, inventory=self.inventory
        )
        if scope_err is not None:
            return err(
                "invalid_argument",
                scope_err,
                remediation="Fix the scope arguments per the message.",
            ), {"error": "invalid_argument"}

        try:
            compiled = ureg.compile(pattern, ureg.IGNORECASE)
        except Exception as exc:
            return err(
                "invalid_regex",
                f"Pattern failed to compile: {exc}",
                remediation="Fix the regex syntax (Python `regex` module flavour); escape '(' ')' '\\' '.' if matching literally.",
                pattern=pattern,
            ), {"error": "invalid_regex"}

        if unit_type == "page":
            scanned, positive, negative, match_counts, citations, total_matches = self._scan_pages(
                compiled, scope,
            )
        elif unit_type == "passage":
            if self.inventory is None:
                return _no_inventory_err(unit_type)
            scanned, positive, negative, match_counts, citations, total_matches = self._scan_atoms(
                compiled, scope, atom_kind="passage",
            )
        else:
            if self.inventory is None:
                return _no_inventory_err(unit_type)
            scanned, positive, negative, match_counts, citations, total_matches = self._scan_atoms(
                compiled, scope, atom_kind="table_row",
            )

        context.add_retrieval_log(
            tool_name="pattern_search",
            tokens=0,
            metadata={
                "pattern": pattern,
                "unit_type": unit_type,
                "scope": scope.as_dict(),
                "scanned": len(scanned),
                "positive": len(positive),
                "negative": len(negative),
                "total_matches": total_matches,
            },
        )

        truncated = total_matches > _MAX_TOTAL_CITATIONS + sum(
            max(0, match_counts[u] - _MAX_CITATIONS_PER_UNIT) for u in positive
        )

        return (
            ok(
                "PatternScanObservation",
                pattern=pattern,
                scope=scope.as_dict(),
                exhaustive=True,
                index_completeness="indexed_only",
                unit_type=unit_type,
                scanned_count=len(scanned),
                scanned_units=scanned,
                positive_units=positive,
                negative_units=negative,
                match_counts=match_counts,
                citations=citations,
                total_matches=total_matches,
                citations_truncated=truncated,
            ),
            {
                "retrieved_tokens": 0,
                f"positive_{unit_type}s": len(positive),
                f"negative_{unit_type}s": len(negative),
            },
        )

    # ---------------------------------------------------------- per-unit scanners

    def _scan_pages(self, compiled, scope: Scope):
        in_scope = filter_pages(all_pages(self.page_store), scope)
        in_scope.sort(key=lambda p: (p.file_id, p.page_number or 0, p.page_id))
        scanned, positive, negative = [], [], []
        match_counts: Dict[str, int] = {}
        citations: List[Dict[str, Any]] = []
        total_matches = 0
        for page in in_scope:
            gid = page.global_id
            scanned.append(gid)
            text = page.text_markdown or ""
            line_hits = _find_line_hits(text, compiled, _MAX_CITATIONS_PER_UNIT)
            if line_hits:
                in_unit = sum(1 for _ in compiled.finditer(text))
                positive.append(gid)
                match_counts[gid] = in_unit
                total_matches += in_unit
                for hit in line_hits:
                    if len(citations) >= _MAX_TOTAL_CITATIONS:
                        break
                    citations.append(
                        {
                            "global_id": gid,
                            "file_id": page.file_id,
                            "page_id": page.page_id,
                            "page_number": page.page_number,
                            "line_no": hit["line_no"],
                            "match": hit["match"],
                        }
                    )
            else:
                negative.append(gid)
        return scanned, positive, negative, match_counts, citations, total_matches

    def _scan_atoms(self, compiled, scope: Scope, *, atom_kind: str):
        atoms = self._collect_atoms(scope, atom_kind=atom_kind)
        scanned, positive, negative = [], [], []
        match_counts: Dict[str, int] = {}
        citations: List[Dict[str, Any]] = []
        total_matches = 0
        for atom in atoms:
            uid = atom["unit_id"]
            text = atom["text"] or ""
            scanned.append(uid)
            in_unit = sum(1 for _ in compiled.finditer(text))
            if in_unit:
                positive.append(uid)
                match_counts[uid] = in_unit
                total_matches += in_unit
                first = compiled.search(text)
                if len(citations) < _MAX_TOTAL_CITATIONS and first is not None:
                    citations.append(
                        {
                            "global_id": uid,
                            "file_id": atom["file_id"],
                            "page_id": atom["page_id"],
                            "page_number": atom.get("page_number"),
                            "match": _truncate(first.group(0), 80),
                        }
                    )
            else:
                negative.append(uid)
        return scanned, positive, negative, match_counts, citations, total_matches

    def _collect_atoms(self, scope: Scope, *, atom_kind: str) -> List[Dict[str, Any]]:
        """Return atoms that fall inside ``scope``, in stable order."""

        inv = self.inventory
        store = inv.passage_store if atom_kind == "passage" else inv.table_row_store
        files = list(scope.file_ids) if scope.file_ids else _all_files(self.page_store)
        section_ranges = scope.section_ranges
        page_range = scope.page_range
        out: List[Dict[str, Any]] = []
        for fid in files:
            try:
                items = (
                    store.passages_for_file(fid)
                    if atom_kind == "passage"
                    else store.rows_for_file(fid)
                )
            except Exception as exc:                            # cache missing etc.
                logger.warning("pattern_search: %s atoms for %s unavailable: %s", atom_kind, fid, exc)
                continue
            for atom in items:
                page_num = getattr(atom, "page_number", None)
                if page_range is not None:
                    lo, hi = page_range
                    if page_num is None or page_num < lo or page_num > hi:
                        continue
                if section_ranges is not None:
                    if page_num is None:
                        continue
                    inside = any(
                        sf == fid and slo <= page_num <= shi
                        for sf, slo, shi in section_ranges
                    )
                    if not inside:
                        continue
                out.append(
                    {
                        "unit_id": getattr(
                            atom,
                            "passage_id" if atom_kind == "passage" else "table_row_id",
                        ),
                        "file_id": atom.file_id,
                        "page_id": atom.page_id,
                        "page_number": page_num,
                        "text": atom.text,
                    }
                )
        out.sort(key=lambda a: (a["file_id"], a["page_number"] or 0, a["unit_id"]))
        return out


def _find_line_hits(text: str, compiled, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not text:
        return out
    for line_no, line in enumerate(text.splitlines(), start=1):
        if len(out) >= limit:
            break
        m = compiled.search(line)
        if m is None:
            continue
        out.append({"line_no": line_no, "match": _truncate(m.group(0), 80)})
    return out


def _all_files(page_store: PageStore) -> List[str]:
    seen: List[str] = []
    visited: set[str] = set()
    for gid in page_store.ids():
        fid = gid.split("/", 1)[0]
        if fid and fid not in visited:
            visited.add(fid)
            seen.append(fid)
    return seen


def _no_inventory_err(unit_type: str):
    return err(
        "misconfigured",
        f"unit_type={unit_type!r} requires an InventoryStore; rebuild the agent with one wired in.",
    ), {"error": "misconfigured"}


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"
