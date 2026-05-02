"""Exhaustive regex scan over per-page Markdown.

Unit of retrieval is the page; mode is regex (use bm25_search for
literal-substring lookup with ranking). The scan is exhaustive within the
requested scope, so the result is a complete partition of the scanned
universe into ``positive_units`` and ``negative_units``. This shape is
the seed for a future ScanClaim — the proof-state layer relies on the
positive/negative sets covering the scope exactly.

Citations are first-match only per page (line number + matched text), so
output stays bounded even on very common patterns. The agent is expected
to call read_page on positive pages before quoting verbatim.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import regex as ureg

from agentic.tools.acquisition._common import (
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


# Cap citations per page so a runaway pattern (e.g. ``\d+``) cannot blow
# the tool result past the agent's per-turn context window.
_MAX_CITATIONS_PER_PAGE = 3
_MAX_TOTAL_CITATIONS = 200


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
                    "Scan page Markdown with a Python `regex` pattern. "
                    "Exhaustive over the requested scope: every in-scope "
                    "page is classified as either positive (pattern "
                    "matched at least once) or negative (no match). "
                    "Returns `positive_units`, `negative_units`, and up "
                    "to a handful of citations per positive page.\n\n"
                    "Use this when the question is 'which pages contain "
                    "X' / 'how often does X occur' / 'is there any page "
                    "that does NOT mention X'. For ranked retrieval of an "
                    "exact term, prefer bm25_search.\n\n"
                    "The pattern is matched case-insensitive, Unicode-"
                    "aware. Anchor with literal terms — bare `.+` / `.*` / "
                    "`\\d+` will match almost every page and waste budget.\n\n"
                    "Scope conventions (file_ids, page_range, section_ids) "
                    "match the other retrieval tools and AND together. "
                    "Section ids come from `toc` and look like "
                    "'<file_id>:sec_NNN'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": (
                                "Python regex (the `regex` module flavor; "
                                "Unicode-aware, supports `\\p{Han}` etc.)."
                            ),
                        },
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional file id allow-list.",
                        },
                        "page_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "Optional [start, end] inclusive 1-based page-number filter."
                            ),
                        },
                        "section_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of section ids "
                                "(e.g. '<file_id>:sec_003') from `toc`. "
                                "A page must lie inside at least one to qualify."
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
        file_ids: Optional[List[str]] = None,
        page_range: Optional[List[int]] = None,
        section_ids: Optional[List[str]] = None,
    ):
        if not pattern or not str(pattern).strip():
            return err("invalid_argument", "`pattern` must be a non-empty string."), {"error": "invalid_argument"}
        scope, scope_err = parse_scope(
            file_ids, page_range, section_ids, inventory=self.inventory
        )
        if scope_err is not None:
            return err("invalid_argument", scope_err), {"error": "invalid_argument"}

        try:
            compiled = ureg.compile(pattern, ureg.IGNORECASE)
        except Exception as exc:
            return (
                err("invalid_regex", f"Pattern failed to compile: {exc}", pattern=pattern),
                {"error": "invalid_regex"},
            )

        corpus = all_pages(self.page_store)
        in_scope = filter_pages(corpus, scope)
        if not in_scope:
            # Empty intersection (whether corpus is empty or the scope
            # filters narrowed to nothing) is still a legitimate
            # "exhaustive scan over no pages" — return ok with empty
            # unit lists so a future ScanClaim ingester sees a valid
            # (empty) partition. The agent reads scanned_count=0 and
            # widens scope on its own; surfacing this as an error
            # would tempt it to treat user-typo and empty-corpus as
            # qualitatively different.
            return (
                ok(
                    "PatternScanObservation",
                    pattern=pattern,
                    scope=scope.as_dict(),
                    exhaustive=True,
                    index_completeness="indexed_only",
                    unit_type="page",
                    scanned_count=0,
                    scanned_units=[],
                    positive_units=[],
                    negative_units=[],
                    match_counts={},
                    citations=[],
                    total_matches=0,
                    citations_truncated=False,
                ),
                {"retrieved_tokens": 0, "positive_pages": 0, "negative_pages": 0},
            )

        # Stable ordering: group by file_id, then page_number.
        in_scope.sort(key=lambda p: (p.file_id, p.page_number or 0, p.page_id))

        # The output carries two parallel views: ``scanned_units`` /
        # ``positive_units`` / ``negative_units`` are flat global-id
        # lists (the shape a future ScanClaim ingester wants — it must
        # check that ``positive ∪ negative == scanned`` to confirm
        # exhaustive coverage). ``match_counts`` and ``citations`` carry
        # the per-unit detail without diluting the unit-id sets.
        scanned_units: List[str] = []
        positive_units: List[str] = []
        negative_units: List[str] = []
        match_counts: Dict[str, int] = {}
        citations: List[Dict[str, Any]] = []
        total_matches = 0

        for page in in_scope:
            gid = page.global_id
            scanned_units.append(gid)
            text = page.text_markdown or ""
            line_citations: List[Dict[str, Any]] = []
            if text:
                for line_no, line in enumerate(text.splitlines(), start=1):
                    if len(line_citations) >= _MAX_CITATIONS_PER_PAGE:
                        break
                    m = compiled.search(line)
                    if not m:
                        continue
                    line_citations.append(
                        {
                            "global_id": gid,
                            "file_id": page.file_id,
                            "page_id": page.page_id,
                            "page_number": page.page_number,
                            "line_no": line_no,
                            "match": _truncate(m.group(0), 80),
                        }
                    )
            if line_citations:
                total_in_page = sum(1 for _ in compiled.finditer(text))
                positive_units.append(gid)
                match_counts[gid] = total_in_page
                if len(citations) < _MAX_TOTAL_CITATIONS:
                    citations.extend(line_citations[: _MAX_TOTAL_CITATIONS - len(citations)])
                total_matches += total_in_page
            else:
                negative_units.append(gid)

        log_meta = {
            "pattern": pattern,
            "scope": scope.as_dict(),
            "scanned": len(scanned_units),
            "positive": len(positive_units),
            "negative": len(negative_units),
            "total_matches": total_matches,
        }
        context.add_retrieval_log(tool_name="pattern_search", tokens=0, metadata=log_meta)

        return (
            ok(
                "PatternScanObservation",
                pattern=pattern,
                scope=scope.as_dict(),
                exhaustive=True,
                # ``exhaustive`` is over the indexed corpus snapshot,
                # not over "every file the user ever uploaded". Files
                # whose parse failed (no page_assets/<file_id>.json)
                # are NOT scanned. A future ScanClaim ingester should
                # check this flag against the inventory's known-broken
                # set before treating the partition as a soundness
                # witness.
                index_completeness="indexed_only",
                unit_type="page",
                scanned_count=len(scanned_units),
                scanned_units=scanned_units,
                positive_units=positive_units,
                negative_units=negative_units,
                match_counts=match_counts,
                citations=citations,
                total_matches=total_matches,
                citations_truncated=(
                    total_matches
                    > _MAX_TOTAL_CITATIONS
                    + sum(max(0, match_counts[gid] - _MAX_CITATIONS_PER_PAGE) for gid in positive_units)
                ),
            ),
            {
                "retrieved_tokens": 0,
                "positive_pages": len(positive_units),
                "negative_pages": len(negative_units),
            },
        )


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"
