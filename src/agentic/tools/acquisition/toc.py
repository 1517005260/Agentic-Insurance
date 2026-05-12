"""Section outline of one file.

The actual heading-extraction + span-resolution lives in
:class:`storage.InventoryStore` so the section IDs the agent sees here
are the same ids it can later pass back through any retrieval tool's
``section_ids`` argument. This tool is a thin presentation layer:

* depth-cap filter (``max_depth`` defaults to 3)
* "no headings" / "file not found" diagnostics
* compact JSON envelope shared with every other tool

The first call for a given file populates
``STORAGE_PATH/inventory/<file_id>.json`` so subsequent calls (and any
``section_ids`` resolution from other tools) hit a hot cache.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agentic.tools.acquisition._common import err, normalize_file_id, ok
from agentic.tools.base import BaseTool
from storage.inventory_store import InventoryStore
from storage.page_store import PageStore

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


_DEFAULT_MAX_DEPTH = 3


class TocTool(BaseTool):
    def __init__(
        self,
        page_store: PageStore,
        inventory: Optional[InventoryStore] = None,
    ):
        self.page_store = page_store
        self.inventory = inventory or InventoryStore(page_store=page_store)

    @property
    def name(self) -> str:
        return "toc"

    def warm_up(self) -> None:
        """Pre-build per-file section inventory.

        Doing this at startup keeps the first ``section_ids`` reference
        from any retrieval tool fast and persists ``inventory/<file_id>.json``
        files so future processes skip the rebuild.
        """
        try:
            self.inventory.warm_up()
        except Exception:
            # Inventory build is best-effort; missing markdown headings
            # or unreadable page assets shouldn't break the agent loop.
            pass

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "toc",
                "description": (
                    "Return the section outline of one file, derived from "
                    "the page Markdown's heading structure (`#`, `##`, "
                    "...). Each section reports `section_id` (a stable id "
                    "of the form '<file_id>:sec_NNN'), `title`, `depth`, "
                    "`page_start`, `page_end`, and `parent_section_id`. "
                    "Sections are listed in document order; nesting is "
                    "implied by `depth` and confirmed by `parent_section_id`.\n\n"
                    "Use this before searching inside a long file so you "
                    "can pass either a tight `page_range` OR the section "
                    "ids you care about as `section_ids` to retrieval tools.\n\n"
                    "If the file has no Markdown headings (e.g. scanned-"
                    "only PDFs) the result is empty — fall back to "
                    "list_files / read_page in that case."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "File id to outline (from list_files).",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": (
                                "Drop headings deeper than this. Default 3 "
                                "(i.e. `#`, `##`, `###` are kept)."
                            ),
                        },
                    },
                    "required": ["file_id"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        file_id: str,
        max_depth: int = _DEFAULT_MAX_DEPTH,
    ):
        if not file_id or not str(file_id).strip():
            return err(
                "invalid_argument",
                "`file_id` is required.",
                remediation="Call list_files first to discover valid file_ids in this corpus, then pass one as `file_id`.",
                valid_example={"file_id": "<file_id>"},
            ), {"error": "invalid_argument"}
        try:
            depth_cap = int(max_depth)
        except (TypeError, ValueError):
            return err(
                "invalid_argument",
                "`max_depth` must be an integer.",
                remediation="Pass `max_depth` as an integer in [1, 6] (default 3), or omit the field.",
                valid_example={"max_depth": 3},
            ), {"error": "invalid_argument"}
        if not 1 <= depth_cap <= 6:
            return (
                err(
                    "invalid_argument",
                    "`max_depth` must be between 1 and 6.",
                    remediation="Set `max_depth` to a value in [1, 6] (default 3 keeps #, ##, ### headings).",
                    valid_example={"max_depth": 3},
                ),
                {"error": "invalid_argument"},
            )

        # ``list_files`` returns filename = "<file_id>.pdf" alongside the
        # canonical file_id; the LLM routinely passes the filename here.
        # Normalize before the page-index check so a single ".pdf" slip
        # doesn't surface as file_not_found.
        file_id_s = normalize_file_id(file_id)
        # Reach into PageStore once just to confirm the file is indexed —
        # an empty inventory could mean either "no headings" or "no
        # such file"; we want a distinct error for the latter.
        has_pages = any(gid.startswith(file_id_s + "/") for gid in self.page_store.ids())
        if not has_pages:
            return (
                err(
                    "file_not_found",
                    f"No pages indexed for file_id={file_id_s!r}.",
                    remediation="Call list_files to enumerate ingested file_ids in this corpus, then pass one of those ids as `file_id`. The id you supplied is unknown or its parse never completed.",
                    file_id=file_id_s,
                ),
                {"error": "file_not_found"},
            )

        sections = [
            {
                "section_id": s.section_id,
                "title": s.title,
                "depth": s.depth,
                "page_start": s.page_start,
                "page_end": s.page_end,
                "parent_section_id": s.parent_section_id,
            }
            for s in self.inventory.sections_for_file(file_id_s)
            if s.depth <= depth_cap
        ]
        page_count = sum(1 for gid in self.page_store.ids() if gid.startswith(file_id_s + "/"))

        log_meta = {"file_id": file_id_s, "sections_found": len(sections)}
        context.add_retrieval_log(tool_name="toc", tokens=0, metadata=log_meta)

        return (
            ok(
                "TocObservation",
                file_id=file_id_s,
                max_depth=depth_cap,
                sections=sections,
                page_count=page_count,
            ),
            {"retrieved_tokens": 0, "sections_found": len(sections)},
        )
