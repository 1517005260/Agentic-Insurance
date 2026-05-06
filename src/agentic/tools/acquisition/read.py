"""Unit-aware read.

One tool, three observation flavours, all backed by the same scope
filters as the search tools (file_ids / section_ids / page_range)
plus an optional ``unit_ids`` for direct addressing:

* ``unit_type="page"``   → ``PageReadObservation`` with markdown + tables
                           + (page-only) optional VLM summary, AND a
                           ``children`` block listing the passage_ids
                           and table_row_ids on each page so the agent
                           can drill into fine-grained evidence without
                           a second discovery step.
* ``unit_type="passage"`` → ``PassageReadObservation`` carrying just
                           the cleaned passage text + parent section.
* ``unit_type="table_row"`` → ``TableRowReadObservation`` carrying
                           html + flattened text + parent section.

Plant only mints WitnessClaim / ValueClaim from a *ReadObservation
whose ``unit_type`` matches the obligation's. Returned units are
sorted in document order; the agent can rely on that for
"contiguous, in-order" reading even across multiple unit_ids.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import tiktoken

from agentic.tools.acquisition._common import Scope, err, ok, parse_scope
from agentic.tools.acquisition._vlm import VlmReader, default_vlm_reader
from agentic.tools.base import BaseTool
from storage.inventory_store import InventoryStore
from storage.page_store import PageAsset, PageStore, make_global_id

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


logger = logging.getLogger(__name__)


_VALID_UNIT_TYPES = {"page", "passage", "table_row"}
_VALID_MODES = {"text", "text_with_img"}
_DEFAULT_VLM_PARALLELISM = 4


class ReadTool(BaseTool):
    def __init__(
        self,
        page_store: PageStore,
        inventory: InventoryStore,
        vlm_reader: Optional[VlmReader] = None,
        vlm_parallelism: int = _DEFAULT_VLM_PARALLELISM,
    ) -> None:
        self.page_store = page_store
        self.inventory = inventory
        self.vlm_reader = vlm_reader if vlm_reader is not None else default_vlm_reader()
        self.vlm_parallelism = max(1, int(vlm_parallelism))
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    def name(self) -> str:
        return "read"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "read",
                "description": (
                    "Read units of a chosen granularity from one file. "
                    "Returns verbatim text in document order.\n\n"
                    "Pick `unit_type` to match the obligation you intend "
                    "to ingest a claim against:\n"
                    "* page: full page markdown + table blocks + optional "
                    "VLM. Each entry includes `children.passage_ids` and "
                    "`children.table_row_ids` so you can drill in.\n"
                    "* passage: paragraph-level atoms.\n"
                    "* table_row: single table rows.\n\n"
                    "Address units either by `unit_ids` (precise) OR by "
                    "scope (`file_ids`, `section_ids`, `page_range`). "
                    "Plant accepts WitnessClaim / ValueClaim only when "
                    "the cited unit_id is in this observation's `units` "
                    "AND the obligation's unit_type equals this read's."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "unit_type": {
                            "type": "string",
                            "enum": sorted(_VALID_UNIT_TYPES),
                            "default": "page",
                        },
                        "unit_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Precise unit ids to read.",
                        },
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional file allow-list (only used when unit_ids omitted).",
                        },
                        "section_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional section allow-list.",
                        },
                        "page_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional [start, end] page-number filter.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": sorted(_VALID_MODES),
                            "default": "text",
                            "description": "VLM extras (page only).",
                        },
                    },
                    "required": [],
                },
            },
        }

    # ---------------------------------------------------------- execute

    def execute(
        self,
        context: "AgentContext",
        unit_type: str = "page",
        unit_ids: Optional[List[str]] = None,
        file_ids: Optional[List[str]] = None,
        section_ids: Optional[List[str]] = None,
        page_range: Optional[List[int]] = None,
        mode: str = "text",
    ) -> Tuple[str, Dict[str, Any]]:
        if unit_type not in _VALID_UNIT_TYPES:
            return _bad_arg(f"unit_type must be one of {sorted(_VALID_UNIT_TYPES)}")
        if mode not in _VALID_MODES:
            return _bad_arg(f"mode must be one of {sorted(_VALID_MODES)}")

        if unit_ids:
            target_ids = [str(u).strip() for u in unit_ids if str(u).strip()]
        else:
            scope, scope_err = parse_scope(file_ids, page_range, section_ids, inventory=self.inventory)
            if scope_err is not None:
                return _bad_arg(scope_err)
            target_ids = sorted(self.inventory.units(
                unit_type,
                file_ids=list(scope.file_ids) if scope.file_ids else None,
                section_ids=list(scope.section_ids) if scope.section_ids else None,
            ))
            if scope.page_range is not None:
                target_ids = self._filter_by_page_range(target_ids, unit_type, scope.page_range)

        if not target_ids:
            return _empty(unit_type)

        if unit_type == "page":
            return self._read_pages(context, target_ids, mode=mode)
        if unit_type == "passage":
            return self._read_passages(context, target_ids)
        return self._read_table_rows(context, target_ids)

    # ---------------------------------------------------------- page mode

    def _read_pages(
        self,
        context: "AgentContext",
        global_ids: List[str],
        *,
        mode: str,
    ) -> Tuple[str, Dict[str, Any]]:
        units: List[Dict[str, Any]] = []
        new_pages, already_read, not_found = [], [], []
        vlm_targets: List[Tuple[int, PageAsset]] = []
        total_tokens = 0

        for gid in global_ids:
            page = self.page_store.get(gid)
            if page is None:
                not_found.append(gid)
                units.append({"unit_id": gid, "status": "not_found"})
                continue
            entry = self._page_unit_entry(page)
            if context.is_page_read(gid):
                entry["status"] = "already_read"
                already_read.append(gid)
            else:
                new_pages.append(gid)
                if mode == "text_with_img":
                    vlm_targets.append((len(units), page))
            units.append(entry)

        if vlm_targets and self.vlm_reader is not None:
            workers = min(self.vlm_parallelism, len(vlm_targets))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._vlm_one, page): idx for idx, page in vlm_targets}
                for fut, idx in futures.items():
                    try:
                        units[idx].update(fut.result())
                    except Exception as exc:
                        logger.warning("read: VLM future raised: %s", exc)
                        units[idx].update({"vlm_error": str(exc)})

        for u in units:
            if u.get("status") in {"not_found", "already_read"}:
                continue
            total_tokens += len(self.tokenizer.encode(u.get("text", "") or ""))
            if u.get("vlm_summary"):
                total_tokens += len(self.tokenizer.encode(u["vlm_summary"]))
            context.mark_page_as_read(u["unit_id"])

        units.sort(key=_page_unit_sort_key)

        context.add_retrieval_log(
            tool_name="read",
            tokens=total_tokens,
            metadata={
                "unit_type": "page",
                "requested": len(global_ids),
                "new": len(new_pages),
                "already_read": len(already_read),
                "not_found": len(not_found),
            },
        )

        return (
            ok(
                "PageReadObservation",
                unit_type="page",
                units=units,
                summary={
                    "requested": len(global_ids),
                    "new": len(new_pages),
                    "already_read": len(already_read),
                    "not_found": len(not_found),
                },
            ),
            {
                "retrieved_tokens": total_tokens,
                "unit_type": "page",
                "new_pages_count": len(new_pages),
            },
        )

    def _page_unit_entry(self, page: PageAsset) -> Dict[str, Any]:
        passages = []
        try:
            passages = [
                p.passage_id
                for p in self.inventory.passage_store.passages_for_page(page.file_id, page.page_id)
            ]
        except Exception:
            pass
        rows = []
        try:
            rows = [
                r.table_row_id
                for r in self.inventory.table_row_store.rows_for_file(page.file_id)
                if r.page_id == page.page_id
            ]
        except Exception:
            pass
        return {
            "unit_id": page.global_id,
            "file_id": page.file_id,
            "page_id": page.page_id,
            "page_number": page.page_number,
            "text": page.text_markdown or "",
            "table_blocks": page.table_blocks,
            "image_refs": [
                {"image_id": b.get("image_id"), "path": b.get("path"), "type": b.get("type")}
                for b in page.image_blocks
            ],
            "vlm_summary": "",
            "vlm_extracted_items": [],
            "children": {
                "passage_ids": passages,
                "table_row_ids": rows,
            },
        }

    def _vlm_one(self, page: PageAsset) -> Dict[str, Any]:
        if self.vlm_reader is None or not page.page_image_path:
            return {"vlm_summary": "", "vlm_error": None if page.page_image_path else "no_page_image"}
        try:
            result = self.vlm_reader(page.page_image_path, page) or {}
        except Exception as exc:
            return {"vlm_summary": "", "vlm_error": "exception", "vlm_error_message": str(exc)}
        out = {
            "vlm_summary": result.get("summary", "") or "",
            "vlm_extracted_items": result.get("items", []) or [],
        }
        if result.get("error"):
            out["vlm_error"] = result["error"]
            if result.get("error_message"):
                out["vlm_error_message"] = result["error_message"]
        return out

    # ---------------------------------------------------------- passage / row modes

    def _read_passages(
        self,
        context: "AgentContext",
        passage_ids: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        units: List[Dict[str, Any]] = []
        not_found: List[str] = []
        total = 0
        store = self.inventory.passage_store
        for pid in passage_ids:
            atom = store.get(pid)
            if atom is None:
                not_found.append(pid)
                units.append({"unit_id": pid, "status": "not_found"})
                continue
            text = atom.text or ""
            units.append(
                {
                    "unit_id": atom.passage_id,
                    "file_id": atom.file_id,
                    "page_id": atom.page_id,
                    "page_number": atom.page_number,
                    "parent_section_id": atom.parent_section_id,
                    "block_label": atom.block_label,
                    "text": text,
                }
            )
            total += len(self.tokenizer.encode(text))

        units.sort(key=_atom_unit_sort_key)
        context.add_retrieval_log(
            tool_name="read",
            tokens=total,
            metadata={"unit_type": "passage", "requested": len(passage_ids), "not_found": len(not_found)},
        )
        return (
            ok(
                "PassageReadObservation",
                unit_type="passage",
                units=units,
                summary={"requested": len(passage_ids), "not_found": len(not_found)},
            ),
            {"retrieved_tokens": total, "unit_type": "passage"},
        )

    def _read_table_rows(
        self,
        context: "AgentContext",
        row_ids: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        units: List[Dict[str, Any]] = []
        not_found: List[str] = []
        total = 0
        store = self.inventory.table_row_store
        for rid in row_ids:
            atom = store.get(rid)
            if atom is None:
                not_found.append(rid)
                units.append({"unit_id": rid, "status": "not_found"})
                continue
            text = atom.text or ""
            units.append(
                {
                    "unit_id": atom.table_row_id,
                    "file_id": atom.file_id,
                    "page_id": atom.page_id,
                    "page_number": atom.page_number,
                    "parent_section_id": atom.parent_section_id,
                    "table_index": atom.table_index,
                    "row_index": atom.row_index,
                    "is_header_row": atom.is_header_row,
                    "html": atom.html,
                    "text": text,
                }
            )
            total += len(self.tokenizer.encode(text))

        units.sort(key=_table_row_unit_sort_key)
        context.add_retrieval_log(
            tool_name="read",
            tokens=total,
            metadata={"unit_type": "table_row", "requested": len(row_ids), "not_found": len(not_found)},
        )
        return (
            ok(
                "TableRowReadObservation",
                unit_type="table_row",
                units=units,
                summary={"requested": len(row_ids), "not_found": len(not_found)},
            ),
            {"retrieved_tokens": total, "unit_type": "table_row"},
        )

    # ---------------------------------------------------------- helpers

    def _filter_by_page_range(
        self,
        target_ids: List[str],
        unit_type: str,
        page_range: Tuple[int, int],
    ) -> List[str]:
        lo, hi = page_range
        out: List[str] = []
        for uid in target_ids:
            page_no = self._unit_page_number(uid, unit_type)
            if page_no is not None and lo <= page_no <= hi:
                out.append(uid)
        return out

    def _unit_page_number(self, unit_id: str, unit_type: str) -> Optional[int]:
        if unit_type == "page":
            page = self.page_store.get(unit_id)
            return page.page_number if page else None
        if unit_type == "passage":
            atom = self.inventory.passage_store.get(unit_id)
            return atom.page_number if atom else None
        atom = self.inventory.table_row_store.get(unit_id)
        return atom.page_number if atom else None


# ---------------------------------------------------------------- module helpers


def _bad_arg(message: str) -> Tuple[str, Dict[str, Any]]:
    return (
        err("invalid_argument", message),
        {"error": "invalid_argument"},
    )


def _empty(unit_type: str) -> Tuple[str, Dict[str, Any]]:
    obs = {
        "page": "PageReadObservation",
        "passage": "PassageReadObservation",
        "table_row": "TableRowReadObservation",
    }[unit_type]
    return (
        ok(obs, unit_type=unit_type, units=[], summary={"requested": 0}),
        {"retrieved_tokens": 0, "unit_type": unit_type},
    )


def _page_unit_sort_key(u: Dict[str, Any]) -> Tuple[Any, ...]:
    return (u.get("file_id") or "", u.get("page_number") or 0, u.get("unit_id") or "")


def _atom_unit_sort_key(u: Dict[str, Any]) -> Tuple[Any, ...]:
    return (u.get("file_id") or "", u.get("page_number") or 0, u.get("unit_id") or "")


def _table_row_unit_sort_key(u: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        u.get("file_id") or "",
        u.get("page_number") or 0,
        u.get("table_index") or 0,
        u.get("row_index") or 0,
    )


__all__ = ["ReadTool"]
