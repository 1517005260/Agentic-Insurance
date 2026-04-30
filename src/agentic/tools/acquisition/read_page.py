"""The single read primitive.

Pages are addressed by their **global id** ``"file_id/page_id"`` (page_id
alone collides across files since each file's first page is ``p_0001``).
The schema accepts either:

* ``page_ids``: list of global IDs, e.g. ``["fileA_xxx/p_0001", ...]``
* or ``file_id`` + ``page_ids`` (per-file page IDs, joined automatically)

Returns a structured ``PageObservation`` per page: Markdown text, embedded
tables, image refs, optional VLM summary, source citation. The mode parameter
selects channels:

* ``text``           — Markdown only.
* ``text_with_img``  — Markdown + VLM read of the rendered page image.
* ``auto``           — defer to each page's own ``page_mode`` flag.

VLM is pluggable via ``vlm_reader``; when unset (or no image), the VLM
block is empty.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import tiktoken

from agentic.tools.base import BaseTool
from storage.page_store import PageAsset, PageStore, make_global_id

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


_VALID_MODES = {"auto", "text", "text_with_img"}


class ReadPageTool(BaseTool):
    def __init__(self, page_store: PageStore, vlm_reader=None):
        self.page_store = page_store
        self.vlm_reader = vlm_reader
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    @property
    def name(self) -> str:
        return "read_page"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "read_page",
                "description": (
                    "Read the full content of one or more pages. This is the "
                    "canonical read primitive — search tools only return "
                    "abbreviated snippets, so call read_page on candidate pages "
                    "before answering.\n\n"
                    "Pages are addressed by their global id 'file_id/page_id' "
                    "(e.g. 'axa_abcd1234/p_0001'). Pass them in `page_ids`. "
                    "Alternatively, supply `file_id` and pass per-file page ids "
                    "in `page_ids` (e.g. 'p_0001').\n\n"
                    "Each result is a structured PageObservation. Previously "
                    "read pages are flagged so you do not re-cite stale snippets."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Global page ids ('file_id/page_id') or per-file page ids.",
                        },
                        "file_id": {
                            "type": "string",
                            "description": "Optional file id; joined with each per-file page_id.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["auto", "text", "text_with_img"],
                            "default": "auto",
                        },
                    },
                    "required": ["page_ids"],
                },
            },
        }

    def _resolve_effective_mode(self, page: PageAsset, mode: str) -> str:
        if mode == "auto":
            return page.page_mode if page.page_mode in {"text", "text_with_img"} else "text"
        return mode

    def _read_text(self, page: PageAsset) -> Dict[str, Any]:
        return {
            "text_markdown": page.text_markdown,
            "table_blocks": page.table_blocks,
            "image_refs": [
                {"image_id": b.get("image_id"), "path": b.get("path"), "type": b.get("type")}
                for b in page.image_blocks
            ],
        }

    def _read_vlm(self, page: PageAsset) -> Dict[str, Any]:
        if self.vlm_reader is None or not page.page_image_path:
            return {"vlm_summary": "", "vlm_extracted_items": []}
        result = self.vlm_reader(page.page_image_path, page) or {}
        return {
            "vlm_summary": result.get("summary", ""),
            "vlm_extracted_items": result.get("items", []),
        }

    def _build_observation(self, page: PageAsset, mode: str) -> Dict[str, Any]:
        observation: Dict[str, Any] = {
            "observation_type": "PageObservation",
            "global_id": page.global_id,
            "file_id": page.file_id,
            "page_id": page.page_id,
            "page_number": page.page_number,
            "page_mode": mode,
        }
        observation.update(self._read_text(page))
        if mode == "text_with_img":
            observation.update(self._read_vlm(page))
        else:
            observation["vlm_summary"] = ""
            observation["vlm_extracted_items"] = []
        observation["source_citation"] = {
            "file_id": page.file_id,
            "page_number": page.page_number,
        }
        return observation

    def execute(
        self,
        context: "AgentContext",
        page_ids: List[str] = None,
        page_id: str = None,
        file_id: Optional[str] = None,
        mode: str = "auto",
    ) -> Tuple[str, Dict[str, Any]]:
        if page_ids is None:
            if page_id is not None:
                page_ids = [str(page_id)]
            else:
                return "Error: No page IDs provided", {"retrieved_tokens": 0}

        if mode not in _VALID_MODES:
            return (
                f"Error: invalid mode '{mode}'. Expected one of {sorted(_VALID_MODES)}.",
                {"retrieved_tokens": 0, "error": "invalid_mode"},
            )

        # Normalize each page_id to a global id.
        normalized: List[str] = []
        for pid in page_ids:
            pid = str(pid)
            if "/" in pid:
                normalized.append(pid)
            elif file_id:
                normalized.append(make_global_id(file_id, pid))
            else:
                normalized.append(pid)  # fallback: per-file lookup

        observations: List[Dict[str, Any]] = []
        already_read: List[str] = []
        new_pages_read: List[str] = []
        not_found: List[str] = []
        total_tokens = 0

        for pid in normalized:
            page = self.page_store.get(pid)
            if page is None:
                not_found.append(pid)
                observations.append({"global_id": pid, "status": "not_found"})
                continue

            gid = page.global_id
            effective_mode = self._resolve_effective_mode(page, mode)

            if context.is_page_read(gid):
                already_read.append(gid)
                observations.append(
                    {
                        "observation_type": "PageObservation",
                        "global_id": gid,
                        "file_id": page.file_id,
                        "page_id": page.page_id,
                        "status": "already_read",
                        "note": "This page has been read before in the current session.",
                    }
                )
                continue

            obs = self._build_observation(page, effective_mode)
            observations.append(obs)
            new_pages_read.append(gid)
            total_tokens += len(self.tokenizer.encode(obs.get("text_markdown", "")))
            if obs.get("vlm_summary"):
                total_tokens += len(self.tokenizer.encode(obs["vlm_summary"]))

            context.mark_page_as_read(gid)

        tool_result = json.dumps(
            {"observation_type": "ReadPageResult", "results": observations},
            ensure_ascii=False,
            indent=2,
        )

        context.add_retrieval_log(
            tool_name="read_page",
            tokens=total_tokens,
            metadata={
                "page_ids_requested": normalized,
                "new_pages_read": new_pages_read,
                "already_read": already_read,
                "not_found": not_found,
                "mode": mode,
            },
        )

        return tool_result, {
            "retrieved_tokens": total_tokens,
            "new_pages_count": len(new_pages_read),
            "already_read_count": len(already_read),
            "not_found_count": len(not_found),
        }
