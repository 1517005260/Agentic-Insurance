"""The single read primitive.

Pages are addressed by their **global id** ``"file_id/page_id"`` (page_id
alone collides across files since each file's first page is ``p_0001``).
The schema accepts either:

* ``page_ids``: list of global IDs, e.g. ``["fileA_xxx/p_0001", ...]``
* or ``file_id`` + ``page_ids`` (per-file page IDs, joined automatically)

Returns a structured ``PageObservation`` per page: Markdown text, embedded
tables, image refs, optional VLM summary, source citation. The mode
parameter selects channels:

* ``text``           — Markdown only.
* ``text_with_img``  — Markdown + parallel VLM read of the rendered page image.
* ``auto``           — defer to each page's own ``page_mode`` flag.

VLM is pluggable via the ``vlm_reader`` callable; if not supplied the
tool builds the default OpenAI-compat reader from the ``VLM_*`` env
vars. The reader is invoked **in parallel across pages** so a multi-
page read does not block sequentially on each VLM HTTP round-trip.
VLM failures degrade gracefully — Markdown is still returned and the
error is surfaced on the per-page observation under ``vlm_error``.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import tiktoken

from agentic.tools.acquisition._common import err
from agentic.tools.acquisition._vlm import VlmReader, default_vlm_reader
from agentic.tools.base import BaseTool
from storage.page_store import PageAsset, PageStore, make_global_id

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


logger = logging.getLogger(__name__)


_VALID_MODES = {"auto", "text", "text_with_img"}
_DEFAULT_VLM_PARALLELISM = 4


class ReadPageTool(BaseTool):
    def __init__(
        self,
        page_store: PageStore,
        vlm_reader: Optional[VlmReader] = None,
        vlm_parallelism: int = _DEFAULT_VLM_PARALLELISM,
    ):
        self.page_store = page_store
        self.vlm_reader = vlm_reader if vlm_reader is not None else default_vlm_reader()
        self.vlm_parallelism = max(1, int(vlm_parallelism))
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

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
                    "Every result is a structured PageObservation containing "
                    "Markdown text, embedded tables, image refs, and (for "
                    "figure/table/chart-heavy pages) a VLM summary of the "
                    "rendered page image. A re-fetch of a page already read "
                    "in this trajectory still returns the full snapshot but "
                    "with `status='already_read'` set so the agent can decide "
                    "whether the prior read is stale; the snapshot is always "
                    "available so WitnessClaim citations can quote spans."
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
            return {
                "vlm_summary": "",
                "vlm_extracted_items": [],
                "vlm_error": None if page.page_image_path else "no_page_image",
            }
        try:
            result = self.vlm_reader(page.page_image_path, page) or {}
        except Exception as exc:
            logger.warning("read_page: vlm_reader raised: %s", exc)
            return {
                "vlm_summary": "",
                "vlm_extracted_items": [],
                "vlm_error": "exception",
                "vlm_error_message": str(exc),
            }
        out: Dict[str, Any] = {
            "vlm_summary": result.get("summary", "") or "",
            "vlm_extracted_items": result.get("items", []) or [],
        }
        if result.get("error"):
            out["vlm_error"] = result["error"]
            if result.get("error_message"):
                out["vlm_error_message"] = result["error_message"]
        return out

    def _build_observation(self, page: PageAsset, mode: str) -> Dict[str, Any]:
        # Build a Markdown-only skeleton; VLM merging happens after the
        # parallel batch finishes (see ``execute``).
        observation: Dict[str, Any] = {
            "observation_type": "PageObservation",
            "global_id": page.global_id,
            "file_id": page.file_id,
            "page_id": page.page_id,
            "page_number": page.page_number,
            "page_mode": mode,
        }
        observation.update(self._read_text(page))
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
                return (
                    err(
                        "invalid_argument",
                        "No page IDs provided. Pass `page_ids` (list).",
                        remediation="Pass `page_ids` as a list of global ids ('file_id/page_id') or per-file page ids (with `file_id` set). Discover page_ids from a search tool's results first.",
                        valid_example={"page_ids": ["<file_id>/p_0001", "<file_id>/p_0002"]},
                    ),
                    {"retrieved_tokens": 0, "error": "invalid_argument"},
                )

        if mode not in _VALID_MODES:
            return (
                err(
                    "invalid_argument",
                    f"Invalid mode {mode!r}. Expected one of {sorted(_VALID_MODES)}.",
                    remediation="Set `mode` to 'auto' (default), 'text', or 'text_with_img'; or omit the field.",
                    valid_example={"mode": "auto"},
                    mode=mode,
                ),
                {"retrieved_tokens": 0, "error": "invalid_argument"},
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

        # Pass 1: build the Markdown skeletons synchronously and collect
        # the (index, page) pairs that need a VLM call. We mark pages
        # read AFTER the VLM batch so a failed batch doesn't poison the
        # session's already-read set.
        vlm_targets: List[Tuple[int, PageAsset]] = []
        for pid in normalized:
            page = self.page_store.get(pid)
            if page is None:
                not_found.append(pid)
                observations.append({"global_id": pid, "status": "not_found"})
                continue

            gid = page.global_id
            effective_mode = self._resolve_effective_mode(page, mode)

            obs = self._build_observation(page, effective_mode)
            if context.is_page_read(gid):
                # The page was already read once this session; the
                # ProofAgent still needs the text snapshot to build
                # WitnessClaim citations against, so we always return
                # the full PageObservation. The repeat marker is
                # informational — token-budget bookkeeping treats this
                # call as a re-read.
                obs["status"] = "already_read"
                obs["note"] = "Re-read; full snapshot returned for citation."
                already_read.append(gid)
                observations.append(obs)
                continue

            observations.append(obs)
            new_pages_read.append(gid)
            if effective_mode == "text_with_img":
                vlm_targets.append((len(observations) - 1, page))

        # Pass 2: parallel VLM calls for the figure-heavy pages.
        if vlm_targets and self.vlm_reader is not None:
            workers = min(self.vlm_parallelism, len(vlm_targets))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._read_vlm, page): idx for idx, page in vlm_targets
                }
                for fut, idx in futures.items():
                    try:
                        observations[idx].update(fut.result())
                    except Exception as exc:
                        logger.warning("read_page: VLM future raised: %s", exc)
                        observations[idx].update(
                            {
                                "vlm_summary": "",
                                "vlm_extracted_items": [],
                                "vlm_error": "exception",
                                "vlm_error_message": str(exc),
                            }
                        )

        # Pass 3: token accounting and read-set marking. Token cost is
        # measured against the final text + VLM summary so an empty VLM
        # block does not inflate the cost.
        for obs in observations:
            if obs.get("status") in {"already_read", "not_found"}:
                continue
            total_tokens += len(self.tokenizer.encode(obs.get("text_markdown", "") or ""))
            if obs.get("vlm_summary"):
                total_tokens += len(self.tokenizer.encode(obs["vlm_summary"]))
            context.mark_page_as_read(obs["global_id"])

        # Top-level summary so the agent can tell at a glance whether
        # any page failed without iterating per-result statuses (a
        # full-fail batch otherwise looks like a successful tool call
        # with cryptic-looking stubs).
        summary = {
            "requested": len(normalized),
            "new_pages_read": len(new_pages_read),
            "already_read": len(already_read),
            "not_found": len(not_found),
        }
        tool_result = json.dumps(
            {
                "observation_type": "ReadPageResult",
                "ok": True,
                "summary": summary,
                "results": observations,
            },
            ensure_ascii=False,
            separators=(",", ":"),
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
