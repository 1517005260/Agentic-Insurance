"""Enumerate indexed files.

Source of truth: ``page_assets/<file_id>.json`` — every successfully
parsed file lands there. We do not re-walk paddle_ocr to discover files
because page_assets is the *post-success* set; an unfinished parse won't
be in page_assets and shouldn't be visible to the agent.

Filename + upload time are pulled from ``paddle_ocr/<file_id>/meta.json``
(``source_path`` -> basename, mtime of the meta file -> upload_time).
We tolerate a missing paddle meta and return ``"unknown"`` rather than
hiding the file from the listing.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import regex as ureg

from agentic.tools.acquisition._common import err, ok
from agentic.tools.base import BaseTool
from config.settings import page_assets_root, paddle_ocr_root

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


_DEFAULT_RECENT_N = 10


class ListFilesTool(BaseTool):
    def __init__(
        self,
        page_assets_dir: Optional[Path] = None,
        paddle_ocr_dir: Optional[Path] = None,
    ):
        self._page_assets_dir = Path(page_assets_dir) if page_assets_dir else page_assets_root()
        self._paddle_ocr_dir = Path(paddle_ocr_dir) if paddle_ocr_dir else paddle_ocr_root()

    @property
    def name(self) -> str:
        return "list_files"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": (
                    "List indexed files (file_id, filename, page_count, "
                    "parse_status, upload_time). Defaults to the 10 most "
                    "recent; `recent_n` widens. `filename_regex` (case-"
                    "insensitive Python regex) filters the full corpus then "
                    "keeps the most recent `recent_n` matches. Cheap; call "
                    "whenever unsure which files exist."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recent_n": {
                            "type": "integer",
                            "description": "Maximum entries to return; default 10.",
                        },
                        "filename_regex": {
                            "type": "string",
                            "description": "Optional Python regex on filename (case-insensitive).",
                        },
                    },
                    "required": [],
                },
            },
        }

    # ------------------------------------------------------------- internals

    def _scan(self) -> List[Dict[str, Any]]:
        """Walk page_assets and collect a row per file.

        Each row carries everything we need *before* applying recent_n /
        filename_regex so the caller can filter without re-IO.
        """
        if not self._page_assets_dir.is_dir():
            return []

        rows: List[Dict[str, Any]] = []
        for asset_path in sorted(self._page_assets_dir.glob("*.json")):
            file_id = asset_path.stem
            page_count, parse_status = self._read_page_count(asset_path)
            filename, upload_time = self._read_paddle_meta(file_id, asset_path)
            rows.append(
                {
                    "file_id": file_id,
                    "filename": filename,
                    "page_count": page_count,
                    "parse_status": parse_status,
                    "upload_time": upload_time,
                    # Use a numeric key for sorting; the ISO string is for the agent.
                    "_upload_ts": _coerce_ts(upload_time, asset_path),
                }
            )
        return rows

    @staticmethod
    def _read_page_count(asset_path: Path) -> Tuple[int, str]:
        try:
            data = json.loads(asset_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0, "broken"
        if not isinstance(data, list):
            return 0, "broken"
        return len(data), "ready"

    def _read_paddle_meta(
        self, file_id: str, asset_path: Path
    ) -> Tuple[str, str]:
        meta_path = self._paddle_ocr_dir / file_id / "meta.json"
        filename = "unknown"
        upload_time: Optional[str] = None
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                source = str(meta.get("source_path") or "").strip()
                if source:
                    filename = os.path.basename(source) or filename
                upload_time = _file_iso_mtime(meta_path)
            except (OSError, json.JSONDecodeError):
                pass
        if upload_time is None:
            upload_time = _file_iso_mtime(asset_path)
        return filename, upload_time

    # ------------------------------------------------------------- execute

    def execute(
        self,
        context: "AgentContext",
        recent_n: int = _DEFAULT_RECENT_N,
        filename_regex: Optional[str] = None,
    ):
        try:
            limit = int(recent_n) if recent_n is not None else _DEFAULT_RECENT_N
        except (TypeError, ValueError):
            return err(
                "invalid_argument",
                "`recent_n` must be an integer.",
                remediation="Pass `recent_n` as a positive integer (default 10), or omit it.",
                valid_example={"recent_n": 10},
            ), {"error": "invalid_argument"}
        if limit < 1:
            return err(
                "invalid_argument",
                "`recent_n` must be >= 1.",
                remediation="Pass `recent_n` as an integer >= 1 (e.g. 10).",
                valid_example={"recent_n": 10},
            ), {"error": "invalid_argument"}

        compiled = None
        if filename_regex:
            try:
                compiled = ureg.compile(filename_regex, ureg.IGNORECASE)
            except Exception as exc:
                return (
                    err(
                        "invalid_regex",
                        f"`filename_regex` failed to compile: {exc}",
                        remediation="Re-emit `filename_regex` as a valid Python regex (the `regex` module flavor); escape literals like '.' and '(' if matching them literally.",
                        pattern=filename_regex,
                    ),
                    {"error": "invalid_regex"},
                )

        all_rows = self._scan()
        all_rows.sort(key=lambda r: r["_upload_ts"], reverse=True)

        if compiled is not None:
            matched = [r for r in all_rows if compiled.search(r["filename"])]
        else:
            matched = all_rows
        total_matched = len(matched)
        truncated = total_matched > limit
        rows = matched[:limit]

        files = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

        # Zero-hit recovery: when the LLM's filename_regex matches no
        # files, return the most-recent files anyway as a fallback so
        # the LLM has something to orient on. Filenames in this corpus
        # are routinely in a different language / use product code
        # names that the LLM's English regex won't match — without
        # this hint the agent burns turns guessing regex variants.
        recent_fallback: List[Dict[str, Any]] = []
        hint: Optional[str] = None
        if compiled is not None and total_matched == 0:
            recent_fallback = [
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in all_rows[:_DEFAULT_RECENT_N]
            ]
            hint = (
                "filename_regex matched zero files. Filenames in this "
                "corpus may be in another language than your query "
                "(e.g. Chinese vs English product names) or use code "
                "names rather than English titles. The 10 most recent "
                "files are returned in `recent_fallback` for orientation; "
                "their content may still match your query — try "
                "semantic_search / bm25_search over those file_ids."
            )

        log_meta = {
            "files_returned": len(files),
            "total_matched": total_matched,
            "recent_n": limit,
            "filename_regex": filename_regex,
            "truncated": truncated,
            "fallback_returned": len(recent_fallback),
        }
        context.add_retrieval_log(tool_name="list_files", tokens=0, metadata=log_meta)

        kwargs: Dict[str, Any] = {
            "files": files,
            "truncated": truncated,
            "total_matched": total_matched,
        }
        if recent_fallback:
            kwargs["recent_fallback"] = recent_fallback
        if hint:
            kwargs["hint"] = hint
        return (
            ok("FileListObservation", **kwargs),
            {"retrieved_tokens": 0, "files_returned": len(files), "total_matched": total_matched},
        )


# --------------------------------------------------------------------- helpers


def _file_iso_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
    except OSError:
        return "unknown"
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _coerce_ts(iso_or_unknown: str, fallback_path: Path) -> float:
    if iso_or_unknown and iso_or_unknown != "unknown":
        try:
            return datetime.fromisoformat(iso_or_unknown).timestamp()
        except ValueError:
            pass
    try:
        return fallback_path.stat().st_mtime
    except OSError:
        return 0.0
