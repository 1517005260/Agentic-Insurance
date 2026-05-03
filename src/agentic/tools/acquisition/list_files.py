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
                    "List indexed files in the corpus. Returns one entry per "
                    "file with file_id, original filename, page_count, "
                    "parse_status, and upload_time (ISO 8601 UTC).\n\n"
                    "Defaults to the 10 most recently uploaded files, ordered "
                    "newest first. Use `recent_n` to widen or narrow the "
                    "window, and `filename_regex` to filter by filename "
                    "(Python regex, case-insensitive). When `filename_regex` "
                    "is provided we filter the FULL corpus first and then "
                    "take the most recent `recent_n` matches.\n\n"
                    "This tool is read-only and cheap — call it whenever you "
                    "are unsure which files exist."
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
                            "description": (
                                "Optional Python regex matched against the "
                                "original filename (case-insensitive)."
                            ),
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

        rows = self._scan()
        if compiled is not None:
            rows = [r for r in rows if compiled.search(r["filename"])]
        rows.sort(key=lambda r: r["_upload_ts"], reverse=True)
        total_matched = len(rows)  # pre-truncation count (the actual match population)
        truncated = total_matched > limit
        rows = rows[:limit]

        files = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        log_meta = {
            "files_returned": len(files),
            "total_matched": total_matched,
            "recent_n": limit,
            "filename_regex": filename_regex,
            "truncated": truncated,
        }
        context.add_retrieval_log(tool_name="list_files", tokens=0, metadata=log_meta)

        return (
            ok(
                "FileListObservation",
                files=files,
                truncated=truncated,
                total_matched=total_matched,
                # `total_matched` is the count BEFORE truncation so the
                # agent can decide whether raising `recent_n` is worth it;
                # `truncated` is the boolean shortcut.
            ),
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
