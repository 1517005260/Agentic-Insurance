"""Default VLM reader for ``read_page`` (text_with_img mode).

The reader receives a ``(rel_image_path, PageAsset)`` pair, resolves the
on-disk file under :func:`config.settings.paddle_ocr_root`, base64-encodes
it, and asks an OpenAI-compatible vision endpoint for a structured
visual-extraction summary.

The endpoint is whatever the user configures in
``VLM_API_BASE_URL`` / ``VLM_API_KEY`` / ``VLM_MODEL``. Any
OpenAI-compatible relay works (the user's setup uses gpt-4o through a
relay; a local llama.cpp server with a vision model would also work).

Failures (missing image, missing creds, HTTP error, malformed JSON) are
absorbed and surface as ``{"summary": "", "items": [], "error": "..."}``
so the read_page tool can still return the OCR Markdown — a partial
observation is more useful to the agent than a hard failure.
"""

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

from agentic.tools.acquisition._common import safe_resolve_path
from config.settings import (
    VLM_API_BASE_URL,
    VLM_API_KEY,
    VLM_MODEL,
    paddle_ocr_root,
)
from storage.page_store import PageAsset


logger = logging.getLogger(__name__)


_PROMPT = """\
You are reading the rendered page image of a single page from a long PDF.
The OCR Markdown for this page has already been extracted separately, so
do NOT repeat the body text. Focus on what OCR is likely to miss:
- chart axes, titles, legends, and the qualitative trend
- table layout when the structure is complex (merged cells, hierarchical headers)
- figure captions, callouts, embedded text inside images
- formulas, symbols, and diagrams

Output STRICT JSON only — no Markdown fences, no prose preamble. Schema:

{
  "summary": "<2-4 sentence holistic description in the page's language>",
  "items": [
    {"text": "<one visually-extracted item, e.g. a chart-title, a table-row label, a figure caption>"}
  ]
}

If the page has no chart / table / figure / formula content (i.e. it is
plain text), return {"summary": "", "items": []}.
"""


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


VlmReader = Callable[[str, PageAsset], Dict[str, Any]]


def default_vlm_reader(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    storage_root: Optional[Path] = None,
    request_timeout: int = 120,
) -> VlmReader:
    """Build a `vlm_reader` closure that talks to a configured VLM endpoint.

    Returns a no-op reader (logs once, then returns empty payloads) when
    the VLM env vars are missing — that way an unconfigured corpus still
    runs read_page successfully on text-only pages and surfaces a clear
    diagnostic on figure-heavy ones, instead of crashing the agent loop.
    """
    base = (base_url or VLM_API_BASE_URL or "").rstrip("/")
    key = api_key or VLM_API_KEY
    mdl = model or VLM_MODEL
    root = Path(storage_root) if storage_root else paddle_ocr_root()

    if not (base and key and mdl):
        logger.warning(
            "VLM not configured (VLM_API_BASE_URL / VLM_API_KEY / VLM_MODEL); "
            "read_page text_with_img pages will return empty VLM blocks."
        )
        return _disabled_reader

    def _reader(rel_image_path: str, page: PageAsset) -> Dict[str, Any]:
        return _call_vlm(rel_image_path, page, base=base, key=key, model=mdl, root=root, timeout=request_timeout)

    return _reader


# --------------------------------------------------------------------- impl


def _disabled_reader(rel_image_path: str, page: PageAsset) -> Dict[str, Any]:
    return {
        "summary": "",
        "items": [],
        "error": "vlm_disabled",
        "error_message": "VLM endpoint is not configured.",
    }


def _call_vlm(
    rel_image_path: str,
    page: PageAsset,
    *,
    base: str,
    key: str,
    model: str,
    root: Path,
    timeout: int,
) -> Dict[str, Any]:
    # `page_image_path` is relative to ``paddle_ocr_root() / file_id``.
    # We resolve under that per-file directory specifically (NOT just
    # under paddle_ocr_root) so a malformed path with ``..`` cannot
    # escape into a sibling file's images. Absolute paths and any
    # ``..`` segments are rejected outright before resolution.
    rel = rel_image_path.lstrip("/")
    if any(part == ".." for part in rel.replace("\\", "/").split("/")):
        return _err(
            "image_not_found",
            f"page_image_path contains a '..' segment: {rel_image_path!r}",
        )
    file_root = root / page.file_id
    abs_path = safe_resolve_path(file_root, rel)
    if abs_path is None:
        return _err(
            "image_not_found",
            f"Resolved image path is missing or escapes file root: "
            f"{page.file_id}/{rel}",
        )

    try:
        with abs_path.open("rb") as fh:
            data = fh.read()
    except OSError as exc:
        return _err("image_read_failed", f"Could not read image: {exc}")

    mime = _guess_mime(abs_path.suffix.lower())
    data_uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(
            f"{base}/chat/completions", headers=headers, json=payload, timeout=timeout
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return _err("http_error", f"VLM HTTP call failed: {exc}")

    try:
        body = resp.json()
        raw = (body["choices"][0]["message"].get("content") or "").strip()
    except (ValueError, KeyError, IndexError) as exc:
        return _err("malformed_response", f"VLM response missing message content: {exc}")

    parsed = _parse_json(raw)
    if parsed is None:
        return _err("invalid_json", "VLM response was not valid JSON.", raw_preview=raw[:200])

    return {
        "summary": str(parsed.get("summary") or ""),
        "items": [
            {"text": str(item.get("text") or "")}
            for item in (parsed.get("items") or [])
            if isinstance(item, dict) and item.get("text")
        ],
    }


def _err(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    out = {"summary": "", "items": [], "error": code, "error_message": message}
    out.update(extra)
    return out


def _guess_mime(suffix: str) -> str:
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    fenced = _FENCE_RE.match(raw)
    body = fenced.group(1) if fenced else raw
    try:
        result = json.loads(body)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None
