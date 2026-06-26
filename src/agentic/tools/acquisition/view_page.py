"""view_page — load a compiled page image into the agent's multimodal context.

The shell sandbox returns text, so a page's RENDERED image — tables, charts,
stamps, signatures, the visual layout the OCR Markdown flattens — is invisible
to a text-only locator. ``view_page`` is the read affordance for it: given a
page path under the corpus, it resolves the page image emitted next to the page
Markdown and hands its real bytes back to the agent loop, which injects them as
an image into the next turn so a vision model SEES the page.

It is a host-side tool, not a sandbox script, on purpose: OpenAI-compatible APIs
accept images only in a user message (not a tool result), so the harness — not
the hermetic shell — has to carry the pixels into the conversation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from agentic.tools.acquisition._common import err, ok, safe_resolve_path
from agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from agentic.core.context import AgentContext

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


class ViewPageTool(BaseTool):
    """Resolve a page image under the corpus and surface it for injection."""

    def __init__(self, corpus_root: Path):
        self.corpus_root = Path(corpus_root).resolve()

    @property
    def name(self) -> str:
        return "view_page"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "view_page",
                "description": (
                    "Look at the RENDERED image of a page — what the OCR text "
                    "cannot convey: tables, charts, stamps, signatures, figures, "
                    "the visual layout. Pass the path of a page under the corpus, "
                    "e.g. `documents/<doc>/pages/page_0005.md` or its sibling "
                    "`page_0005.jpg`. The page image is added to the conversation "
                    "so you can read it directly. Not every corpus has page images."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to a page (its .md or image) relative to the corpus root.",
                        }
                    },
                    "required": ["path"],
                },
            },
        }

    def _resolve_image(self, path: str) -> Optional[Path]:
        """Accept the page .md, the image itself, or an extensionless page id;
        the on-disk image sits next to the page Markdown as ``page_NNNN.<ext>``.

        Tolerant of the leading ``documents/`` segment either way: the base agent
        roots its shell at ``documents/`` (so it refers to ``d_0001/…``) while
        the graph agent roots at the FS root (``documents/d_0001/…``), and the
        model doesn't always match the convention."""
        candidates = [path]
        if path.startswith("documents/"):
            candidates.append(path[len("documents/"):])
        else:
            candidates.append("documents/" + path)
        for cand in candidates:
            direct = safe_resolve_path(self.corpus_root, cand)
            if direct and direct.suffix.lower() in _IMAGE_EXTS:
                return direct
            suffix = Path(cand).suffix
            stem_rel = cand[: -len(suffix)] if suffix else cand
            for ext in _IMAGE_EXTS:
                hit = safe_resolve_path(self.corpus_root, stem_rel + ext)
                if hit:
                    return hit
        return None

    def execute(self, context: "AgentContext", path: str = "", **_: Any) -> Tuple[str, Dict[str, Any]]:
        path = (path or "").strip()
        if not path:
            return err(
                "invalid_argument",
                "`path` must be a page path under the corpus.",
                valid_example={"path": "documents/d_0001/pages/page_0005.md"},
            ), {"error": "invalid_argument"}

        image = self._resolve_image(path)
        if image is None:
            return err(
                "not_found",
                f"No page image for '{path}'. `ls` the page directory to see the "
                "`page_NNNN.jpg` next to `page_NNNN.md` — not every corpus has images.",
            ), {"error": "not_found"}

        rel = str(image.relative_to(self.corpus_root))
        context.add_retrieval_log(tool_name="view_page", tokens=0, metadata={"image": rel})
        # ``image_paths`` in the tool log is the signal the agent loop watches:
        # it reads these files and appends them as an image user-message.
        return (
            ok("PageImageObservation", path=rel, note="Page image added below — read it directly."),
            {"retrieved_tokens": 0, "image_paths": [str(image)], "image_labels": [rel]},
        )
