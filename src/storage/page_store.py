"""Page asset record and JSON-backed in-memory store.

The canonical key is the **global page id** ``f"{file_id}/{page_id}"`` —
``page_id`` alone collides across files (every file's first page is
``p_0001``). Use :meth:`PageAsset.global_id` or :func:`make_global_id`
when you need a stable cross-file handle.

A ``PageStore`` can be constructed from a single ``page_assets/<file_id>.json``
file (single-file mode) or from a directory containing many such files
(multi-file mode, the default for the global agent).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


def make_global_id(file_id: str, page_id: str) -> str:
    return f"{file_id}/{page_id}"


def split_global_id(global_id: str) -> tuple[str, str]:
    file_id, _, page_id = global_id.partition("/")
    return file_id, page_id


@dataclass
class PageAsset:
    page_id: str
    file_id: str
    page_number: Optional[int] = None
    text_markdown: str = ""
    page_image_path: Optional[str] = None
    table_blocks: List[Dict[str, Any]] = field(default_factory=list)
    image_blocks: List[Dict[str, Any]] = field(default_factory=list)
    layout_blocks: List[Dict[str, Any]] = field(default_factory=list)
    page_mode: str = "text"  # "text" | "text_with_img"
    quality_flags: Dict[str, bool] = field(default_factory=dict)

    @property
    def global_id(self) -> str:
        return make_global_id(self.file_id, self.page_id)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageAsset":
        return cls(
            page_id=str(data.get("page_id") or data.get("id")),
            file_id=str(data.get("file_id", "")),
            page_number=data.get("page_number"),
            text_markdown=data.get("text_markdown") or data.get("text") or "",
            page_image_path=data.get("page_image_path"),
            table_blocks=data.get("table_blocks", []) or [],
            image_blocks=data.get("image_blocks", []) or [],
            layout_blocks=data.get("layout_blocks", []) or [],
            page_mode=data.get("page_mode", "text"),
            quality_flags=data.get("quality_flags", {}) or {},
        )


class PageStore:
    """Keyed by global_id (``f"{file_id}/{page_id}"``).

    ``source`` may be a single JSON file or a directory of JSON files.
    Multi-file mode is the default global view for the agent loop.
    """

    def __init__(self, source: Union[str, Path]):
        self.source = Path(source)
        self._by_global_id: Dict[str, PageAsset] = {}
        self._load()

    def _load(self):
        files: Iterable[Path]
        if self.source.is_dir():
            files = sorted(self.source.glob("*.json"))
        else:
            files = [self.source]
        for fp in files:
            data = json.loads(fp.read_text(encoding="utf-8"))
            for item in data:
                page = PageAsset.from_dict(item)
                if page.page_id:
                    self._by_global_id[page.global_id] = page

    def get(self, key: str) -> Optional[PageAsset]:
        """Look up by ``"file_id/page_id"`` (global) or by bare ``page_id``.

        Bare page_id resolution is only allowed when **exactly one** asset
        in the store has that page_id; otherwise we return ``None`` and
        force the caller to disambiguate. This avoids the classic bug
        where ``p_0001`` from file A silently matches file B's first page.
        """
        if "/" in key:
            return self._by_global_id.get(key)
        matches = [page for page in self._by_global_id.values() if page.page_id == key]
        if len(matches) == 1:
            return matches[0]
        return None

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        return len(self._by_global_id)

    def ids(self) -> List[str]:
        return list(self._by_global_id.keys())
