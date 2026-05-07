"""IndexBuilder ABC.

Stores are global: each builder writes into a fixed directory under
``STORAGE_PATH/`` and re-running on a different ``file_id`` appends into the
same store. ``file_id`` is carried in row metadata so per-file filtering and
removal remain possible.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from storage.page_store import PageAsset


@dataclass
class IndexBuildResult:
    """Outcome of one builder run. ``skipped_reason`` is overloaded: it
    holds both *legitimate skips* (``no sentences extracted``, ``visual
    model not configured``, ``file already indexed``) and *failures*
    (``build raised``). The ``failed`` flag disambiguates so the
    pipeline can mark the file failed only on real errors, not skips.
    """
    index_name: str
    file_id: str
    output_dir: str
    item_count: int = 0
    skipped_reason: str | None = None
    failed: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


class IndexBuilder(ABC):
    """Append (file_id, pages) into one global index."""

    name: str

    @property
    def output_dir(self) -> Path:
        """Subclass returns the global on-disk directory for its store."""
        raise NotImplementedError

    def build(self, file_id: str, pages: List[PageAsset]) -> IndexBuildResult:
        return self._build(file_id, pages)

    @abstractmethod
    def _build(self, file_id: str, pages: List[PageAsset]) -> IndexBuildResult: ...
