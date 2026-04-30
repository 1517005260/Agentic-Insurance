"""Per-run agent state: retrieval cost log and read-page bookkeeping."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class RetrievalLog:
    tool_name: str
    tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentContext:
    """Tracks token cost and which pages have been read this run.

    Read deduplication is page-level so the model is not handed the same
    Markdown blob twice within a single trajectory.
    """

    def __init__(self):
        self.total_retrieved_tokens: int = 0
        self.retrieval_logs: List[RetrievalLog] = []

        self.read_page_ids: Set[str] = set()
        self.search_history: List[Dict[str, Any]] = []

    def add_retrieval_log(
        self,
        tool_name: str,
        tokens: int,
        metadata: Dict[str, Any] = None,
    ):
        log = RetrievalLog(
            tool_name=tool_name,
            tokens=tokens,
            metadata=metadata or {},
        )
        self.retrieval_logs.append(log)
        self.total_retrieved_tokens += tokens

    def mark_page_as_read(self, page_id: str):
        self.read_page_ids.add(str(page_id))

    def is_page_read(self, page_id: str) -> bool:
        return str(page_id) in self.read_page_ids

    def reset(self):
        self.retrieval_logs = []
        self.read_page_ids = set()
        self.search_history = []
        self.total_retrieved_tokens = 0

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_retrieved_tokens": self.total_retrieved_tokens,
            "retrieval_logs": [
                {
                    "tool_name": log.tool_name,
                    "tokens": log.tokens,
                    "metadata": log.metadata,
                }
                for log in self.retrieval_logs
            ],
            "pages_read_count": len(self.read_page_ids),
            "pages_read_ids": list(self.read_page_ids),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.get_summary()
