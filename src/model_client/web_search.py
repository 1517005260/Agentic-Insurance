"""Tavily Search client.

Powers the chat web mode + the web agent. The client is intentionally
tiny — Tavily's REST endpoint is one POST that accepts a JSON body and
returns ranked results plus an optional LLM-style ``answer`` field.

Two methods:

* :meth:`TavilyClient.search` — raw search; returns the parsed JSON.
* :meth:`TavilyClient.search_results` — convenience wrapper that returns
  a list of :class:`SearchResult` dataclasses (the shape the runners want).

The client is **fail-soft on missing key**: ``available()`` returns False
so callers can short-circuit without raising at import time. Construction
itself never raises; only :meth:`search` does, and only when the network
or auth fails.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

from config.http import make_retry_session
from config.shared import shared_session
from config.settings import TAVILY_API_BASE_URL, TAVILY_API_KEY


_SEARCH_PATH = "/search"


@dataclass
class SearchResult:
    """One Tavily hit — flattened for downstream rendering / LLM ingestion."""

    title: str
    url: str
    content: str
    score: float
    published_date: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class TavilyClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or TAVILY_API_KEY
        self.base_url = (base_url or TAVILY_API_BASE_URL).rstrip("/")
        self.timeout = timeout
        # Default ``make_retry_session()`` uses total=5/read=5 — fine for
        # cheap embedding/rerank calls but catastrophic for Tavily: each
        # 30 s read timeout gets retried 5×, turning a slow first call
        # into a ~150 s wall-clock stall on a cold-start chat web turn.
        # Tavily's own SLA is "1-3 s typical,
        # 30 s outlier"; one retry is plenty. Connection retries stay at
        # 2 in case a TLS handshake races a transient DNS hiccup.
        # Process-wide shared session — distinct profile from the other
        # clients because the retry policy is tighter.
        self._session = shared_session(
            "tavily-tight", lambda: make_retry_session(total=2, read_retries=1)
        )

    def available(self) -> bool:
        return bool(self.api_key)

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: Literal["basic", "advanced"] = "basic",
        include_answer: bool = False,
        include_domains: Optional[Sequence[str]] = None,
        exclude_domains: Optional[Sequence[str]] = None,
        topic: Literal["general", "news"] = "general",
        days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Raw Tavily call. Returns the parsed JSON body verbatim.

        ``days`` is honored only when ``topic="news"``; ignored otherwise.
        """
        if not self.available():
            raise RuntimeError(
                "TavilyClient.search called without TAVILY_API_KEY. "
                "Set the env var or check available() first."
            )

        payload: Dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "topic": topic,
        }
        if include_domains:
            payload["include_domains"] = list(include_domains)
        if exclude_domains:
            payload["exclude_domains"] = list(exclude_domains)
        if topic == "news" and days is not None:
            payload["days"] = days

        url = f"{self.base_url}{_SEARCH_PATH}"
        response = self._session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def search_results(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Convenience wrapper — returns flattened :class:`SearchResult` list.

        ``**kwargs`` are forwarded to :meth:`search` unchanged.
        """
        body = self.search(query, **kwargs)
        items: List[SearchResult] = []
        for r in body.get("results", []) or []:
            items.append(
                SearchResult(
                    title=r.get("title", "") or "",
                    url=r.get("url", "") or "",
                    content=r.get("content", "") or "",
                    score=float(r.get("score", 0.0) or 0.0),
                    published_date=r.get("published_date"),
                    raw=r,
                )
            )
        return items
