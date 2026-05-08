"""Tavily-backed web search.

Companion to :class:`WebFetchTool`. The agent calls ``web_search``
to discover candidate URLs, then ``web_fetch`` to pull the full text
of the most promising hit for verbatim cite. Search alone is not
enough — Tavily's snippet field is ~300 chars, which is fine for
triage but loses the surrounding context most cites need.

The tool is configured by the optional :class:`ConfigStore` passed
in at construction time; admin-managed defaults
(``tavily.max_results``, ``tavily.search_depth``) flow through that
single dependency without re-reading env vars.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agentic.tools.acquisition._common import err, ok
from agentic.tools.base import BaseTool
from model_client.web_search import TavilyClient

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from config.config_store import ConfigStore


logger = logging.getLogger(__name__)


_DEFAULT_MAX_RESULTS = 5
_MAX_RESULTS_CAP = 20


class WebSearchTool(BaseTool):
    def __init__(
        self,
        tavily_client: TavilyClient,
        config_store: Optional["ConfigStore"] = None,
    ):
        self._client = tavily_client
        # The store is only consulted when the agent omits per-call
        # overrides. Reads happen on the worker thread, so a concurrent
        # admin PATCH could land between two tool calls in the same
        # run. The web agent already runs with snapshotted ``max_loops``
        # / ``max_token_budget`` / ``system_prompt`` (the workbench
        # contract); the tavily defaults are advisory only and cheap to
        # tolerate small-window drift on, so we keep the live read.
        self._config = config_store

    @property
    def name(self) -> str:
        return "web_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the public web via Tavily. Returns up to "
                    "`max_results` ranked hits with title, URL, snippet "
                    "(~300 chars), score, optional published_date.\n\n"
                    "Workflow: call web_search to discover candidates, "
                    "then call web_fetch on the URL whose snippet best "
                    "addresses the query to read the full page before "
                    "citing.\n\n"
                    "This is the only way to bring information from "
                    "outside the local corpus into the answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": (
                                f"Number of hits (1-{_MAX_RESULTS_CAP}); "
                                f"default {_DEFAULT_MAX_RESULTS}."
                            ),
                        },
                        "search_depth": {
                            "type": "string",
                            "enum": ["basic", "advanced"],
                            "description": (
                                "`advanced` triggers a deeper crawl "
                                "(slower, costs more credits). Default "
                                "`basic`."
                            ),
                        },
                        "include_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Whitelist; only return hits from these "
                                "domains. Use for jurisdiction-scoped "
                                "compliance queries (e.g. ['ia.org.hk', "
                                "'hkma.gov.hk'] for HK insurance regulation)."
                            ),
                        },
                        "exclude_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Blacklist filter.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        query: str,
        max_results: Optional[int] = None,
        search_depth: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ):
        if not query or not str(query).strip():
            return err(
                "invalid_argument",
                "`query` must be a non-empty string.",
                remediation="Pass a search-engine-style query as `query`.",
                valid_example={"query": "Hong Kong MPF regulation 2025"},
            ), {"error": "invalid_argument"}

        if not self._client.available():
            return err(
                "unavailable",
                "Tavily client not configured (missing TAVILY_API_KEY).",
                remediation="Operator must set TAVILY_API_KEY in the env.",
            ), {"error": "unavailable"}

        # Defaults from admin config when caller omits the argument.
        if max_results is None:
            max_results = self._cfg_int("tavily.max_results", _DEFAULT_MAX_RESULTS)
        try:
            max_results = max(1, min(int(max_results), _MAX_RESULTS_CAP))
        except (TypeError, ValueError):
            max_results = _DEFAULT_MAX_RESULTS

        if search_depth is None:
            search_depth = self._cfg_str("tavily.search_depth", "basic")
        if search_depth not in ("basic", "advanced"):
            search_depth = "basic"

        try:
            results = self._client.search_results(
                query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains or None,
                exclude_domains=exclude_domains or None,
            )
        except Exception as exc:
            logger.warning("web_search failed query=%r: %r", query, exc)
            return err(
                "fetch_error",
                f"Tavily search failed: {type(exc).__name__}: {exc}",
                remediation="Retry once; if it persists, refine the query or check connectivity.",
            ), {"error": "fetch_error"}

        hits = [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.content,
                "score": round(float(r.score), 4),
                "published_date": r.published_date,
            }
            for r in results
        ]
        # Rough token estimate so token-budget tracking sees a non-zero
        # cost. 4 chars ≈ 1 token is OpenAI's published heuristic; good
        # enough for accounting.
        approx_tokens = sum(len(h["snippet"] or "") for h in hits) // 4
        context.add_retrieval_log(
            tool_name="web_search",
            tokens=approx_tokens,
            metadata={
                "query": query,
                "n_results": len(hits),
                "search_depth": search_depth,
                "include_domains": include_domains,
            },
        )
        return (
            ok(
                "WebSearchObservation",
                query=query,
                n_results=len(hits),
                results=hits,
            ),
            {
                "retrieved_tokens": approx_tokens,
                "n_results": len(hits),
            },
        )

    # ---------- config plumbing ----------

    def _cfg_int(self, key: str, default: int) -> int:
        if self._config is None:
            return default
        try:
            v = self._config.get(key)
            return int(v) if v is not None else default
        except Exception:
            return default

    def _cfg_str(self, key: str, default: str) -> str:
        if self._config is None:
            return default
        try:
            v = self._config.get(key)
            return str(v) if v is not None else default
        except Exception:
            return default
