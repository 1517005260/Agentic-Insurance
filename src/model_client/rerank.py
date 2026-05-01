"""Reranker client — DashScope text-rerank (native API).

The endpoint sits at

    POST {base_url}/services/rerank/text-rerank/text-rerank

All DashScope rerank models use the *nested* body shape
``{"model", "input": {...}, "parameters": {...}}``. Two flavors:

* **Plain-text rerankers** (``qwen3-rerank``, ``gte-rerank-v2``, …) —
  query/documents are bare strings::

    {"model": "qwen3-rerank",
     "input": {"query": str, "documents": [str, ...]},
     "parameters": {"top_n", "return_documents"}}

* **Multimodal reranker** (``qwen3-vl-rerank``) — each item carries a
  content-type tag::

    {"model": "qwen3-vl-rerank",
     "input": {"query": {"text": str},
               "documents": [{"text": str} | {"image": url} | {"video": url}, ...]},
     "parameters": {"top_n", "return_documents"}}

The response is uniform: ``output.results`` is a list of
``{"index", "relevance_score"}`` tuples; ``relevance_score`` is only
comparable WITHIN a single request — never across calls.
"""

from typing import Any, Dict, List, Optional, Sequence

import requests

from config.settings import RERANKER_API_BASE_URL, RERANKER_API_KEY, RERANKER_MODEL


_DASHSCOPE_PATH = "/services/rerank/text-rerank/text-rerank"


class RerankResult(dict):
    """{"index": int, "relevance_score": float}; dict for cheap json log."""


class RerankClient:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.model = model or RERANKER_MODEL
        self.api_key = api_key or RERANKER_API_KEY
        self.base_url = (base_url or RERANKER_API_BASE_URL).rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.model and self.api_key)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> List[RerankResult]:
        """Return top-``top_n`` results sorted by relevance_score desc."""
        if not self.available():
            raise RuntimeError(
                "RerankClient is not configured (set RERANKER_API_KEY and "
                "RERANKER_MODEL)."
            )
        if not documents:
            return []
        if "compatible-mode" in self.base_url:
            raise RuntimeError(
                f"RERANKER_API_BASE_URL is '{self.base_url}', the OpenAI-compatible "
                f"path. DashScope rerank only works on the native '/api/v1' path."
            )

        url = (
            self.base_url + _DASHSCOPE_PATH
            if not self.base_url.endswith(_DASHSCOPE_PATH)
            else self.base_url
        )
        capped_top_n = int(min(top_n, len(documents)))
        payload = self._build_payload(query, list(documents), capped_top_n)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if response.status_code >= 400:
            # Surface the server's error body — DashScope returns useful
            # diagnostics (invalid_parameter, bad model name, …) that
            # raise_for_status() would otherwise hide.
            raise RuntimeError(
                f"DashScope rerank {response.status_code}: {response.text!r} "
                f"(payload model={self.model!r})"
            )
        body = response.json()
        results = (body.get("output") or {}).get("results")
        if results is None:
            raise RuntimeError(f"DashScope rerank response missing output.results: {body!r}")
        return [
            RerankResult(index=int(r["index"]), relevance_score=float(r["relevance_score"]))
            for r in results
        ]

    def _build_payload(
        self, query: str, documents: List[str], top_n: int
    ) -> Dict[str, Any]:
        # qwen3-vl-rerank is the only model that needs content-type tags
        # because it can mix text/image/video items in one request.
        if self.model == "qwen3-vl-rerank":
            return {
                "model": self.model,
                "input": {
                    "query": {"text": query},
                    "documents": [{"text": d} for d in documents],
                },
                "parameters": {"top_n": top_n, "return_documents": False},
            }
        # All plain-text rerankers (qwen3-rerank, gte-rerank-v2, …) take
        # the same nested body with bare-string query/documents.
        return {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_n, "return_documents": False},
        }
