"""Text embedding client over an OpenAI-compatible /embeddings endpoint.

Returns L2-normalized float32 vectors so dot product equals cosine similarity
downstream. Batches requests to avoid hitting per-request size limits.
"""

from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np

from config.http import make_retry_session
from config.shared import shared_session
from config.settings import (
    EMBEDDING_API_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
)

if TYPE_CHECKING:
    from model_client.qwen_text_embedding import QwenEmbeddingClient


class EmbeddingClient:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 64,
        timeout: float = 120.0,
    ):
        self.model = model or EMBEDDING_MODEL or "text-embedding-3-small"
        self.api_key = api_key or EMBEDDING_API_KEY
        self.base_url = (base_url or EMBEDDING_API_BASE_URL).rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "Embedding API key required. Set EMBEDDING_API_KEY in .env or pass api_key."
            )

        # Shared urllib3 connection pool keyed by retry profile so this
        # client doesn't get its own ~30 MB session on top of the LLM /
        # rerank / visual / Tavily clients. ``shared_session`` is
        # documented thread-safe.
        self._session = shared_session(
            "embedding-default", lambda: make_retry_session()
        )

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        *,
        is_query: bool = False,
    ) -> np.ndarray:
        """Embed a string (1-D) or list of strings (2-D). L2-normalized.

        ``is_query`` is accepted for backend-parity with
        :class:`QwenEmbeddingClient` but **ignored** by the HTTP path:
        the OpenAI ``/embeddings`` schema has no instruction field, and
        the only deployments that expose one (DashScope's ``input_type``
        family) do so under non-standard keys we don't auto-detect.
        Switch to ``EMBEDDING_BACKEND=local`` to get query-prefix
        semantics on Qwen3-Embedding.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        all_vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            payload = {"model": self.model, "input": batch}
            response = self._session.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json().get("data", [])
            data.sort(key=lambda d: d.get("index", 0))
            all_vectors.extend(item["embedding"] for item in data)

        arr = np.asarray(all_vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms

        if single:
            return arr[0]
        return arr


@lru_cache(maxsize=1)
def get_cached_embedding_client() -> "Union[EmbeddingClient, QwenEmbeddingClient]":
    """Process-wide singleton text embedder, backend-selected.

    Every embedding callsite (lifespan, ingest builders, RAG channels,
    agent factory/tools) routes through here, so this is the single
    chokepoint that decides API vs local: ``EMBEDDING_BACKEND=local``
    hands out the GPU Qwen3-Embedding client, anything else the
    OpenAI-compatible HTTP client. Both expose the identical
    ``encode(str | Sequence[str]) -> np.ndarray`` (L2-normalized
    float32) contract, so the choice is invisible downstream. Each
    warmed instance is heavy (HTTP session ~30 MB / GPU handle); the
    singleton avoids the ~6× duplication. Cleared by
    ``config.shared.clear_caches`` on a backend/model swap.
    """
    if EMBEDDING_BACKEND == "local":
        # Lazy: importing the local client pulls torch/transformers,
        # which API-only deployments must not pay for.
        from model_client.qwen_text_embedding import QwenEmbeddingClient

        return QwenEmbeddingClient()
    return EmbeddingClient()
