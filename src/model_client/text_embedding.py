"""Text embedding client over an OpenAI-compatible /embeddings endpoint.

Returns L2-normalized float32 vectors so dot product equals cosine similarity
downstream. Batches requests to avoid hitting per-request size limits.
"""

from typing import List, Optional, Sequence, Union

import numpy as np

from config.http import make_retry_session
from config.settings import (
    EMBEDDING_API_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL,
)


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

        self._session = make_retry_session()

    def encode(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        """Embed a string (1-D) or list of strings (2-D). L2-normalized."""
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
