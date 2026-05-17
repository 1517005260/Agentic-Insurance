"""Page-image embedding client — DashScope multimodal embedding (native API).

DashScope's multimodal embedding endpoint does NOT speak the OpenAI
``/v1/embeddings`` shape. It lives at the native path

    POST {base_url}/services/embeddings/multimodal-embedding/multimodal-embedding

with its own request/response shape::

    {
      "model": "qwen3-vl-embedding",
      "input":  {"contents": [{"image": "data:image/jpeg;base64,..."}, ...]},
      "parameters": {}
    }

    -> {"output": {"embeddings": [{"index": 0, "embedding": [...]}, ...]}, ...}

Use ``base_url=https://dashscope.aliyuncs.com/api/v1`` (Beijing) or
``…-intl.aliyuncs.com/api/v1`` (Singapore). DO NOT use the
``compatible-mode/v1`` URL — that path is text-embedding only.
"""
import base64
import logging
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np

from functools import lru_cache

from config.http import make_retry_session
from config.shared import shared_session
from config.settings import (
    VISUAL_EMBEDDING_API_BASE_URL,
    VISUAL_EMBEDDING_API_KEY,
    VISUAL_EMBEDDING_BACKEND,
    VISUAL_EMBEDDING_MODEL,
)

if TYPE_CHECKING:
    from model_client.qwen_vl_embedding import QwenVLEmbeddingClient

logger = logging.getLogger(__name__)

_DASHSCOPE_PATH = "/services/embeddings/multimodal-embedding/multimodal-embedding"


class VisualEmbeddingClient:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 5,
        timeout: float = 180.0,
    ):
        self.model = model or VISUAL_EMBEDDING_MODEL
        self.api_key = api_key or VISUAL_EMBEDDING_API_KEY
        self.base_url = (base_url or VISUAL_EMBEDDING_API_BASE_URL).rstrip("/")
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self._session = shared_session(
            "visual-embedding-default", lambda: make_retry_session()
        )

    def available(self) -> bool:
        return bool(self.model and self.api_key)

    def encode_paths(self, image_paths: Sequence[Union[str, Path]]) -> np.ndarray:
        """Embed every image path. Returns (N, D), L2-normalized."""
        if not image_paths:
            return np.zeros((0, 0), dtype=np.float32)
        contents = [{"image": self._image_to_data_uri(Path(p))} for p in image_paths]
        return self._encode_contents(contents)

    def encode_text(self, text: str) -> np.ndarray:
        """Embed a single text string in the SHARED multimodal space.

        qwen3-vl-embedding (and tongyi-embedding-vision-*) accept ``{"text":
        ...}`` items in the same ``input.contents`` payload as ``{"image":
        ...}`` items; the resulting vector lives in the same space as image
        vectors, so cosine similarity between a text query and a page-image
        embedding is meaningful (cross-modal retrieval).
        """
        return self._encode_contents([{"text": text}])[0]

    # ---------------------------------------------------------- internals

    def _encode_contents(self, contents: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Send a flat list of content items to the multimodal endpoint.

        Each item (``{"image": ...}`` or ``{"text": ...}``) is one
        independent input; the response returns one embedding per item,
        aligned by the ``index`` field.
        """
        if not self.available():
            raise RuntimeError(
                "VisualEmbeddingClient is not configured (set "
                "VISUAL_EMBEDDING_API_KEY and VISUAL_EMBEDDING_MODEL)."
            )
        if "compatible-mode" in self.base_url:
            raise RuntimeError(
                f"VISUAL_EMBEDDING_API_BASE_URL is set to '{self.base_url}', "
                f"which is the OpenAI-compatible URL. DashScope's multimodal "
                f"embedding API only works on the native '/api/v1' path. Edit "
                f".env so the URL ends in '/api/v1'."
            )

        url = f"{self.base_url}{_DASHSCOPE_PATH}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        all_vectors: List[List[float]] = []
        for start in range(0, len(contents), self.batch_size):
            batch = list(contents[start : start + self.batch_size])
            payload: Dict[str, Any] = {
                "model": self.model,
                "input": {"contents": batch},
                "parameters": {},
            }
            response = self._session.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            embeddings = self._parse_embeddings(response.json())
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"DashScope returned {len(embeddings)} vectors for "
                    f"{len(batch)} inputs (model={self.model})."
                )
            all_vectors.extend(embeddings)

        arr = np.asarray(all_vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _image_to_data_uri(path: Path) -> str:
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "image/jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _parse_embeddings(body: Dict[str, Any]) -> List[List[float]]:
        """Pull per-input vectors out of a DashScope multimodal response.

        Shape::

            {"output": {"embeddings": [{"index": 0, "embedding": [...]}, ...]},
             "usage":  {...},
             "request_id": "..."}
        """
        output = body.get("output") or {}
        items = output.get("embeddings") or output.get("contents") or []
        if not items:
            raise RuntimeError(
                f"DashScope response missing 'output.embeddings'; got: {body!r}"
            )
        items = sorted(items, key=lambda d: d.get("index", 0))
        return [item["embedding"] for item in items]


@lru_cache(maxsize=1)
def get_cached_visual_embedding_client() -> "Union[VisualEmbeddingClient, QwenVLEmbeddingClient]":
    """Process-wide singleton page-image embedder, backend-selected.

    Every visual-embedding callsite (vision_dense builder, the image
    retrieval channel) routes through here, so this is the single
    chokepoint deciding API vs local: ``VISUAL_EMBEDDING_BACKEND=local``
    hands out the GPU Qwen3-VL-Embedding client, anything else the
    DashScope HTTP client. Both expose the identical
    ``encode_paths`` / ``encode_text`` / ``available`` / ``model``
    surface, so the choice is invisible downstream. Cleared by
    ``config.shared.clear_caches`` on a backend/model swap.
    """
    if VISUAL_EMBEDDING_BACKEND == "local":
        # Lazy: importing the local client pulls torch/transformers,
        # which API-only deployments must not pay for.
        from model_client.qwen_vl_embedding import QwenVLEmbeddingClient

        return QwenVLEmbeddingClient()
    return VisualEmbeddingClient()
