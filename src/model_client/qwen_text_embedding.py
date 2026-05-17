"""Local text-embedding client backed by Qwen3-Embedding-0.6B.

Drop-in alternative to :class:`model_client.text_embedding.EmbeddingClient`:
the public surface is exactly ``encode(str | Sequence[str]) -> np.ndarray``
returning **L2-normalized float32** vectors (1-D for a single string, 2-D
for a sequence), so every existing callsite — ingest builders, RAG
channels, agent tools — works unchanged once
``get_cached_embedding_client`` hands this out instead of the HTTP client.

Qwen3-Embedding is a decoder model: the sentence vector is the hidden
state of the final (EOS) token. The shared loader sets
``padding_side="left"`` so that token sits at index ``-1`` for every row
in a batch, making pooling a single slice with no attention-mask
gather. Weights are a flat local snapshot under
``settings.embed_model_dir()`` (``STORAGE_PATH/models/Qwen3-Embedding-0.6B``),
pre-fetched by ``python download_models.py`` — same storage discipline
as the reranker so both models move with the storage volume.

GPU FP16 is mandatory (enforced in ``shared_qwen_embedding``); deployments
without CUDA should keep ``EMBEDDING_BACKEND=api``.
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
import torch

from config.settings import EMBED_MODEL_ID, embed_model_dir
from config.shared import shared_qwen_embedding


class QwenEmbeddingClient:
    """Last-token-pooled sentence embedder over a local Qwen3-Embedding.

    Thread-safe: the cached (tokenizer, model) handle is shared under
    PyTorch's read-only ``eval()`` forward; per-instance state is
    immutable after ``__init__``. ``model_dir`` / ``batch_size`` /
    ``max_length`` are injectable for tests and future admin-config
    wiring; production callers use the no-arg form via the factory.
    """

    def __init__(
        self,
        model_dir: Optional[Any] = None,
        batch_size: int = 64,
        max_length: int = 1024,
    ):
        resolved_dir = model_dir if model_dir is not None else embed_model_dir()
        # ``model`` mirrors EmbeddingClient.model: the model identifier
        # string that ingest builders record as index metadata
        # (text_dense / graph_linearrag read ``embedding_client.model``).
        self.model = EMBED_MODEL_ID
        self.model_dir = resolved_dir
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self._tokenizer, self._model = shared_qwen_embedding(resolved_dir)

    @torch.no_grad()
    def encode(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        """Embed a string (1-D) or list of strings (2-D). L2-normalized.

        Mirrors ``EmbeddingClient.encode`` byte-for-byte in shape and
        normalization so the two backends are interchangeable behind
        the cached factory.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        else:
            texts = list(texts)

        device = self._model.device
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)
            # Left padding ⇒ the final real token of every row is at -1.
            hidden = self._model(**inputs).last_hidden_state[:, -1]
            hidden = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
            chunks.append(hidden.cpu().numpy())

        arr = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        if single:
            return arr[0]
        return arr
