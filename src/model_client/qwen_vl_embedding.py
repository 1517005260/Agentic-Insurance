"""Local multimodal embedding client backed by Qwen3-VL-Embedding-2B.

Drop-in alternative to
:class:`model_client.visual_embedding.VisualEmbeddingClient`: same public
surface (``encode_paths`` → ``(N, D)`` L2-normalized float32,
``encode_text`` → ``(D,)`` single vector in the *same* space,
``available()``, ``model``), so ``get_cached_visual_embedding_client``
can hand this out instead of the DashScope HTTP client with nothing
downstream changing (``vision_dense`` builder, the image retrieval
channel).

Qwen3-VL-Embedding-2B puts page images and text into one shared 2048-d
space — exactly the cross-modal property the image channel relies on.
The model snapshot ships its own ``scripts/qwen3_vl_embedding.py``
(``Qwen3VLEmbedder``) which does last-token pooling + L2-normalization
internally; the shared loader (``config.shared.shared_qwen_vl_embedding``)
materialises it once on GPU FP16. Weights are a flat local snapshot
under ``settings.vl_embed_model_dir()`` — same storage discipline as the
reranker / text-embedding models.

GPU FP16 is mandatory (enforced in ``shared_qwen_vl_embedding``);
deployments without CUDA keep ``VISUAL_EMBEDDING_BACKEND=api``.
"""

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from config.settings import VL_EMBED_MODEL_ID, vl_embed_model_dir
from config.shared import shared_qwen_vl_embedding

# Query-side instruction (image side is embedded raw). Kept here as a
# default; injectable per call-site / future admin config.
_DEFAULT_TEXT_INSTRUCTION = "Retrieve relevant documents for the query."


class QwenVLEmbeddingClient:
    """Multimodal embedder over a local Qwen3-VL-Embedding-2B.

    Thread-safe: the cached embedder is shared under PyTorch's
    read-only ``eval()`` forward; per-instance state is immutable after
    ``__init__``. ``model_dir`` / ``batch_size`` / ``text_instruction``
    are injectable for tests and future admin-config wiring.
    """

    def __init__(
        self,
        model_dir: Optional[Any] = None,
        batch_size: int = 8,
        text_instruction: str = _DEFAULT_TEXT_INSTRUCTION,
    ):
        resolved_dir = model_dir if model_dir is not None else vl_embed_model_dir()
        # ``model`` mirrors VisualEmbeddingClient.model: the identifier
        # string recorded as index metadata (vision_dense reads it).
        self.model = VL_EMBED_MODEL_ID
        self.model_dir = resolved_dir
        self.batch_size = max(1, int(batch_size))
        self.text_instruction = text_instruction
        self._embedder = shared_qwen_vl_embedding(Path(str(resolved_dir)))

    def available(self) -> bool:
        # Construction loads the model (raises on missing weights / no
        # CUDA), so a constructed instance is always usable — kept as a
        # method for interface parity with VisualEmbeddingClient.
        return True

    def _to_np(self, vectors: Any) -> np.ndarray:
        """``Qwen3VLEmbedder.process`` → (N, D) L2-normalized float32."""
        arr = vectors.float().cpu().numpy()
        return np.ascontiguousarray(arr, dtype=np.float32)

    def encode_paths(
        self, image_paths: Sequence[Union[str, Path]]
    ) -> np.ndarray:
        """Embed every image path. Returns (N, D), L2-normalized."""
        if not image_paths:
            return np.zeros((0, 0), dtype=np.float32)
        out = []
        items = [{"image": str(Path(p).resolve())} for p in image_paths]
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            out.append(self._to_np(self._embedder.process(batch)))
        return np.concatenate(out, axis=0)

    def encode_text(self, text: str) -> np.ndarray:
        """Embed one text string into the SHARED multimodal space."""
        vec = self._embedder.process(
            [{"text": text, "instruction": self.text_instruction}]
        )
        return self._to_np(vec)[0]
