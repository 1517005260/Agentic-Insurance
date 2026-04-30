"""HTTP clients for OpenAI-compatible chat, text-embedding and visual-embedding endpoints.

All three are thin wrappers: they read endpoint + model + key from
`config.settings`, send the request via `requests`, and return either a
parsed completion (chat) or an L2-normalized float32 vector (embeddings).
No runtime model code lives here.
"""

from model_client.chat import LLMClient
from model_client.text_embedding import EmbeddingClient
from model_client.visual_embedding import VisualEmbeddingClient

__all__ = ["LLMClient", "EmbeddingClient", "VisualEmbeddingClient"]
