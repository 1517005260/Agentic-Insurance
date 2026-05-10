"""HTTP clients for OpenAI-compatible chat, text-embedding and visual-embedding endpoints.

All three are thin wrappers: they read endpoint + model + key from
`config.settings`, send the request via `requests`, and return either a
parsed completion (chat) or an L2-normalized float32 vector (embeddings).
No runtime model code lives here.
"""

from model_client.chat import LLMClient, StreamProtocolError, get_cached_client
from model_client.rerank import RerankClient, get_cached_rerank_client
from model_client.text_embedding import EmbeddingClient, get_cached_embedding_client
from model_client.visual_embedding import (
    VisualEmbeddingClient,
    get_cached_visual_embedding_client,
)
from model_client.web_search import SearchResult, TavilyClient

__all__ = [
    "LLMClient",
    "StreamProtocolError",
    "EmbeddingClient",
    "VisualEmbeddingClient",
    "RerankClient",
    "TavilyClient",
    "SearchResult",
    # Cached factories — prefer these over no-arg constructors so
    # ingest builders / RAG channels / agent factories share a single
    # instance with lifespan instead of each holding their own.
    "get_cached_client",
    "get_cached_embedding_client",
    "get_cached_visual_embedding_client",
    "get_cached_rerank_client",
]
