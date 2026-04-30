"""Single source of truth for environment variables and shared constants.

`.env` is loaded once on import; afterwards every module reads from this
module instead of touching `os.environ` directly. This keeps the env surface
documented in one place and avoids the per-module `getenv` sprinkle.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key, default)
    if value is None or value == "":
        return None
    return value


# ---------------------------------------------------------------- storage ----

STORAGE_PATH: Path = Path(_get("STORAGE_PATH", "./local_storage") or "./local_storage")

# Subdirectory under STORAGE_PATH that holds raw PaddleOCR outputs.
PADDLE_OCR_SUBDIR: str = "paddle_ocr"


def paddle_ocr_root() -> Path:
    """Return the root directory for PaddleOCR outputs."""
    return STORAGE_PATH / PADDLE_OCR_SUBDIR


# Subdirectory under STORAGE_PATH that holds downloaded model weights
# (currently the spaCy NER model used by the entity layer).
MODELS_SUBDIR: str = "models"


def models_root() -> Path:
    """Return the root directory for downloaded model weights."""
    return STORAGE_PATH / MODELS_SUBDIR


# Subdirectory holding the canonical PageAsset JSON for each ingested file.
# One file per ingested document: <STORAGE_PATH>/page_assets/<file_id>.json.
PAGE_ASSETS_SUBDIR: str = "page_assets"


def page_assets_root() -> Path:
    return STORAGE_PATH / PAGE_ASSETS_SUBDIR


def page_assets_path(file_id: str) -> Path:
    return page_assets_root() / f"{file_id}.json"


# All faiss-backed embedding stores live under one global root. Stores are
# global (not per-file): new files append into the same index; the meta
# table carries `file_id` so per-file filtering is still cheap. This makes
# cross-file entity alignment and incremental graph growth natural.
FAISS_SUBDIR: str = "faiss"


def faiss_root() -> Path:
    return STORAGE_PATH / FAISS_SUBDIR


def faiss_dense_dir() -> Path:
    """Sentence-level text embeddings (semantic_search source)."""
    return faiss_root() / "dense"


def faiss_visual_dir() -> Path:
    """Page-image embeddings (visual channel of semantic_search)."""
    return faiss_root() / "visual"


def faiss_graph_dir() -> Path:
    """LinearRAG embedding stores live here, three sub-namespaces."""
    return faiss_root() / "graph"


def faiss_graph_passage_dir() -> Path:
    return faiss_graph_dir() / "passage"


def faiss_graph_entity_dir() -> Path:
    return faiss_graph_dir() / "entity"


def faiss_graph_sentence_dir() -> Path:
    return faiss_graph_dir() / "sentence"


# BM25 (tantivy) is also a global cross-file index — same model as faiss
# stores, just a different backend. Documents carry `file_id` for filtering.
BM25_SUBDIR: str = "bm25"


def bm25_root() -> Path:
    return STORAGE_PATH / BM25_SUBDIR


# ----------------------------------------------------------- paddle ocr ----

PADDLE_OCR_API_URL: str | None = _get("API_URL")
PADDLE_OCR_TOKEN: str | None = _get("TOKEN")

# Hard cap from the PaddleOCR layout-parsing API. Files larger than this are
# split into batches before submission.
PADDLE_OCR_MAX_PAGES_PER_BATCH: int = 50

# Inserted between adjacent batches in the concatenated Markdown so downstream
# page builders can detect batch boundaries.
PADDLE_OCR_BATCH_SEPARATOR: str = "\n\n<!-- agentic:batch_boundary -->\n\n"

# PaddleOCR fileType values.
PADDLE_OCR_FILE_TYPE_PDF: int = 0
PADDLE_OCR_FILE_TYPE_IMAGE: int = 1


# ----------------------------------------------------------------- chat ----

CHAT_API_KEY: str | None = _get("CHAT_API_KEY")
CHAT_API_BASE_URL: str = _get("CHAT_API_BASE_URL") or "https://api.openai.com/v1"
CHAT_MODEL: str | None = _get("CHAT_MODEL")


# ------------------------------------------------------------------ vlm ----

VLM_API_KEY: str | None = _get("VLM_API_KEY")
VLM_API_BASE_URL: str = _get("VLM_API_BASE_URL") or "https://api.openai.com/v1"
VLM_MODEL: str | None = _get("VLM_MODEL")


# ------------------------------------------------------------ embeddings ----

EMBEDDING_API_KEY: str | None = _get("EMBEDDING_API_KEY")
EMBEDDING_API_BASE_URL: str = _get("EMBEDDING_API_BASE_URL") or "https://api.openai.com/v1"
EMBEDDING_MODEL: str | None = _get("EMBEDDING_MODEL")

VISUAL_EMBEDDING_API_KEY: str | None = _get("VISUAL_EMBEDDING_API_KEY")
VISUAL_EMBEDDING_API_BASE_URL: str = (
    _get("VISUAL_EMBEDDING_API_BASE_URL") or "https://api.openai.com/v1"
)
VISUAL_EMBEDDING_MODEL: str | None = _get("VISUAL_EMBEDDING_MODEL")


__all__ = [
    "STORAGE_PATH",
    "PADDLE_OCR_SUBDIR",
    "paddle_ocr_root",
    "MODELS_SUBDIR",
    "models_root",
    "PAGE_ASSETS_SUBDIR",
    "page_assets_root",
    "page_assets_path",
    "FAISS_SUBDIR",
    "faiss_root",
    "faiss_dense_dir",
    "faiss_visual_dir",
    "faiss_graph_dir",
    "faiss_graph_passage_dir",
    "faiss_graph_entity_dir",
    "faiss_graph_sentence_dir",
    "BM25_SUBDIR",
    "bm25_root",
    "PADDLE_OCR_API_URL",
    "PADDLE_OCR_TOKEN",
    "PADDLE_OCR_MAX_PAGES_PER_BATCH",
    "PADDLE_OCR_BATCH_SEPARATOR",
    "PADDLE_OCR_FILE_TYPE_PDF",
    "PADDLE_OCR_FILE_TYPE_IMAGE",
    "CHAT_API_KEY",
    "CHAT_API_BASE_URL",
    "CHAT_MODEL",
    "VLM_API_KEY",
    "VLM_API_BASE_URL",
    "VLM_MODEL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_API_BASE_URL",
    "EMBEDDING_MODEL",
    "VISUAL_EMBEDDING_API_KEY",
    "VISUAL_EMBEDDING_API_BASE_URL",
    "VISUAL_EMBEDDING_MODEL",
]
