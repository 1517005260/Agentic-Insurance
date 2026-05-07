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
# (e.g. the spaCy NER model used by the entity layer).
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


# Per-file structural inventory (sections derived from Markdown headings).
# One file per indexed document at <STORAGE_PATH>/inventory/<file_id>.json.
# Lazily built on first access by ``storage.InventoryStore``; re-ingest
# invalidates by overwrite.
INVENTORY_SUBDIR: str = "inventory"


def inventory_root() -> Path:
    return STORAGE_PATH / INVENTORY_SUBDIR


def inventory_path(file_id: str) -> Path:
    return inventory_root() / f"{file_id}.json"


# Sibling stores that share the inventory's per-file lazy-build pattern
# but operate at sub-page granularity:
# * passages   — paragraph / paragraph_title blocks from PaddleOCR's
#                ``parsing_res_list``
# * table_rows — individual ``<tr>`` rows extracted from each page's
#                rendered HTML tables
# Both live under <STORAGE_PATH>/inventory_atoms/<kind>/<file_id>.json.
INVENTORY_ATOMS_SUBDIR: str = "inventory_atoms"


def inventory_atoms_root(kind: str) -> Path:
    return STORAGE_PATH / INVENTORY_ATOMS_SUBDIR / kind


def passage_atoms_path(file_id: str) -> Path:
    return inventory_atoms_root("passages") / f"{file_id}.json"


def table_row_atoms_path(file_id: str) -> Path:
    return inventory_atoms_root("table_rows") / f"{file_id}.json"


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


# SQLite database for the FastAPI web app (users, sessions, messages,
# file records, system config). Kept under STORAGE_PATH so a single
# environment variable controls every persistent artifact in the system.
APP_DB_SUBDIR: str = "db"


def app_db_dir() -> Path:
    return STORAGE_PATH / APP_DB_SUBDIR


def app_db_path() -> Path:
    return app_db_dir() / "app.db"


# Originals of every uploaded document. Kept around so a re-ingest can
# replay PaddleOCR if its cache is wiped; safe to delete once the four
# downstream indexes are stable. One file per upload at
# <STORAGE_PATH>/uploads/<file_id><suffix>.
UPLOADS_SUBDIR: str = "uploads"


def uploads_root() -> Path:
    return STORAGE_PATH / UPLOADS_SUBDIR


def upload_path(file_id: str, suffix: str) -> Path:
    # Suffix is the original extension (".pdf", ".png", ...) so the file is
    # human-debuggable and PdfParser can detect type by extension.
    return uploads_root() / f"{file_id}{suffix}"


# ----------------------------------------------------------- tracer paths ----
# Tracer (src/tracer/) writes to ``STORAGE_PATH/<flavor>/<date>/<run_id>``.
# The web layer stores only the relative tail (``<flavor>/<date>/<run_id>``) in
# chat_messages.metadata_json so that switching ``STORAGE_PATH`` between
# environments doesn't require a DB rewrite.


def trace_run_path(rel: str) -> Path:
    """Resolve a stored relative trace path against ``STORAGE_PATH``.

    The web layer persists ``<flavor>/<date>/<run_id>`` (e.g.
    ``agentic/2026-05-06/202458_a1b2c3d4``); this returns the absolute
    on-disk path for endpoint handlers that need to read trace files.

    Refuses absolute inputs and any ``..`` traversal — the resolved
    path must stay strictly inside ``STORAGE_PATH``. The caller
    (today: row from ``chat_messages.metadata_json``) is server-written
    so this is defense-in-depth; once a trace endpoint accepts user
    input, it's the necessary boundary check.
    """
    candidate = (STORAGE_PATH / rel).resolve()
    base = STORAGE_PATH.resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(
            f"trace_run_path: {rel!r} resolves outside STORAGE_PATH"
        ) from exc
    return candidate


def relativize_trace_dir(absolute: Path) -> str:
    """Inverse of :func:`trace_run_path`; raises if ``absolute`` is outside STORAGE_PATH.

    Used by runners after ``Tracer.session()`` returns its absolute
    ``run_dir``. We refuse to silently store an absolute path because
    that would couple the DB to a deployment-specific filesystem layout.
    """
    return str(absolute.resolve().relative_to(STORAGE_PATH.resolve()))


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


# --------------------------------------------------------------- web app ----

# JWT signing config for the FastAPI auth layer. The secret MUST be
# overridden in production (.env). HS256 with an opaque secret keeps the
# deployment self-contained — no asymmetric key distribution to manage.
# ``_JWT_SECRET_PLACEHOLDER`` is the sentinel app/main.py uses to refuse
# startup when nobody set a real secret; ``ALLOW_INSECURE_JWT=1`` bypasses
# that check for local development. Keeping these centralized here means
# main.py never touches os.environ directly and never re-states the
# placeholder string.
_JWT_SECRET_PLACEHOLDER: str = "change-me-in-prod"
JWT_SECRET: str = _get("JWT_SECRET") or _JWT_SECRET_PLACEHOLDER
JWT_SECRET_IS_DEFAULT: bool = JWT_SECRET == _JWT_SECRET_PLACEHOLDER
ALLOW_INSECURE_JWT: bool = (_get("ALLOW_INSECURE_JWT") or "0") == "1"
JWT_ALGORITHM: str = _get("JWT_ALGORITHM") or "HS256"
JWT_EXP_MINUTES: int = int(_get("JWT_EXP_MINUTES") or "720")

# Bootstrap admin: created on first startup if no users exist. Idempotent —
# subsequent starts are a no-op when the user already exists. The
# ``IS_DEFAULT`` flag drives a startup warning so we don't sprinkle the
# placeholder string across modules.
_DEFAULT_ADMIN_PASSWORD_PLACEHOLDER: str = "admin123"
DEFAULT_ADMIN_USERNAME: str = _get("DEFAULT_ADMIN_USERNAME") or "admin"
DEFAULT_ADMIN_PASSWORD: str = _get("DEFAULT_ADMIN_PASSWORD") or _DEFAULT_ADMIN_PASSWORD_PLACEHOLDER
DEFAULT_ADMIN_PASSWORD_IS_DEFAULT: bool = (
    DEFAULT_ADMIN_PASSWORD == _DEFAULT_ADMIN_PASSWORD_PLACEHOLDER
)

# Comma-separated list. The Vite dev server defaults are baked in so the
# frontend works out of the box.
CORS_ORIGINS: list[str] = [
    o.strip()
    for o in (_get("CORS_ORIGINS") or "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]


# -------------------------------------------------------------- tavily ----

# Tavily Search powers the regulation-lookup workbench in the web app. The
# free tier is sufficient for the demo. Unset key => /insurance/regulation-
# search returns 503 and the UI shows a disabled state; nothing else
# depends on it.
TAVILY_API_KEY: str | None = _get("TAVILY_API_KEY")
TAVILY_API_BASE_URL: str = _get("TAVILY_API_BASE_URL") or "https://api.tavily.com"


# -------------------------------------------------------------- reranker ----

# DashScope native rerank API. Base URL ends at /api/v1; the client appends
# /services/rerank/text-rerank/text-rerank. Default model is gte-rerank-v2
# because it's available on both Beijing and Singapore endpoints; switch to
# qwen3-rerank with the -intl URL for higher quality where available.
RERANKER_API_KEY: str | None = _get("RERANKER_API_KEY")
RERANKER_API_BASE_URL: str = (
    _get("RERANKER_API_BASE_URL") or "https://dashscope.aliyuncs.com/api/v1"
)
RERANKER_MODEL: str = _get("RERANKER_MODEL") or "gte-rerank-v2"


__all__ = [
    "STORAGE_PATH",
    "PADDLE_OCR_SUBDIR",
    "paddle_ocr_root",
    "MODELS_SUBDIR",
    "models_root",
    "PAGE_ASSETS_SUBDIR",
    "page_assets_root",
    "page_assets_path",
    "INVENTORY_SUBDIR",
    "inventory_root",
    "inventory_path",
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
    "APP_DB_SUBDIR",
    "app_db_dir",
    "app_db_path",
    "UPLOADS_SUBDIR",
    "uploads_root",
    "upload_path",
    "trace_run_path",
    "relativize_trace_dir",
    "JWT_SECRET",
    "JWT_SECRET_IS_DEFAULT",
    "ALLOW_INSECURE_JWT",
    "JWT_ALGORITHM",
    "JWT_EXP_MINUTES",
    "DEFAULT_ADMIN_USERNAME",
    "DEFAULT_ADMIN_PASSWORD",
    "DEFAULT_ADMIN_PASSWORD_IS_DEFAULT",
    "CORS_ORIGINS",
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
    "RERANKER_API_KEY",
    "RERANKER_API_BASE_URL",
    "RERANKER_MODEL",
    "TAVILY_API_KEY",
    "TAVILY_API_BASE_URL",
]
