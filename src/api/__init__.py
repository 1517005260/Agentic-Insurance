"""FastAPI web layer.

Thin facade over the algorithm layer (`agentic/`, `rag/`, `ingestion/`).
The web app only owns: identity (``users``), conversation persistence
(``chat_sessions`` / ``chat_messages``), file lifecycle (``files`` /
``ingest_jobs``), runtime tunables (``app_config``) and an audit trail.
Everything else lives in the algorithm packages and is imported here.
"""
