"""ORM models for the web app — 7 tables, RBAC = ``admin`` / ``analyst``.

Design notes:

* ``files.file_id`` is the **PaddleOCR-derived id** (``<stem>_<sha16>``),
  not a surrogate UUID. Every downstream artifact on disk
  (``paddle_ocr/<file_id>/``, faiss meta rows, BM25 docs, graph vertices)
  is keyed by this same string, so using it as the row PK keeps the
  on-disk ↔ db mapping trivial and makes cascade-delete a one-id sweep.

* ``ingest_jobs`` records every async parse / re-index / delete attempt.
  A row exists per attempt (not per file) so a re-ingest after a failure
  doesn't lose the failure record. Foreign-keyed to ``files`` with
  ``ON DELETE CASCADE`` so wiping a file doesn't leave orphan jobs.

* ``app_config`` is a generic key-value store — the admin tunables
  panel (per project policy: every algorithmic constant should
  eventually become a config-center entry).

* ``audit_log`` is an append-only trail for any mutating action
  (uploads, deletes, role changes, config edits). No FK ``ON DELETE``
  on ``user_id`` — deleting a user must NOT erase their audit history.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ----------------------------------------------------------------- users

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    is_active: Mapped[bool] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    sessions: Mapped[list["ChatSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("role IN ('admin','analyst')", name="ck_users_role"),
        CheckConstraint("is_active IN (0,1)", name="ck_users_is_active"),
    )


# ------------------------------------------------------------------ files

# Status state machine (transitions are linear and one-way per attempt):
#
#     pending → parsing → indexing → ready
#                                  ↘ failed
#                              ↘ failed
#                       ↘ failed
#     ready → deleting → (row removed)   |   ready → indexing (re-ingest) → ready
#
# A failed file stays in `files` so the user can retry or delete it.
FILE_STATUSES = ("pending", "parsing", "indexing", "ready", "failed", "deleting")


class FileRecord(Base):
    __tablename__ = "files"

    file_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    suffix: Mapped[str] = mapped_column(String(16), nullable=False, default="")
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    page_count: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    error_msg: Mapped[Optional[str]] = mapped_column(Text)
    uploaded_by: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL")
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    jobs: Mapped[list["IngestJob"]] = relationship(
        back_populates="file", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','parsing','indexing','ready','failed','deleting')",
            name="ck_files_status",
        ),
        Index("ix_files_status", "status"),
    )


# ----------------------------------------------------------- ingest_jobs

INGEST_JOB_KINDS = ("parse_index", "reingest", "delete")
INGEST_JOB_STATUSES = ("pending", "running", "done", "failed")


class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("files.file_id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    error_msg: Mapped[Optional[str]] = mapped_column(Text)
    log_tail: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    file: Mapped["FileRecord"] = relationship(back_populates="jobs")

    __table_args__ = (
        CheckConstraint(
            "kind IN ('parse_index','reingest','delete')", name="ck_jobs_kind"
        ),
        CheckConstraint(
            "status IN ('pending','running','done','failed')", name="ck_jobs_status"
        ),
        Index("ix_jobs_file_id", "file_id"),
        Index("ix_jobs_status", "status"),
    )


# --------------------------------------------------------- chat sessions

CHAT_MODES = ("rag", "agent")
AGENT_KINDS = ("base", "proof", "graph")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New chat")
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    agent_kind: Mapped[Optional[str]] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    user: Mapped["User"] = relationship(back_populates="sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("mode IN ('rag','agent')", name="ck_sessions_mode"),
        CheckConstraint(
            "agent_kind IS NULL OR agent_kind IN ('base','proof','graph')",
            name="ck_sessions_agent_kind",
        ),
        # `mode='agent'` requires `agent_kind`; `mode='rag'` forbids it.
        CheckConstraint(
            "(mode = 'agent' AND agent_kind IS NOT NULL) OR "
            "(mode = 'rag' AND agent_kind IS NULL)",
            name="ck_sessions_mode_kind",
        ),
        Index("ix_sessions_user_updated", "user_id", "updated_at"),
    )


# --------------------------------------------------------- chat messages

MESSAGE_ROLES = ("user", "assistant", "tool", "system")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON blob: citations [{file_id, page_id, k}], tool calls, agent
    # events. Stored as TEXT so SQLite stays trivial; callers parse.
    metadata_json: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    session: Mapped["ChatSession"] = relationship(back_populates="messages")

    __table_args__ = (
        CheckConstraint(
            "role IN ('user','assistant','tool','system')", name="ck_msg_role"
        ),
        Index("ix_messages_session_id", "session_id", "id"),
    )


# ------------------------------------------------------------- app_config

class AppConfig(Base):
    """Generic key-value runtime config (admin tunables panel)."""

    __tablename__ = "app_config"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    updated_by: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL")
    )


# --------------------------------------------------------------- audit_log

class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # nullable: a deleted user shouldn't void their audit trail (FK ON
    # DELETE SET NULL). System-initiated entries (lifespan seed) also
    # carry NULL.
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL")
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    target: Mapped[Optional[str]] = mapped_column(String(255))
    payload_json: Mapped[Optional[str]] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_audit_at", "at"),)


__all__ = [
    "Base",
    "User",
    "FileRecord",
    "FILE_STATUSES",
    "IngestJob",
    "INGEST_JOB_KINDS",
    "INGEST_JOB_STATUSES",
    "ChatSession",
    "CHAT_MODES",
    "AGENT_KINDS",
    "ChatMessage",
    "MESSAGE_ROLES",
    "AppConfig",
    "AuditLog",
]
