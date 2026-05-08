"""Chat session / message DTOs.

Two surfaces:
- session-aware: persists user + assistant messages, requires ownership
- session-less smoke: ``/rag/stream`` and ``/agent/stream``, no DB writes
"""
import json
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ----------------------------------------------------------- session ----

ModeLiteral = Literal["rag", "agent"]
AgentKindLiteral = Literal["base", "proof", "graph"]


class SessionCreate(BaseModel):
    """Create-time payload. mode + agent_kind + web are immutable post-create."""

    mode: ModeLiteral
    agent_kind: Optional[AgentKindLiteral] = None
    web: bool = False
    title: str = Field(default="New chat", max_length=255)

    @model_validator(mode="after")
    def _enforce_mode_kind(self) -> "SessionCreate":
        # Mirror the DB CHECK ck_sessions_mode_kind so we 422 cleanly
        # before hitting the engine, with a more helpful message.
        if self.mode == "agent" and self.agent_kind is None:
            raise ValueError("agent_kind is required when mode='agent'")
        if self.mode == "rag" and self.agent_kind is not None:
            raise ValueError("agent_kind must be omitted when mode='rag'")
        # web=1 forbidden with proof/graph; chat UI never offers them
        # alongside the web toggle, so a request like that is almost
        # certainly a buggy client.
        if self.web and self.mode == "agent" and self.agent_kind in ("proof", "graph"):
            raise ValueError(
                "web=true is only valid with mode='rag' or "
                "(mode='agent' AND agent_kind='base'); proof / graph "
                "agents do not have a web variant"
            )
        return self


class SessionUpdate(BaseModel):
    """Patchable session fields. mode / agent_kind intentionally not patchable."""

    title: str = Field(..., min_length=1, max_length=255)


class SessionOut(BaseModel):
    id: int
    title: str
    mode: ModeLiteral
    agent_kind: Optional[AgentKindLiteral]
    web: bool = False
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("web", mode="before")
    @classmethod
    def _coerce_web(cls, v: Any) -> bool:
        # SQLite stores the column as int 0/1; translate to a real bool
        # for the API surface so clients see ``"web": true`` not ``1``.
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return bool(v)
        return bool(v)


# ---------------------------------------------------------- message ----


class MessagePost(BaseModel):
    """User message body. mode/agent_kind come from the parent session."""

    content: str = Field(..., min_length=1, max_length=8000)


class MessageOut(BaseModel):
    id: int
    role: Literal["user", "assistant", "tool", "system"]
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("metadata", mode="before")
    @classmethod
    def _parse_metadata(cls, v: Any) -> Any:
        # ORM stores metadata_json as TEXT; the source attr name on the
        # ORM is ``metadata_json``. We accept both shapes — when used
        # via ``from_attributes=True`` Pydantic looks up ``metadata`` on
        # the ORM object first; we catch the rename in the loader too.
        if v is None or isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return None
        return None


# ---------------------------------------------------- streaming bodies ----


class RagStreamRequest(BaseModel):
    """Body for ``POST /rag/stream`` (session-less)."""

    query: str = Field(..., min_length=1, max_length=8000)


class AgentStreamRequest(BaseModel):
    """Body for ``POST /agent/stream`` (session-less)."""

    query: str = Field(..., min_length=1, max_length=8000)
    kind: AgentKindLiteral = "base"
    web: bool = False

    @model_validator(mode="after")
    def _enforce_web_compat(self) -> "AgentStreamRequest":
        if self.web and self.kind in ("proof", "graph"):
            raise ValueError(
                "web=true is only valid with kind='base'; proof / graph "
                "agents do not have a web variant"
            )
        return self


class WebRagStreamRequest(BaseModel):
    """Body for ``POST /web-rag/stream`` (session-less smoke)."""

    query: str = Field(..., min_length=1, max_length=8000)
    include_domains: Optional[list[str]] = Field(default=None, max_length=20)
    exclude_domains: Optional[list[str]] = Field(default=None, max_length=20)
