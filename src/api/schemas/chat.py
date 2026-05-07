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
    """Create-time payload. mode + agent_kind are immutable post-create."""

    mode: ModeLiteral
    agent_kind: Optional[AgentKindLiteral] = None
    title: str = Field(default="New chat", max_length=255)

    @model_validator(mode="after")
    def _enforce_mode_kind(self) -> "SessionCreate":
        # Mirror the DB CHECK ck_sessions_mode_kind so we 422 cleanly
        # before hitting the engine, with a more helpful message.
        if self.mode == "agent" and self.agent_kind is None:
            raise ValueError("agent_kind is required when mode='agent'")
        if self.mode == "rag" and self.agent_kind is not None:
            raise ValueError("agent_kind must be omitted when mode='rag'")
        return self


class SessionUpdate(BaseModel):
    """Patchable session fields. mode / agent_kind intentionally not patchable."""

    title: str = Field(..., min_length=1, max_length=255)


class SessionOut(BaseModel):
    id: int
    title: str
    mode: ModeLiteral
    agent_kind: Optional[AgentKindLiteral]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


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
