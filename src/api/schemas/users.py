"""Admin user-management DTOs.

Roles are constrained at the DB layer (`ck_users_role`) and mirrored
here so the OpenAPI spec is self-documenting. Soft-delete is the
only deletion mode exposed; hard delete is not surfaced.
"""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


RoleLiteral = Literal["admin", "analyst"]


class UserOut(BaseModel):
    id: int
    username: str
    role: RoleLiteral
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class UserCreate(BaseModel):
    """Admin-side user creation payload."""

    # Username constraint mirrors the column (VARCHAR(64)) and adds a
    # restrictive charset so we don't accept whitespace / punctuation
    # that would later confuse downstream tooling. ASCII-only by
    # design — admin should use the visible login the tester types.
    username: str = Field(..., min_length=2, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    # Password lower bound is intentionally low (8) for the demo
    # admin's seed flow but the route handler enforces a stronger
    # bar (>= 8 + at least one digit + one letter).
    password: str = Field(..., min_length=8, max_length=128)
    role: RoleLiteral


class UserUpdate(BaseModel):
    """Patchable user fields. Username is immutable post-create.

    All fields optional so callers can patch a single attribute
    without echoing the rest. The route rejects an empty body.
    """

    role: Optional[RoleLiteral] = None
    is_active: Optional[bool] = None


class PasswordReset(BaseModel):
    """Admin-side password reset. Lands as a new bcrypt hash."""

    new_password: str = Field(..., min_length=8, max_length=128)
