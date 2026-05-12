"""Login + identity + self-registration routes."""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import (
    create_access_token,
    enforce_password_policy,
    hash_password,
    verify_password,
)
from api.deps import get_current_user, get_session
from api.models import AuditLog, User


logger = logging.getLogger(__name__)


router = APIRouter(prefix="/auth", tags=["auth"])


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class MeOut(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool


class RegisterIn(BaseModel):
    """Self-registration payload — mirrors admin :class:`UserCreate` but
    without ``role`` (every new self-registered account is an
    ``analyst``; admin elevation is a separate admin-only action).
    """

    # Same charset / length bounds as admin UserCreate so the two
    # paths produce indistinguishable rows.
    username: str = Field(..., min_length=2, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    password: str = Field(..., min_length=8, max_length=128)


@router.post("/login", response_model=TokenOut)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_session),
) -> TokenOut:
    # OAuth2PasswordRequestForm gives us ``username`` + ``password`` from
    # a x-www-form-urlencoded body — the standard /docs login flow uses
    # this. Frontend sends the same shape.
    res = await db.execute(select(User).where(User.username == form.username))
    user = res.scalar_one_or_none()
    if user is None or not verify_password(form.password, user.password_hash):
        # Same error for "no such user" and "wrong password" — don't
        # leak which usernames exist.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="account disabled",
        )
    token = create_access_token(uid=user.id, username=user.username, role=user.role)
    return TokenOut(access_token=token, role=user.role, username=user.username)


@router.get("/me", response_model=MeOut)
async def me(user: User = Depends(get_current_user)) -> MeOut:
    return MeOut(
        id=user.id,
        username=user.username,
        role=user.role,
        is_active=bool(user.is_active),
    )


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterIn,
    db: AsyncSession = Depends(get_session),
) -> TokenOut:
    """Public self-registration. Always creates an ``analyst`` account.

    Returns a fresh access token so the caller can drop straight into
    the app without a second round-trip through ``/auth/login``. The
    audit row records the registration with no actor (``user_id``
    NULL) — the new user is both subject and originator.
    """
    enforce_password_policy(body.password)

    # Pre-check for clearer 409 over the IntegrityError that the UNIQUE
    # constraint would raise at flush. Race-condition note: a concurrent
    # second register with the same username would still land in the
    # UNIQUE-violation branch below; both are surfaced as 409.
    existing = (
        await db.execute(select(User.id).where(User.username == body.username))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"username {body.username!r} already taken",
        )

    user = User(
        username=body.username,
        password_hash=hash_password(body.password),
        role="analyst",
        is_active=1,
    )
    db.add(user)
    try:
        await db.flush()
    except IntegrityError:
        # Concurrent register lost the UNIQUE race. Rolling back leaves
        # the session usable; surface as 409 so the frontend can
        # highlight the username field. Any other DB failure (operational
        # error, schema drift, …) propagates as a 500 instead of being
        # masked as "name taken".
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"username {body.username!r} already taken",
        )

    # The new user is both subject and originator — record them as the
    # actor so the audit UI joins to ``users.username`` instead of
    # rendering "(deleted)". FK ``ON DELETE SET NULL`` still leaves the
    # row searchable after a future hard delete.
    db.add(
        AuditLog(
            user_id=user.id,
            action="user.register",
            target=str(user.id),
            payload_json=json.dumps(
                {"username": user.username, "role": user.role},
                ensure_ascii=False,
            ),
        )
    )

    token = create_access_token(uid=user.id, username=user.username, role=user.role)
    return TokenOut(access_token=token, role=user.role, username=user.username)
