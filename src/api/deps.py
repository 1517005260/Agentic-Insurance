"""Shared FastAPI dependencies: DB session, current user, RBAC guard."""
from typing import AsyncIterator

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import decode_token
from api.db import SessionLocal
from api.models import User


# tokenUrl points at the login route — this is what enables the "Authorize"
# button in the auto-generated /docs page. ``auto_error=False`` would let
# us treat missing tokens as "anonymous"; we don't need that, every
# protected route requires auth.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_session() -> AsyncIterator[AsyncSession]:
    """Per-request DB session. Commits on success, rolls back on exception."""
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_session),
) -> User:
    try:
        payload = decode_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    uid = payload.get("uid")
    if not isinstance(uid, int):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="malformed token: missing uid",
        )

    # We re-fetch the user on every request: cheap (single PK lookup,
    # ~0.1ms) and lets us enforce ``is_active=False`` immediately on
    # disable without waiting for the token to expire.
    user = await session.get(User, uid)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="user not found or disabled",
        )
    return user


def require_role(*allowed: str):
    """Factory for a role-guard dependency.

    Usage::

        @router.delete("/files/{file_id}", dependencies=[Depends(require_role("admin"))])
    """

    async def _dep(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"requires role(s): {', '.join(allowed)}",
            )
        return user

    return _dep


require_admin = require_role("admin")
