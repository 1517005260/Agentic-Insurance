"""Login + identity routes."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import create_access_token, verify_password
from api.deps import get_current_user, get_session
from api.models import User


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
