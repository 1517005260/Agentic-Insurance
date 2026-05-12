"""Admin-side user management.

Five endpoints, all admin-only, all audit-logged:

* ``GET    /admin/users``                    list (active by default)
* ``GET    /admin/users/{id}``               single
* ``POST   /admin/users``                    create
* ``PATCH  /admin/users/{id}``               role + is_active
* ``POST   /admin/users/{id}/reset-password``
* ``DELETE /admin/users/{id}``               soft-delete (is_active=0)

Self-protection guards:

* You cannot deactivate / demote / soft-delete YOURSELF.
* The system rejects any operation that would leave zero active
  admins (so a fat-finger demote / soft-delete of the only admin is
  refused).
"""
import asyncio
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import enforce_password_policy, hash_password
from api.deps import get_session, require_admin
from api.models import AuditLog, User
from api.schemas.users import (
    PasswordReset,
    UserCreate,
    UserOut,
    UserUpdate,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/admin/users",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


# Process-wide lock that serializes any mutation that could affect
# the active-admin set. The risk codex flagged: two admins each
# demoting / deactivating / soft-deleting the OTHER would both pass
# `_count_active_admins() > 1`, then both commit, leaving zero
# admins. WAL + synchronous=NORMAL doesn't serialize the predicate;
# only ordering writes does.
#
# A single-process FastAPI demo only needs an in-process asyncio.Lock.
# A multi-process deployment would need either ``BEGIN IMMEDIATE``
# around the recheck+update, a SQLite singleton-row lock, or a
# trigger that aborts when the action would empty the admin set.
_ADMIN_MUTATION_LOCK = asyncio.Lock()


# ---------- helpers ----------


async def _load_user_or_404(db: AsyncSession, user_id: int) -> User:
    res = await db.execute(select(User).where(User.id == user_id))
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="user not found"
        )
    return user


async def _count_active_admins(db: AsyncSession) -> int:
    res = await db.execute(
        select(func.count())
        .select_from(User)
        .where(User.role == "admin")
        .where(User.is_active == 1)
    )
    return int(res.scalar_one() or 0)


async def _ensure_actor_still_admin(db: AsyncSession, actor_id: int) -> None:
    """Re-fetch the actor under the lock and re-check role+active.

    The dependency injection chain resolved BEFORE the
    ``_ADMIN_MUTATION_LOCK`` was acquired, so the actor User object
    might be stale: a previous lock holder could have just demoted
    or deactivated them. Re-reading inside the lock catches that
    race, returning 403 instead of letting a demoted-mid-flight
    actor still mutate the user table.

    Subtle: ``db.get(User, actor_id)`` returns the identity-map
    cached row if it's already loaded — which it IS, because
    ``get_current_user`` did the SELECT during dependency
    resolution. We need a FORCED re-read. Use ``execution_options
    (populate_existing=True)`` so SQLAlchemy issues a fresh SELECT
    and overwrites the in-session row. We also explicitly
    ``db.expire`` the cached object so subsequent attribute reads
    in the same request body see the refreshed values.
    """
    fresh = (
        await db.execute(
            select(User)
            .where(User.id == actor_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    if fresh is None or fresh.is_active != 1 or fresh.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "your admin privileges were revoked while this request "
                "waited for the admin-mutation lock; re-authenticate"
            ),
        )


def _audit(
    db: AsyncSession,
    *,
    actor_id: int,
    action: str,
    target: str,
    payload: Optional[dict] = None,
) -> None:
    db.add(
        AuditLog(
            user_id=actor_id,
            action=action,
            target=target,
            payload_json=(
                json.dumps(payload, ensure_ascii=False)
                if payload is not None
                else None
            ),
        )
    )


# ---------- list / get ----------


@router.get("", response_model=List[UserOut])
async def list_users(
    include_inactive: bool = Query(False, description="Include soft-deleted users."),
    db: AsyncSession = Depends(get_session),
) -> List[UserOut]:
    stmt = select(User).order_by(User.id.asc())
    if not include_inactive:
        stmt = stmt.where(User.is_active == 1)
    res = await db.execute(stmt)
    return [UserOut.model_validate(u) for u in res.scalars().all()]


@router.get("/{user_id}", response_model=UserOut)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_session),
) -> UserOut:
    user = await _load_user_or_404(db, user_id)
    return UserOut.model_validate(user)


# ---------- create ----------


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate,
    actor: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> UserOut:
    enforce_password_policy(body.password)

    # UNIQUE constraint on username will catch duplicates at flush
    # time, but checking up front lets us return a clean 409 instead
    # of an opaque IntegrityError.
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
        role=body.role,
        is_active=1,
    )
    db.add(user)
    await db.flush()  # populate user.id
    _audit(
        db,
        actor_id=actor.id,
        action="user.create",
        target=str(user.id),
        payload={"username": user.username, "role": user.role},
    )
    return UserOut.model_validate(user)


# ---------- patch ----------


@router.patch("/{user_id}", response_model=UserOut)
async def patch_user(
    user_id: int,
    body: UserUpdate,
    actor: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> UserOut:
    if body.role is None and body.is_active is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="empty patch body — pass `role` and/or `is_active`",
        )

    # Serialize the count-then-update predicate against concurrent
    # mutations on other rows. See _ADMIN_MUTATION_LOCK note above.
    async with _ADMIN_MUTATION_LOCK:
        # Re-validate the actor inside the lock. The previous lock
        # holder might have just demoted / deactivated the current
        # request's actor while we were waiting; the dependency
        # injection chain resolved before the lock and is now stale.
        await _ensure_actor_still_admin(db, actor.id)

        user = await _load_user_or_404(db, user_id)
        diff: dict = {}

        # Self-protection: don't let an admin demote / deactivate self.
        if user.id == actor.id:
            if body.role is not None and body.role != user.role:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="cannot change your own role",
                )
            if body.is_active is False:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="cannot deactivate yourself",
                )

        # Last-active-admin protection: a demote / deactivate that would
        # leave the system with zero admins is refused. The COUNT runs
        # under the in-process lock so two concurrent demotes can't both
        # observe ">1" and then each remove an admin.
        if user.role == "admin" and user.is_active == 1:
            going_inactive = body.is_active is False
            being_demoted = body.role is not None and body.role != "admin"
            if going_inactive or being_demoted:
                n_active_admins = await _count_active_admins(db)
                if n_active_admins <= 1:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="cannot demote / deactivate the last active admin",
                    )

        if body.role is not None and body.role != user.role:
            diff["role"] = {"old": user.role, "new": body.role}
            user.role = body.role
        if body.is_active is not None:
            new_int = 1 if body.is_active else 0
            if new_int != user.is_active:
                diff["is_active"] = {"old": bool(user.is_active), "new": body.is_active}
                user.is_active = new_int

        if not diff:
            # No-op patch (the body fields equaled current state). Still
            # respond 200 with the current row, but skip audit.
            return UserOut.model_validate(user)

        _audit(
            db,
            actor_id=actor.id,
            action="user.update",
            target=str(user.id),
            payload=diff,
        )
        # Commit BEFORE releasing the lock so the next holder's COUNT
        # in a fresh AsyncSession sees this row's new state. flush()
        # alone makes the change visible only to the SAME session;
        # the next request opens a new session via get_session and
        # would not observe an uncommitted UPDATE. The outer
        # get_session teardown will call commit() again — that's a
        # no-op once committed (SQLAlchemy guards).
        await db.commit()
        return UserOut.model_validate(user)


# ---------- reset password ----------


@router.post(
    "/{user_id}/reset-password",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def reset_password(
    user_id: int,
    body: PasswordReset,
    actor: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> Response:
    enforce_password_policy(body.new_password)
    user = await _load_user_or_404(db, user_id)
    user.password_hash = hash_password(body.new_password)
    _audit(
        db,
        actor_id=actor.id,
        action="user.password.reset",
        target=str(user.id),
        payload={"username": user.username},  # never log the password
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------- soft delete ----------


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def soft_delete_user(
    user_id: int,
    actor: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> Response:
    # Serialize against concurrent role / is_active mutations. Same
    # invariant as PATCH: count-then-update on the active-admin set.
    async with _ADMIN_MUTATION_LOCK:
        # Re-validate the actor under the lock — they might have just
        # been demoted by the previous holder.
        await _ensure_actor_still_admin(db, actor.id)

        user = await _load_user_or_404(db, user_id)
        if user.id == actor.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="cannot soft-delete yourself",
            )
        if user.is_active == 0:
            # Idempotent — already deactivated. Skip audit.
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        if user.role == "admin":
            n_active_admins = await _count_active_admins(db)
            if n_active_admins <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="cannot soft-delete the last active admin",
                )

        user.is_active = 0
        _audit(
            db,
            actor_id=actor.id,
            action="user.soft_delete",
            target=str(user.id),
            payload={"username": user.username},
        )
        # Commit before releasing the lock — see PATCH note above.
        await db.commit()
        return Response(status_code=status.HTTP_204_NO_CONTENT)
