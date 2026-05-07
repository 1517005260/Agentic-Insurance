"""Admin-only read endpoint for the ``audit_log`` table.

Exists because some events outlive the row they refer to. The clearest
example: a successful file delete cascades the matching ``IngestJob``
rows away, so ``GET /files/{id}/jobs`` returns 404 once the file is
gone — but ``run_delete`` writes ``file.delete.complete`` (with the
per-step purge counts) into ``audit_log`` *before* the cascade fires,
specifically so this endpoint can recover the trail.

Filtering is intentionally narrow (action / target prefix). Anything
richer belongs in a real query layer; for the demo a small WHERE +
DESC + LIMIT/OFFSET is fine.
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_session, require_admin
from api.models import AuditLog


router = APIRouter(
    prefix="/audit",
    tags=["audit"],
    dependencies=[Depends(require_admin)],
)


class AuditEntryOut(BaseModel):
    id: int
    user_id: Optional[int]
    action: str
    target: Optional[str]
    payload_json: Optional[str]
    at: datetime


@router.get("", response_model=list[AuditEntryOut])
async def list_audit(
    action: Optional[str] = Query(None, description="Exact action match, e.g. 'file.delete.complete'"),
    target: Optional[str] = Query(None, description="Exact target match (typically a file_id)"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_session),
) -> list[AuditEntryOut]:
    stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit).offset(offset)
    if action is not None:
        stmt = stmt.where(AuditLog.action == action)
    if target is not None:
        stmt = stmt.where(AuditLog.target == target)
    rows = (await db.execute(stmt)).scalars().all()
    return [
        AuditEntryOut(
            id=r.id,
            user_id=r.user_id,
            action=r.action,
            target=r.target,
            payload_json=r.payload_json,
            at=r.at,
        )
        for r in rows
    ]


@router.get("/{entry_id}", response_model=AuditEntryOut)
async def get_audit_entry(
    entry_id: int,
    db: AsyncSession = Depends(get_session),
) -> AuditEntryOut:
    row = await db.get(AuditLog, entry_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="audit entry not found")
    return AuditEntryOut(
        id=row.id,
        user_id=row.user_id,
        action=row.action,
        target=row.target,
        payload_json=row.payload_json,
        at=row.at,
    )
