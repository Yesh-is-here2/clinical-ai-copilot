"""``GET /audit`` (Phase 7, section 7.7).

A read-only view over the persisted audit trail: who asked what,
with what outcome. Reviewer/admin only - the audit trail is review
material, and a clinician has no business browsing other users'
query history. Filterable by ``request_id`` for tracing one request
end to end.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.deps import (
    Actor,
    get_audit_repository,
    require_roles,
)
from app.db.repositories.audit_repository import AuditLogRepository
from app.models.user import UserRole
from app.schemas.audit import AuditLogListResponse, AuditLogRecordOut

router = APIRouter(prefix="/audit", tags=["Audit"])


@router.get(
    "",
    response_model=AuditLogListResponse,
    summary="List audit log records, newest first",
)
async def list_audit_records(
    actor: Annotated[
        Actor, Depends(require_roles(UserRole.REVIEWER, UserRole.ADMIN))
    ],
    repository: Annotated[AuditLogRepository, Depends(get_audit_repository)],
    request_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AuditLogListResponse:
    """Reviewer/admin only. (``actor`` is the auth gate: the
    dependency enforces the role even though the value is unused.)"""

    del actor
    records = await repository.list_records(
        request_id=request_id, limit=limit, offset=offset
    )
    total = await repository.count_records(request_id=request_id)
    return AuditLogListResponse(
        records=[AuditLogRecordOut.model_validate(record) for record in records],
        total=total,
    )
