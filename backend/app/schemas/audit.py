"""Audit-log read schemas (Phase 7, section 7.7).

GET /audit returns persisted audit rows (reviewer/admin only).
Every field is read-only: the audit trail is written by the
application, never by clients.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AuditLogRecordOut(BaseModel):
    """One persisted audit row."""

    id: uuid.UUID
    request_id: str
    user_id: uuid.UUID | None
    role: str | None
    patient_id: str | None
    question: str | None
    model: str | None
    output_status: str | None
    evidence_sufficiency: str | None
    confidence: float | None
    latency_ms: float | None
    created_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class AuditLogListResponse(BaseModel):
    """GET /audit response."""

    records: list[AuditLogRecordOut]
    total: int
