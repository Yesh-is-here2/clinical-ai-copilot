"""Real (PostgreSQL-backed) implementation of ``app.clinical.audit.
AuditLogWriter`` (Milestone F, section 7).

This is the writer Milestone E's graph (``app.clinical.graph.
write_audit_log_node``) is designed to be handed in production - see
that node's docstring for the non-propagating-failure contract this
writer must never break: a write here must raise on genuine failure (so
the graph node can record it in ``component_errors``) but must never be
called in a way that could turn a safe, already-decided clinical
response into a failed HTTP request.

Deliberately does not commit internally (unlike
``SqlAlchemyEmbeddingRepository``, Milestone C/D's precedent): the audit
row is one of several tables (``queries``, ``retrieval_results``,
``generated_responses``, ``citations``, ``audit_logs``) that must commit
together, atomically, for one clinical query (section 5) - see
``app.services.clinical_query_service.ClinicalQueryService``, which owns
the single commit.

Field mapping is a near-direct copy from ``AuditLogEntry`` (see that
dataclass's docstring) - ``AuditLogRecord`` has no columns for
``question_category``/``confidence_label``/``warnings``/``reasons``/
``stage_timings``, so those are intentionally dropped here rather than
requiring a schema migration for this milestone; nothing about that
represents lost safety-relevant data, since the full set is still
present on the persisted ``queries``/``generated_responses`` rows for
the same request.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.clinical.audit import AuditLogEntry
from app.models.audit import AuditLogRecord


class SqlAlchemyAuditLogWriter:
    """``AuditLogWriter`` protocol implementation backed by an
    ``AsyncSession``. Never store secrets/API keys/raw auth headers/the
    system prompt/chain-of-thought here - see ``AuditLogRecord``'s own
    docstring."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def write(self, entry: AuditLogEntry) -> None:
        record = AuditLogRecord(
            request_id=entry.request_id,
            user_id=entry.actor_user_id,
            role=entry.actor_role,
            patient_id=entry.patient_id,
            question=entry.question,
            model=entry.model,
            retrieved_document_ids=list(entry.retrieved_document_ids),
            output_status=entry.output_status,
            evidence_sufficiency=entry.evidence_sufficiency,
            confidence=entry.confidence,
            latency_ms=entry.latency_ms,
            safety_flags=list(entry.safety_flags),
            hallucination_rate=entry.hallucination_rate,
            citation_correctness=entry.citation_correctness,
        )
        self._session.add(record)
        # Deliberately no `await self._session.flush()`/`commit()` here -
        # see module docstring. The caller commits once, atomically.


class AuditLogRepository:
    """Read-only audit-log queries for GET /audit (Phase 7, section
    7.7). Reviewer/admin only - the route enforces that; this class
    just reads."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_records(
        self,
        *,
        request_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLogRecord]:
        """Audit rows, newest first (stable order for pagination)."""

        stmt = select(AuditLogRecord).order_by(
            AuditLogRecord.created_at.desc(), AuditLogRecord.id.desc()
        )
        if request_id is not None:
            stmt = stmt.where(AuditLogRecord.request_id == request_id)
        stmt = stmt.limit(limit).offset(offset)
        return list((await self._session.execute(stmt)).scalars().all())

    async def count_records(self, *, request_id: str | None = None) -> int:
        """Total audit rows (for the list response), same filter."""

        stmt = select(func.count()).select_from(AuditLogRecord)
        if request_id is not None:
            stmt = stmt.where(AuditLogRecord.request_id == request_id)
        return int((await self._session.execute(stmt)).scalar_one())
