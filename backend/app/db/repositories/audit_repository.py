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
