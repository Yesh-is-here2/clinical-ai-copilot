"""Persistence for one clinical query's full lifecycle (Milestone F,
sections 4/5/6): the ``queries`` row itself, its retrieved-evidence rows
(``retrieval_results``), its generated answer (``generated_responses``),
and that answer's citations (``citations``).

Only this module (and ``app.db.repositories.audit_repository`` for the
``audit_logs`` table) ever constructs a ``QueryRecord``/
``RetrievalResultRecord``/``GeneratedResponseRecord``/``CitationRecord``
ORM row - ``app.services.clinical_query_service.ClinicalQueryService``
talks only to this repository's methods, never to SQLAlchemy directly
(section 6's repository/service boundary).

Deliberately does not commit internally - see
``app.db.repositories.audit_repository`` for why: this repository's
``persist_query`` and the audit writer's ``write`` both add rows to the
*same* session within one request, and the caller commits once,
atomically, so a partially-written query can never exist (section 5).

Every retrieved-evidence chunk and every cited chunk referenced here
always already exists in ``document_chunks`` by construction: real
retrieval (see ``app.api.deps.get_hybrid_retrieval_service``) only ever
surfaces chunks loaded from that table in the first place - so the
``RESTRICT``/``CASCADE`` foreign keys on ``retrieval_results.chunk_id``/
``citations.chunk_id`` are never violated by a normal query.

``QueryRecord.patient_id`` is a real foreign key to ``patients.patient_id``
(``ondelete="SET NULL"``, see app.models.query/app.models.patient) - but
synthetic patients are file-backed (``app.services.patient.
PatientService`` reads ``data/fhir/patients/*.json``) and nothing
inserted them into that SQL table before this milestone. ``ensure_
patient_synced`` closes that gap with an idempotent upsert (``ON
CONFLICT DO NOTHING``, keyed on the existing file-backed ID) so a
clinical query for a known synthetic patient can actually commit,
without requiring a separate seed script or migration and without ever
overwriting the file-backed fixture data (that stays authoritative for
``GET /patients``/``GET /patients/{id}`` - see app.models.patient.
Patient's own docstring on that boundary).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.clinical.citation_validation import CitationValidationReport
from app.clinical.final_response import FinalClinicalResponse
from app.models.patient import Patient
from app.models.query import (
    CitationRecord,
    GeneratedResponseRecord,
    QueryRecord,
    RetrievalResultRecord,
)
from app.schemas.patient import PatientContext


@dataclass(frozen=True)
class PersistedQueryInput:
    """Everything needed to persist one completed clinical query.

    Deliberately takes already-resolved, plain values (not the raw
    ``ClinicalState`` TypedDict) so this module has no dependency on
    ``app.clinical.graph``/``app.clinical.state`` - the service layer is
    what bridges graph state to this shape.
    """

    query_id: uuid.UUID
    request_id: str
    patient_id: str | None
    question: str
    question_category: str | None
    status: str
    final_response: FinalClinicalResponse
    model_name: str | None
    latency_ms: float | None
    retrieval_rows: list["RetrievedEvidenceInput"] = field(default_factory=list)
    citation_status_by_id: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedEvidenceInput:
    chunk_id: str
    retrieval_method: str
    rank: int
    score: float
    reranker_score: float | None = None


class ClinicalQueryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def ensure_patient_synced(self, patient_context: PatientContext | None) -> None:
        """Idempotently upsert a minimal ``patients`` row for
        ``patient_context``, if given. Must run (and be flushed/ordered)
        before ``persist_query`` when ``PersistedQueryInput.patient_id``
        is set, or the ``queries.patient_id`` foreign key will reject the
        insert - see this module's docstring."""

        if patient_context is None:
            return

        stmt = (
            pg_insert(Patient)
            .values(
                patient_id=patient_context.patient_id,
                full_name=patient_context.full_name,
                sex=patient_context.sex,
                birth_date=patient_context.birth_date,
                synthetic=patient_context.synthetic,
            )
            .on_conflict_do_nothing(index_elements=[Patient.patient_id])
        )
        await self._session.execute(stmt)

    async def persist_query(self, data: PersistedQueryInput) -> None:
        query_record = QueryRecord(
            id=data.query_id,
            request_id=data.request_id,
            patient_id=data.patient_id,
            question=data.question,
            question_category=data.question_category,
            evidence_status=data.final_response.evidence_status.value,
            confidence=data.final_response.confidence,
            status=data.status,
            latency_ms=data.latency_ms,
        )
        self._session.add(query_record)

        seen_chunk_ids: set[str] = set()
        for row in data.retrieval_rows:
            if row.chunk_id in seen_chunk_ids:
                # Fused/reranked evidence is already deduplicated by
                # chunk_id upstream (app.retrieval.fusion) - this guard
                # only protects the unique constraint
                # (query_id, chunk_id, retrieval_method) against ever
                # being violated if that upstream invariant changes.
                continue
            seen_chunk_ids.add(row.chunk_id)
            self._session.add(
                RetrievalResultRecord(
                    query_id=data.query_id,
                    chunk_id=row.chunk_id,
                    retrieval_method=row.retrieval_method,
                    rank=row.rank,
                    score=row.score,
                    reranker_score=row.reranker_score,
                )
            )

        generated_response_id = uuid.uuid4()
        self._session.add(
            GeneratedResponseRecord(
                id=generated_response_id,
                query_id=data.query_id,
                answer=data.final_response.answer,
                key_points=list(data.final_response.key_points),
                warnings=list(data.final_response.warnings),
                safety_flags=list(data.final_response.safety_flags),
                model=data.model_name,
            )
        )

        for citation in data.final_response.citations:
            self._session.add(
                CitationRecord(
                    generated_response_id=generated_response_id,
                    chunk_id=citation.chunk_id,
                    citation_label=citation.citation_id,
                    quoted_support=citation.quoted_support,
                    support_classification=data.citation_status_by_id.get(
                        citation.citation_id, "not_applicable"
                    ),
                )
            )

    async def get_by_id(self, query_id: uuid.UUID) -> QueryRecord | None:
        stmt = (
            select(QueryRecord)
            .where(QueryRecord.id == query_id)
            .options(
                selectinload(QueryRecord.generated_response).selectinload(
                    GeneratedResponseRecord.citations
                )
            )
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def exists(self, query_id: uuid.UUID) -> bool:
        stmt = select(QueryRecord.id).where(QueryRecord.id == query_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None


def build_citation_status_map(
    citation_validation: CitationValidationReport | None,
) -> dict[str, str]:
    """Map each citation ID to the status of the (last) claim that used
    it as support, for ``CitationRecord.support_classification``. A
    citation shared by multiple claims gets whichever claim's status was
    seen last - this is a human-readable summary label, not a hard
    invariant, so that ambiguity is acceptable."""

    if citation_validation is None:
        return {}

    status_by_id: dict[str, str] = {}
    for claim_validation in citation_validation.claim_validations:
        for citation_id in claim_validation.supporting_citation_ids:
            status_by_id[citation_id] = claim_validation.status.value

    return status_by_id
