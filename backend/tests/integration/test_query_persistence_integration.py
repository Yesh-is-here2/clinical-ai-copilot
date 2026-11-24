"""Integration tests for Milestone F's query-lifecycle and audit
persistence, requiring a live PostgreSQL + pgvector database.

Excluded from the default test run by backend/pytest.ini
(`addopts = -m "not integration"`). Run explicitly with:

    docker compose exec backend pytest -m integration -q

These are NOT run in the sandbox this code was developed in (no route to
a real Postgres instance there) - do not report them as passing, and do
not weaken/skip-with-a-mock any of the assertions below to make that
happen. See tests/integration/README.md and
tests/integration/test_postgres_integration.py for the established
NullPool/isolated-engine + `itest-` scoped-cleanup pattern this file
follows.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.clinical.audit import AuditLogEntry
from app.clinical.citations import Citation
from app.clinical.confidence import ConfidenceLabel
from app.clinical.evidence_sufficiency import EvidenceSufficiencyStatus
from app.clinical.final_response import DISCLAIMER_TEXT, FinalClinicalResponse
from app.core.config import settings
from app.db.repositories.audit_repository import SqlAlchemyAuditLogWriter
from app.db.repositories.query_repository import (
    ClinicalQueryRepository,
    PersistedQueryInput,
    RetrievedEvidenceInput,
)
from app.models.audit import AuditLogRecord
from app.models.document import ClinicalDocumentRecord, DocumentChunkRecord
from app.models.patient import Patient
from app.models.query import (
    CitationRecord,
    GeneratedResponseRecord,
    QueryRecord,
    RetrievalResultRecord,
)
from app.schemas.patient import PatientContext

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

_PREFIX = "itest-"


async def _delete_test_owned_rows(session: AsyncSession) -> None:
    # Order matters: citations.chunk_id is ondelete=RESTRICT by design
    # (a cited chunk must not silently disappear), so queries must go
    # FIRST - deleting a query cascades through generated_responses to
    # citations. Only then is it safe to delete clinical_documents,
    # which cascades to document_chunks/document_embeddings.
    await session.execute(
        delete(QueryRecord).where(QueryRecord.request_id.like(f"{_PREFIX}%"))
    )
    await session.execute(
        delete(ClinicalDocumentRecord).where(
            ClinicalDocumentRecord.document_id.like(f"{_PREFIX}%")
        )
    )
    await session.execute(
        delete(AuditLogRecord).where(AuditLogRecord.request_id.like(f"{_PREFIX}%"))
    )
    await session.execute(
        delete(Patient).where(Patient.patient_id.like(f"{_PREFIX}%"))
    )
    await session.commit()


@pytest.fixture
async def db_session():
    test_engine = create_async_engine(
        settings.database_url, poolclass=NullPool, echo=False
    )
    test_session_factory = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with test_session_factory() as session:
        await _delete_test_owned_rows(session)
        try:
            yield session
        finally:
            await _delete_test_owned_rows(session)

    await test_engine.dispose()


async def _seed_document_and_chunk(
    session: AsyncSession, *, document_id: str, chunk_id: str
) -> None:
    session.add(
        ClinicalDocumentRecord(
            document_id=document_id,
            title="Integration Test Guideline",
            organization="Test Org",
            source_url="https://example.org/guideline",
            category="hypertension",
            access_date=date(2026, 9, 9),
        )
    )
    session.add(
        DocumentChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            section_id="s1",
            section_heading="Targets",
            chunk_number=0,
            text="Target blood pressure below 130/80 mmHg.",
            token_count=8,
        )
    )
    await session.commit()


def _final_response(*, request_id: str, citations: list[Citation]) -> FinalClinicalResponse:
    return FinalClinicalResponse(
        request_id=request_id,
        patient_summary="No patient context was provided for this question.",
        clinical_question="What is the target blood pressure?",
        answer="Target blood pressure below 130/80 mmHg [C1].",
        key_points=["Target below 130/80 mmHg"],
        citations=citations,
        confidence=0.9,
        confidence_label=ConfidenceLabel.HIGH,
        evidence_status=EvidenceSufficiencyStatus.SUFFICIENT,
        warnings=[],
        safety_flags=[],
        disclaimer=DISCLAIMER_TEXT,
    )


class TestQueryLifecyclePersistence:
    async def test_persist_query_generated_response_and_citations_round_trip(
        self, db_session: AsyncSession
    ) -> None:
        request_id = f"{_PREFIX}req-1"
        document_id = f"{_PREFIX}doc-1"
        chunk_id = f"{_PREFIX}doc-1:s1:0:abc"
        await _seed_document_and_chunk(
            db_session, document_id=document_id, chunk_id=chunk_id
        )

        citation = Citation(
            citation_id="C1",
            chunk_id=chunk_id,
            document_id=document_id,
            document_title="Integration Test Guideline",
            organization="Test Org",
            source_url="https://example.org/guideline",
            section="Targets",
            quoted_support="Target blood pressure below 130/80 mmHg.",
            retrieval_score=0.9,
        )
        query_id = uuid.uuid4()
        repository = ClinicalQueryRepository(db_session)

        await repository.persist_query(
            PersistedQueryInput(
                query_id=query_id,
                request_id=request_id,
                patient_id=None,
                question="What is the target blood pressure?",
                question_category="guideline",
                status="normal",
                final_response=_final_response(
                    request_id=request_id, citations=[citation]
                ),
                model_name="fake-structured-llm",
                latency_ms=42.0,
                retrieval_rows=[
                    RetrievedEvidenceInput(
                        chunk_id=chunk_id, retrieval_method="fused", rank=1, score=0.8
                    )
                ],
                citation_status_by_id={"C1": "supported"},
            )
        )
        await db_session.commit()

        fetched = await repository.get_by_id(query_id)
        assert fetched is not None
        assert fetched.request_id == request_id
        assert fetched.generated_response is not None
        assert fetched.generated_response.answer.startswith("Target blood pressure")
        assert len(fetched.generated_response.citations) == 1
        assert fetched.generated_response.citations[0].citation_label == "C1"

        retrieval_rows = (
            await db_session.execute(
                select(RetrievalResultRecord).where(
                    RetrievalResultRecord.query_id == query_id
                )
            )
        ).scalars().all()
        assert len(retrieval_rows) == 1
        assert retrieval_rows[0].chunk_id == chunk_id

    async def test_ensure_patient_synced_makes_a_query_with_patient_id_committable(
        self, db_session: AsyncSession
    ) -> None:
        # queries.patient_id is a real FK to patients.patient_id, and
        # synthetic patients are otherwise only file-backed - without
        # ensure_patient_synced, this commit would raise IntegrityError.
        request_id = f"{_PREFIX}req-patient-sync-1"
        patient_id = f"{_PREFIX}patient-sync-1"
        repository = ClinicalQueryRepository(db_session)

        await repository.ensure_patient_synced(
            PatientContext(patient_id=patient_id, full_name="Test Patient", sex="female")
        )
        await repository.persist_query(
            PersistedQueryInput(
                query_id=uuid.uuid4(),
                request_id=request_id,
                patient_id=patient_id,
                question="q",
                question_category=None,
                status="normal",
                final_response=_final_response(request_id=request_id, citations=[]),
                model_name=None,
                latency_ms=None,
            )
        )

        await db_session.commit()

        stored = (
            await db_session.execute(
                select(Patient).where(Patient.patient_id == patient_id)
            )
        ).scalar_one()
        assert stored.full_name == "Test Patient"

    async def test_ensure_patient_synced_is_idempotent_on_repeat_calls(
        self, db_session: AsyncSession
    ) -> None:
        patient_id = f"{_PREFIX}patient-sync-2"
        repository = ClinicalQueryRepository(db_session)
        context = PatientContext(
            patient_id=patient_id, full_name="Test Patient Two", sex="male"
        )

        await repository.ensure_patient_synced(context)
        await repository.ensure_patient_synced(context)
        await db_session.commit()

        rows = (
            await db_session.execute(
                select(Patient).where(Patient.patient_id == patient_id)
            )
        ).scalars().all()
        assert len(rows) == 1

    async def test_persist_audit_log_and_fetch_by_request_id(
        self, db_session: AsyncSession
    ) -> None:
        request_id = f"{_PREFIX}req-audit-1"
        writer = SqlAlchemyAuditLogWriter(db_session)

        await writer.write(
            AuditLogEntry(
                request_id=request_id,
                patient_id="patient-001",
                question="What is the target blood pressure?",
                question_category="guideline",
                model="fake-structured-llm",
                retrieved_document_ids=["doc-1"],
                output_status="normal",
                evidence_sufficiency="sufficient",
                confidence=0.9,
                latency_ms=42.0,
                safety_flags=[],
                hallucination_rate=0.0,
                citation_correctness=1.0,
            )
        )
        await db_session.commit()

        stored = (
            await db_session.execute(
                select(AuditLogRecord).where(AuditLogRecord.request_id == request_id)
            )
        ).scalar_one()

        assert stored.patient_id == "patient-001"
        assert stored.output_status == "normal"
        assert stored.confidence == 0.9

    async def test_deleting_a_query_cascades_to_retrieval_results_and_generated_response_and_citations(
        self, db_session: AsyncSession
    ) -> None:
        request_id = f"{_PREFIX}req-cascade-1"
        document_id = f"{_PREFIX}doc-cascade-1"
        chunk_id = f"{_PREFIX}doc-cascade-1:s1:0:abc"
        await _seed_document_and_chunk(
            db_session, document_id=document_id, chunk_id=chunk_id
        )

        citation = Citation(
            citation_id="C1",
            chunk_id=chunk_id,
            document_id=document_id,
            document_title="Integration Test Guideline",
            organization="Test Org",
            source_url="https://example.org/guideline",
            quoted_support="Target blood pressure below 130/80 mmHg.",
        )
        query_id = uuid.uuid4()
        repository = ClinicalQueryRepository(db_session)
        await repository.persist_query(
            PersistedQueryInput(
                query_id=query_id,
                request_id=request_id,
                patient_id=None,
                question="q",
                question_category=None,
                status="normal",
                final_response=_final_response(
                    request_id=request_id, citations=[citation]
                ),
                model_name=None,
                latency_ms=None,
                retrieval_rows=[
                    RetrievedEvidenceInput(
                        chunk_id=chunk_id, retrieval_method="fused", rank=1, score=0.5
                    )
                ],
            )
        )
        await db_session.commit()

        await db_session.execute(delete(QueryRecord).where(QueryRecord.id == query_id))
        await db_session.commit()

        remaining_retrieval = (
            await db_session.execute(
                select(RetrievalResultRecord).where(
                    RetrievalResultRecord.query_id == query_id
                )
            )
        ).scalars().all()
        remaining_generated = (
            await db_session.execute(
                select(GeneratedResponseRecord).where(
                    GeneratedResponseRecord.query_id == query_id
                )
            )
        ).scalars().all()

        assert remaining_retrieval == []
        assert remaining_generated == []

    async def test_transaction_rollback_leaves_no_partial_query_on_constraint_violation(
        self, db_session: AsyncSession
    ) -> None:
        # citations.chunk_id has ondelete=RESTRICT to document_chunks and,
        # more importantly here, is a real FK: citing a chunk_id that was
        # never persisted must fail the whole transaction rather than
        # leaving a query/generated_response row behind with a dangling
        # citation.
        request_id = f"{_PREFIX}req-rollback-1"
        query_id = uuid.uuid4()
        repository = ClinicalQueryRepository(db_session)

        bad_citation = Citation(
            citation_id="C1",
            chunk_id=f"{_PREFIX}does-not-exist",
            document_id=f"{_PREFIX}does-not-exist-doc",
            document_title="Nonexistent",
            organization="Nonexistent",
            source_url="https://example.org/nonexistent",
            quoted_support="...",
        )

        await repository.persist_query(
            PersistedQueryInput(
                query_id=query_id,
                request_id=request_id,
                patient_id=None,
                question="q",
                question_category=None,
                status="normal",
                final_response=_final_response(
                    request_id=request_id, citations=[bad_citation]
                ),
                model_name=None,
                latency_ms=None,
            )
        )

        with pytest.raises(IntegrityError):
            await db_session.commit()

        await db_session.rollback()

        assert (
            await db_session.execute(
                select(QueryRecord).where(QueryRecord.id == query_id)
            )
        ).scalar_one_or_none() is None
