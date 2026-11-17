"""Unit tests for ``SqlAlchemyAuditLogWriter`` (Milestone F, section 7).

No database - a minimal fake session (same pattern as
tests/test_embedding_repository.py) just records what was added, since
the point of this test is the ``AuditLogEntry`` -> ``AuditLogRecord``
field mapping and the "never commits internally" contract, not real SQL.
"""

from __future__ import annotations

import pytest

from app.clinical.audit import AuditLogEntry
from app.db.repositories.audit_repository import SqlAlchemyAuditLogWriter
from app.models.audit import AuditLogRecord


class _FakeAsyncSession:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.committed = False

    def add(self, instance: object) -> None:
        self.added.append(instance)

    async def commit(self) -> None:  # pragma: no cover - must never be called
        self.committed = True


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_write_maps_entry_fields_onto_audit_log_record() -> None:
    session = _FakeAsyncSession()
    writer = SqlAlchemyAuditLogWriter(session)  # type: ignore[arg-type]
    entry = AuditLogEntry(
        request_id="req-1",
        patient_id="patient-001",
        question="What is the target blood pressure?",
        question_category="guideline",
        model="gpt-4o-mini",
        retrieved_document_ids=["doc-1", "doc-2"],
        output_status="normal",
        evidence_sufficiency="sufficient",
        confidence=0.9,
        confidence_label="high",
        latency_ms=123.4,
        safety_flags=["some_flag"],
        hallucination_rate=0.0,
        citation_correctness=1.0,
        warnings=["a warning"],
        reasons=["a reason"],
        stage_timings={"retrieval": 0.1},
    )

    await writer.write(entry)

    assert len(session.added) == 1
    record = session.added[0]
    assert isinstance(record, AuditLogRecord)
    assert record.request_id == "req-1"
    assert record.patient_id == "patient-001"
    assert record.model == "gpt-4o-mini"
    assert record.retrieved_document_ids == ["doc-1", "doc-2"]
    assert record.output_status == "normal"
    assert record.confidence == 0.9
    assert record.safety_flags == ["some_flag"]
    assert record.hallucination_rate == 0.0
    assert record.citation_correctness == 1.0


@pytest.mark.anyio
async def test_write_never_commits_internally() -> None:
    session = _FakeAsyncSession()
    writer = SqlAlchemyAuditLogWriter(session)  # type: ignore[arg-type]
    entry = AuditLogEntry(
        request_id="req-2",
        patient_id=None,
        question="q",
        question_category=None,
        model=None,
    )

    await writer.write(entry)

    assert session.committed is False


@pytest.mark.anyio
async def test_write_never_stores_the_system_prompt_or_raw_evidence_text() -> None:
    # AuditLogEntry itself has no such field - this test guards against a
    # future field being added there and silently mapped through here.
    session = _FakeAsyncSession()
    writer = SqlAlchemyAuditLogWriter(session)  # type: ignore[arg-type]
    entry = AuditLogEntry(
        request_id="req-3",
        patient_id=None,
        question="q",
        question_category=None,
        model=None,
    )

    await writer.write(entry)

    record = session.added[0]
    record_fields = {
        column.name for column in AuditLogRecord.__table__.columns
    }
    assert "system_prompt" not in record_fields
    assert "chain_of_thought" not in record_fields
    assert "raw_evidence_text" not in record_fields
