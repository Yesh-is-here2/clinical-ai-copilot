"""Unit tests for app.clinical.audit (Milestone E, section 2)."""

from __future__ import annotations

import pytest

from app.clinical.audit import AuditLogEntry, InMemoryAuditLogWriter, NullAuditLogWriter


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _entry(**overrides) -> AuditLogEntry:
    defaults = dict(
        request_id="req-1",
        patient_id="synthetic-1",
        question="What is the target blood pressure?",
        question_category="guideline",
        model="fake-structured-llm",
    )
    defaults.update(overrides)
    return AuditLogEntry(**defaults)


@pytest.mark.anyio
async def test_null_writer_is_a_no_op() -> None:
    writer = NullAuditLogWriter()

    # Should not raise, and has no observable state to assert on - the
    # point is that a production deployment can start with this and
    # never fail on the audit-write step.
    await writer.write(_entry())


@pytest.mark.anyio
async def test_in_memory_writer_records_entries_in_order() -> None:
    writer = InMemoryAuditLogWriter()

    await writer.write(_entry(request_id="req-1"))
    await writer.write(_entry(request_id="req-2"))

    assert [e.request_id for e in writer.entries] == ["req-1", "req-2"]


def test_entry_never_carries_a_prompt_or_chain_of_thought_field() -> None:
    entry = _entry()

    field_names = set(entry.__dataclass_fields__.keys())
    forbidden_substrings = ("prompt", "chain_of_thought", "reasoning_trace", "raw_llm")

    for field_name in field_names:
        for forbidden in forbidden_substrings:
            assert forbidden not in field_name.lower()


def test_entry_defaults_are_safe_empty_values() -> None:
    entry = _entry()

    assert entry.retrieved_document_ids == []
    assert entry.safety_flags == []
    assert entry.warnings == []
    assert entry.reasons == []
    assert entry.stage_timings == {}
    assert entry.output_status == "unknown"
