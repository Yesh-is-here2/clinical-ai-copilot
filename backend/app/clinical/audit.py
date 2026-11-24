"""Audit logging boundary (Milestone E, section 2).

Only a protocol + two in-process implementations here - real persistent
storage (writing to the ``audit_logs`` table already migrated in
Milestone C, see app.models.audit.AuditLogRecord) is Milestone F's job.
``AuditLogEntry``'s fields are deliberately named/shaped to match
``AuditLogRecord`` closely (request_id, patient_id, question, model,
retrieved_document_ids, output_status, evidence_sufficiency, confidence,
latency_ms, safety_flags, hallucination_rate, citation_correctness) so a
real writer is a near-direct field mapping, not a redesign.

Only structured, audit-safe data goes on this entry - concise reason
strings (e.g. "Insufficient evidence caps confidence below HIGH") are
fine; the system prompt, retrieved evidence text, or any LLM chain-of-
thought are never stored here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class AuditLogEntry:
    request_id: str
    patient_id: str | None
    question: str
    question_category: str | None
    model: str | None
    retrieved_document_ids: list[str] = field(default_factory=list)
    output_status: str = "unknown"
    evidence_sufficiency: str | None = None
    confidence: float | None = None
    confidence_label: str | None = None
    latency_ms: float | None = None
    safety_flags: list[str] = field(default_factory=list)
    hallucination_rate: float | None = None
    citation_correctness: float | None = None
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)
    # Request actor (Phase 7): who asked. Defaults to None for
    # anonymous/legacy callers (pre-Phase 7 graph runs, tests that
    # build entries by hand). The writer maps these to
    # AuditLogRecord.user_id/role.
    actor_user_id: uuid.UUID | None = None
    actor_role: str | None = None


@runtime_checkable
class AuditLogWriter(Protocol):
    async def write(self, entry: AuditLogEntry) -> None:
        ...


class NullAuditLogWriter:
    """Default no-op writer. Satisfies the protocol so
    app.clinical.graph never needs redesigning when a real, persistent
    writer (Milestone F) is plugged in - only the injected writer
    changes."""

    async def write(self, entry: AuditLogEntry) -> None:
        return None


class InMemoryAuditLogWriter:
    """Test-only writer that records entries for assertions."""

    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    async def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)
