import uuid

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.mixins import TimestampMixin


class AuditLogRecord(TimestampMixin, Base):
    """One immutable audit record for one clinical query (section 27).

    ``patient_id`` is intentionally a plain indexed string, not a foreign
    key: an audit trail must remain complete and queryable even if the
    referenced patient record were ever removed, so it must never be
    affected by a patient-table cascade. Never store secrets/API
    keys/passwords/raw auth headers here - see section 27 and section 47.
    """

    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    patient_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    retrieved_document_ids: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list
    )
    output_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    evidence_sufficiency: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    safety_flags: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    hallucination_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_correctness: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (Index("ix_audit_logs_request_id_created", "request_id", "created_at"),)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"AuditLogRecord(request_id={self.request_id!r})"
