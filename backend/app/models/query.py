import uuid

from sqlalchemy import Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin


class QueryRecord(TimestampMixin, Base):
    """One clinical question submitted through ``POST /clinical/query``."""

    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    patient_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("patients.patient_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", server_default="pending"
    )
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    retrieval_results: Mapped[list["RetrievalResultRecord"]] = relationship(
        back_populates="query",
        cascade="all, delete-orphan",
    )
    generated_response: Mapped["GeneratedResponseRecord | None"] = relationship(
        back_populates="query",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"QueryRecord(id={self.id!r}, request_id={self.request_id!r})"


class RetrievalResultRecord(TimestampMixin, Base):
    """One retrieved chunk for one query, from one retrieval method.

    Stores the *fused/final* per-method view (bm25 / semantic / fused /
    reranked) so the evidence-inspector UI (section 6) can show exactly
    which retriever surfaced which evidence at what rank/score, per the
    project's auditability requirements.
    """

    __tablename__ = "retrieval_results"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    query_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("queries.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[str] = mapped_column(
        String(256),
        ForeignKey("document_chunks.chunk_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    retrieval_method: Mapped[str] = mapped_column(String(32), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    reranker_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    query: Mapped["QueryRecord"] = relationship(back_populates="retrieval_results")

    __table_args__ = (
        UniqueConstraint(
            "query_id",
            "chunk_id",
            "retrieval_method",
            name="uq_retrieval_result_identity",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return (
            f"RetrievalResultRecord(query_id={self.query_id!r}, "
            f"chunk_id={self.chunk_id!r}, method={self.retrieval_method!r})"
        )


class GeneratedResponseRecord(TimestampMixin, Base):
    """The grounded generation result for a query (section 20)."""

    __tablename__ = "generated_responses"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    query_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("queries.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    key_points: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    warnings: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    safety_flags: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    query: Mapped["QueryRecord"] = relationship(back_populates="generated_response")
    citations: Mapped[list["CitationRecord"]] = relationship(
        back_populates="generated_response",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"GeneratedResponseRecord(query_id={self.query_id!r})"


class CitationRecord(TimestampMixin, Base):
    """One [C1]/[C2]/... citation attached to a generated response
    (section 21). ``chunk_id`` uses ``ondelete=RESTRICT`` deliberately -
    a cited evidence chunk should not silently disappear out from under
    an already-generated, already-audited answer."""

    __tablename__ = "citations"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    generated_response_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("generated_responses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[str] = mapped_column(
        String(256),
        ForeignKey("document_chunks.chunk_id", ondelete="RESTRICT"),
        nullable=False,
    )
    citation_label: Mapped[str] = mapped_column(String(16), nullable=False)
    quoted_support: Mapped[str] = mapped_column(Text, nullable=False)
    support_classification: Mapped[str] = mapped_column(String(32), nullable=False)

    generated_response: Mapped["GeneratedResponseRecord"] = relationship(
        back_populates="citations"
    )

    __table_args__ = (
        UniqueConstraint(
            "generated_response_id",
            "citation_label",
            name="uq_citation_label_per_response",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"CitationRecord(label={self.citation_label!r}, chunk_id={self.chunk_id!r})"
