import uuid
from datetime import date

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import DEFAULT_EMBEDDING_DIMENSIONS
from app.db.base import Base
from app.models.mixins import TimestampMixin


class ClinicalDocumentRecord(TimestampMixin, Base):
    """Persisted counterpart of ``app.schemas.document.ClinicalDocument``
    (the ingested-and-extracted guideline). ``document_id`` mirrors the
    registered ``GuidelineSource.document_id`` so ingestion is idempotent:
    re-ingesting a source updates this row rather than duplicating it."""

    __tablename__ = "clinical_documents"

    document_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    organization: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    publication_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    access_date: Mapped[date] = mapped_column(Date, nullable=False)
    license_or_usage_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    chunks: Mapped[list["DocumentChunkRecord"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"ClinicalDocumentRecord(document_id={self.document_id!r})"


class DocumentChunkRecord(TimestampMixin, Base):
    """Persisted counterpart of ``app.schemas.document.DocumentChunk``.

    ``chunk_id`` is the deterministic ID produced by
    ``app.retrieval.chunking.build_chunk_id`` - it is the primary key here
    too, which is what makes re-chunking idempotent (upsert by chunk_id)
    rather than accumulating duplicates on every re-ingestion.
    """

    __tablename__ = "document_chunks"

    chunk_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("clinical_documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    section_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    section_heading: Mapped[str | None] = mapped_column(String(512), nullable=True)
    chunk_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    document: Mapped["ClinicalDocumentRecord"] = relationship(back_populates="chunks")
    embeddings: Mapped[list["DocumentEmbeddingRecord"]] = relationship(
        back_populates="chunk",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id", "chunk_number", name="uq_chunk_document_number"
        ),
        Index("ix_document_chunks_document_chunk_number", "document_id", "chunk_number"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"DocumentChunkRecord(chunk_id={self.chunk_id!r})"


class DocumentEmbeddingRecord(TimestampMixin, Base):
    """One embedding vector for one chunk, under one embedding model.

    Keyed by ``(chunk_id, model)`` rather than ``chunk_id`` alone so a
    future embedding-model change (or an MLflow chunking-config
    comparison run) can coexist with embeddings from a previous model
    without deleting them first - see section 32 of the project spec on
    comparing chunk/retrieval configurations.
    """

    __tablename__ = "document_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    chunk_id: Mapped[str] = mapped_column(
        String(256),
        ForeignKey("document_chunks.chunk_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)

    # Fixed-dimension pgvector column. The dimension is a schema decision
    # (see app.core.constants.DEFAULT_EMBEDDING_DIMENSIONS) - changing the
    # configured embedding model to one with a different output size
    # requires a new migration, not just an env var change.
    embedding: Mapped[list[float]] = mapped_column(
        Vector(DEFAULT_EMBEDDING_DIMENSIONS), nullable=False
    )

    chunk: Mapped["DocumentChunkRecord"] = relationship(back_populates="embeddings")

    __table_args__ = (
        UniqueConstraint("chunk_id", "model", name="uq_embedding_chunk_model"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return (
            f"DocumentEmbeddingRecord(chunk_id={self.chunk_id!r}, "
            f"model={self.model!r})"
        )
