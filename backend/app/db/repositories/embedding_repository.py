"""Real (PostgreSQL-backed) implementation of ``EmbeddingRepository``.

This is the only place ``DocumentEmbeddingRecord`` ORM rows and SQLAlchemy
statements appear for the embedding-persistence path - everything above
this layer (``EmbeddingPersistenceService``) talks only to the
``EmbeddingRepository`` protocol. Exercised by integration tests only
(see tests/integration/test_vector_search_integration.py) - it is not
covered by the default (no-Postgres) test run beyond a query-construction
smoke test in tests/test_embedding_persistence.py.
"""

from __future__ import annotations

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import DEFAULT_EMBEDDING_DIMENSIONS
from app.models.document import DocumentEmbeddingRecord
from app.services.embedding_persistence import EmbeddingRecordInput


class EmbeddingDimensionMismatchError(ValueError):
    """Raised before persistence is attempted, when an embedding's
    dimensionality can't actually be stored.

    ``document_embeddings.embedding`` is a fixed ``VECTOR(N)`` column,
    with N baked in at migration time as
    ``app.core.constants.DEFAULT_EMBEDDING_DIMENSIONS`` - it is a schema
    decision, not something that can vary per row. Without this check, a
    mismatched vector would still be caught (pgvector enforces the
    column's dimensionality), but only after a round trip to the
    database, as an opaque asyncpg ``DataError`` ("expected N
    dimensions, not M"). Raising here instead gives the same protection
    with a clear, actionable message, from code that already knows both
    numbers.
    """


def _validate_embedding_dimensions(record: EmbeddingRecordInput) -> None:
    if record.dimensions != DEFAULT_EMBEDDING_DIMENSIONS:
        raise EmbeddingDimensionMismatchError(
            f"Embedding for chunk '{record.chunk_id}' (model "
            f"'{record.model}') declares {record.dimensions} dimensions, "
            f"but document_embeddings.embedding is a "
            f"VECTOR({DEFAULT_EMBEDDING_DIMENSIONS}) column "
            "(app.core.constants.DEFAULT_EMBEDDING_DIMENSIONS). Changing "
            "the embedding model/provider to one with a different output "
            "size requires a new migration, not just a different "
            "provider configuration."
        )

    if len(record.embedding) != record.dimensions:
        raise EmbeddingDimensionMismatchError(
            f"Embedding for chunk '{record.chunk_id}' (model "
            f"'{record.model}') declares {record.dimensions} dimensions, "
            f"but the actual vector has {len(record.embedding)} values - "
            "the embedding provider produced output inconsistent with "
            "what it reported."
        )


class SqlAlchemyEmbeddingRepository:
    """``EmbeddingRepository`` backed by an ``AsyncSession``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_existing_chunk_ids(
        self, *, model: str, chunk_ids: Sequence[str]
    ) -> set[str]:
        if not chunk_ids:
            return set()

        stmt = select(DocumentEmbeddingRecord.chunk_id).where(
            DocumentEmbeddingRecord.model == model,
            DocumentEmbeddingRecord.chunk_id.in_(chunk_ids),
        )
        result = await self._session.execute(stmt)

        return set(result.scalars().all())

    async def upsert_many(self, records: Sequence[EmbeddingRecordInput]) -> int:
        if not records:
            return 0

        for record in records:
            _validate_embedding_dimensions(record)

        values = [
            {
                "chunk_id": record.chunk_id,
                "model": record.model,
                "dimensions": record.dimensions,
                "embedding": record.embedding,
            }
            for record in records
        ]

        # (chunk_id, model) has a unique constraint (uq_embedding_chunk_model)
        # - ON CONFLICT lets re-ingestion overwrite a stale vector for the
        # same chunk/model pair without a separate SELECT-then-UPDATE
        # round trip.
        stmt = pg_insert(DocumentEmbeddingRecord).values(values)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_embedding_chunk_model",
            set_={
                "embedding": stmt.excluded.embedding,
                "dimensions": stmt.excluded.dimensions,
            },
        )

        await self._session.execute(stmt)
        await self._session.commit()

        return len(records)
