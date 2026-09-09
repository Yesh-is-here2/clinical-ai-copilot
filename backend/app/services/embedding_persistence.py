"""Embedding persistence: chunk -> vector -> stored row.

This module is deliberately split into a small ``EmbeddingRepository``
boundary and an ``EmbeddingPersistenceService`` that only knows about that
boundary, not about SQLAlchemy or ORM rows directly (section 9 of the
spec: "Do not mix database ORM rows directly into all retrieval/business
logic. Use repository/service boundaries."). That's what lets the service
be fully unit tested with an in-memory fake repository - no Postgres
required - while the real ``SqlAlchemyEmbeddingRepository`` is exercised
only by the integration tests that require a live database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from app.schemas.document import DocumentChunk
from app.services.embeddings import EmbeddingProvider


@dataclass(frozen=True)
class EmbeddingRecordInput:
    """One embedding ready to be persisted."""

    chunk_id: str
    model: str
    dimensions: int
    embedding: list[float]


class EmbeddingRepository(Protocol):
    """Persistence boundary the embedding service depends on.

    Implementations only need to support two operations: find out which
    (chunk_id, model) pairs are already embedded (so re-ingestion can
    skip them), and upsert a batch of new/updated embeddings.
    """

    async def get_existing_chunk_ids(
        self, *, model: str, chunk_ids: Sequence[str]
    ) -> set[str]:
        ...

    async def upsert_many(self, records: Sequence[EmbeddingRecordInput]) -> int:
        """Persist ``records`` (insert new, update changed) and return
        how many rows were written."""
        ...


@dataclass(frozen=True)
class EmbeddingPersistenceResult:
    embedded_count: int
    skipped_existing_count: int
    total_chunks: int
    skipped_chunk_ids: list[str] = field(default_factory=list)


class EmbeddingPersistenceService:
    """Embeds document chunks and persists the resulting vectors.

    Duplicate/re-ingestion handling is deliberate, not accidental: by
    default, a chunk whose (chunk_id, model) pair is already persisted is
    skipped rather than re-embedded, because chunk IDs are content
    fingerprinted (see app.retrieval.chunking) - if the text changed, the
    chunk_id changed too, so an existing (chunk_id, model) row really is
    still current. Pass ``force=True`` to re-embed regardless (e.g. after
    a deliberate backfill).
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        repository: EmbeddingRepository,
        *,
        batch_size: int = 100,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self._provider = provider
        self._repository = repository
        self._batch_size = batch_size

    async def embed_and_persist_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        force: bool = False,
    ) -> EmbeddingPersistenceResult:
        if not chunks:
            return EmbeddingPersistenceResult(
                embedded_count=0,
                skipped_existing_count=0,
                total_chunks=0,
            )

        model = self._provider.model

        if force:
            chunks_to_embed = list(chunks)
            skipped_ids: list[str] = []
        else:
            existing = await self._repository.get_existing_chunk_ids(
                model=model,
                chunk_ids=[chunk.chunk_id for chunk in chunks],
            )
            chunks_to_embed = [
                chunk for chunk in chunks if chunk.chunk_id not in existing
            ]
            skipped_ids = [
                chunk.chunk_id for chunk in chunks if chunk.chunk_id in existing
            ]

        embedded_count = 0

        for batch_start in range(0, len(chunks_to_embed), self._batch_size):
            batch = chunks_to_embed[batch_start : batch_start + self._batch_size]

            if not batch:
                continue

            vectors = await self._provider.embed_batch(
                [chunk.text for chunk in batch]
            )

            records = [
                EmbeddingRecordInput(
                    chunk_id=chunk.chunk_id,
                    model=model,
                    dimensions=self._provider.dimensions,
                    embedding=vector,
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            ]

            embedded_count += await self._repository.upsert_many(records)

        return EmbeddingPersistenceResult(
            embedded_count=embedded_count,
            skipped_existing_count=len(skipped_ids),
            total_chunks=len(chunks),
            skipped_chunk_ids=skipped_ids,
        )
