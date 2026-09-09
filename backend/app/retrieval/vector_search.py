"""pgvector-backed semantic search over indexed document chunks.

Uses cosine distance (pgvector's ``<=>`` operator via the SQLAlchemy
``Vector`` comparator's ``.cosine_distance()``), which is the standard
choice for OpenAI embeddings (they are not guaranteed unit-length, but
OpenAI explicitly recommends cosine similarity for its embedding models).
Results are ordered by ascending distance (closest first) and reported
back as a ``score`` in the more intuitive "higher is better" similarity
form (``1 - distance``).

The statement-building step (``build_vector_search_statement``) is kept
separate from ``VectorSearchService.search`` specifically so it can be
unit tested by compiling the SQL and inspecting it - no live database
required - while the service method itself is covered by the
Postgres-only integration test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import (
    ClinicalDocumentRecord,
    DocumentChunkRecord,
    DocumentEmbeddingRecord,
)


class VectorSearchError(ValueError):
    pass


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: str
    document_id: str
    document_title: str
    organization: str
    source_url: str
    section_id: str | None
    section_heading: str | None
    text: str
    score: float
    rank: int


def build_vector_search_statement(
    query_embedding: Sequence[float],
    *,
    model: str,
    top_k: int,
) -> Select:
    """Pure statement builder - no I/O, safe to unit test by compiling
    the resulting SQL and asserting on its shape."""

    if top_k <= 0:
        raise VectorSearchError("top_k must be greater than zero.")

    if not query_embedding:
        raise VectorSearchError("query_embedding must not be empty.")

    distance = DocumentEmbeddingRecord.embedding.cosine_distance(
        list(query_embedding)
    ).label("distance")

    return (
        select(
            DocumentEmbeddingRecord.chunk_id,
            DocumentChunkRecord.document_id,
            ClinicalDocumentRecord.title.label("document_title"),
            ClinicalDocumentRecord.organization,
            ClinicalDocumentRecord.source_url,
            DocumentChunkRecord.section_id,
            DocumentChunkRecord.section_heading,
            DocumentChunkRecord.text,
            distance,
        )
        .join(
            DocumentChunkRecord,
            DocumentEmbeddingRecord.chunk_id == DocumentChunkRecord.chunk_id,
        )
        .join(
            ClinicalDocumentRecord,
            DocumentChunkRecord.document_id == ClinicalDocumentRecord.document_id,
        )
        .where(DocumentEmbeddingRecord.model == model)
        # Stable tie-break: two chunks at the exact same distance (rare
        # but possible, e.g. near-duplicate content) always come back in
        # the same order rather than depending on physical row order.
        .order_by(distance.asc(), DocumentEmbeddingRecord.chunk_id.asc())
        .limit(top_k)
    )


class VectorSearchService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        model: str,
        default_top_k: int = 15,
    ) -> None:
        if default_top_k <= 0:
            raise VectorSearchError("default_top_k must be greater than zero.")

        self._session = session
        self._model = model
        self._default_top_k = default_top_k

    async def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int | None = None,
    ) -> list[VectorSearchResult]:
        resolved_top_k = top_k if top_k is not None else self._default_top_k

        stmt = build_vector_search_statement(
            query_embedding,
            model=self._model,
            top_k=resolved_top_k,
        )

        result = await self._session.execute(stmt)
        rows = result.all()

        return [
            VectorSearchResult(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                document_title=row.document_title,
                organization=row.organization,
                source_url=row.source_url,
                section_id=row.section_id,
                section_heading=row.section_heading,
                text=row.text,
                score=1.0 - row.distance,
                rank=index + 1,
            )
            for index, row in enumerate(rows)
        ]
