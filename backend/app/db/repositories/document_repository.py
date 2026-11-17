"""Persistence for ingested guideline documents and their chunks
(Milestone F, section 10).

Before this milestone, nothing wrote ``ClinicalDocumentRecord``/
``DocumentChunkRecord`` rows at all - ``app.retrieval.cli`` only ever
wrote a file-based manifest (see app.retrieval.manifest). This is the
first real DB-write path for ingested documents, deliberately scoped to
*just* document+chunk persistence: it does not embed chunks or write
``DocumentEmbeddingRecord`` rows (that stays
``app.services.embedding_persistence.EmbeddingPersistenceService``'s
job, and is not triggered by ``POST /documents/ingest`` - no OpenAI call
happens from this endpoint).

Upserts by primary key (``document_id`` / ``chunk_id``) so re-ingesting
an already-registered source updates its row rather than duplicating it
- ``chunk_id`` is already a deterministic, content-fingerprinted ID (see
app.retrieval.chunking.build_chunk_id), so this is naturally idempotent.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import ClinicalDocumentRecord, DocumentChunkRecord
from app.schemas.document import ClinicalDocument, DocumentChunk


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_document_and_chunks(
        self,
        document: ClinicalDocument,
        chunks: list[DocumentChunk],
        *,
        sha256: str | None = None,
    ) -> ClinicalDocumentRecord:
        metadata = document.metadata
        existing = await self._session.get(
            ClinicalDocumentRecord, metadata.document_id
        )

        if existing is None:
            record = ClinicalDocumentRecord(
                document_id=metadata.document_id,
                title=metadata.title,
                organization=metadata.organization,
                source_url=str(metadata.source_url),
                category=metadata.category.value,
                version=metadata.version,
                publication_date=metadata.publication_date,
                access_date=metadata.access_date,
                license_or_usage_note=metadata.license_or_usage_note,
                sha256=sha256,
            )
            self._session.add(record)
        else:
            existing.title = metadata.title
            existing.organization = metadata.organization
            existing.source_url = str(metadata.source_url)
            existing.category = metadata.category.value
            existing.version = metadata.version
            existing.publication_date = metadata.publication_date
            existing.access_date = metadata.access_date
            existing.license_or_usage_note = metadata.license_or_usage_note
            existing.sha256 = sha256
            record = existing

        for chunk in chunks:
            chunk_record = await self._session.get(
                DocumentChunkRecord, chunk.chunk_id
            )
            if chunk_record is None:
                self._session.add(
                    DocumentChunkRecord(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        section_id=chunk.section_id,
                        section_heading=chunk.section_heading,
                        chunk_number=chunk.chunk_number,
                        text=chunk.text,
                        token_count=chunk.token_count,
                    )
                )
            # A chunk_id match with a different (document_id, chunk_number,
            # text) is content-address-impossible by construction (chunk_id
            # embeds a content fingerprint - see build_chunk_id), so an
            # existing row is left untouched rather than rewritten.

        return record

    async def get_document(self, document_id: str) -> ClinicalDocumentRecord | None:
        return await self._session.get(ClinicalDocumentRecord, document_id)

    async def list_documents_with_chunk_counts(
        self,
    ) -> list[tuple[ClinicalDocumentRecord, int]]:
        chunk_count = (
            select(
                DocumentChunkRecord.document_id,
                func.count(DocumentChunkRecord.chunk_id).label("chunk_count"),
            )
            .group_by(DocumentChunkRecord.document_id)
            .subquery()
        )

        stmt = select(
            ClinicalDocumentRecord,
            func.coalesce(chunk_count.c.chunk_count, 0),
        ).outerjoin(
            chunk_count,
            ClinicalDocumentRecord.document_id == chunk_count.c.document_id,
        ).order_by(ClinicalDocumentRecord.document_id)

        result = await self._session.execute(stmt)
        return [(row[0], row[1]) for row in result.all()]
