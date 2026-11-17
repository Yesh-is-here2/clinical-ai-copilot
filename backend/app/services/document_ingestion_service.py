"""Orchestrates ``POST /documents/ingest`` (Milestone F, section 10).

Deliberately reuses, rather than reimplements, every piece of Milestone
C's ingestion pipeline: ``app.retrieval.sources.get_guideline_source``
for allowlist lookup, ``app.retrieval.ingestion.ingest_guideline`` for
download+extraction (which is where the SSRF/allowed-hosts protection
already lives - see ``GuidelineSource.allowed_hosts`` and
``app.retrieval.downloader``), and ``app.retrieval.chunking.
chunk_document`` for deterministic chunking. This service's only new
contribution is the DB-persistence step
(``app.db.repositories.document_repository.DocumentRepository``), which
did not exist before this milestone.

Never triggers embedding generation - no OpenAI call happens anywhere
in this module. That stays a separate, explicit step
(``app.services.embedding_persistence.EmbeddingPersistenceService``,
Milestone C/D), out of scope for this endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.document_repository import DocumentRepository
from app.retrieval.chunking import DEFAULT_CHUNKING_CONFIG, chunk_document
from app.retrieval.ingestion import GuidelineIngestionError, ingest_guideline
from app.retrieval.sources import GuidelineSource, get_guideline_source


class DocumentIngestionServiceError(RuntimeError):
    """Base class for all typed failures this service can raise."""


class UnknownGuidelineSourceError(DocumentIngestionServiceError):
    """``document_id`` does not match any registered
    ``GuidelineSource``. Maps to 404."""


class GuidelineIngestionFailedError(DocumentIngestionServiceError):
    """Download or extraction failed for a *known, registered* source -
    a genuine upstream/network/content problem, not a client error.
    Maps to 502."""


class DocumentPersistenceError(DocumentIngestionServiceError):
    """The document was successfully ingested but could not be
    persisted (e.g. the database is unreachable). Maps to 503."""


@dataclass(frozen=True)
class DocumentIngestionResult:
    document_id: str
    status: str
    chunk_count: int
    section_count: int
    warnings: list[str] = field(default_factory=list)


class DocumentIngestionService:
    def __init__(self, *, session: AsyncSession) -> None:
        self._session = session

    async def ingest(self, document_id: str) -> DocumentIngestionResult:
        source: GuidelineSource | None = get_guideline_source(document_id)

        if source is None:
            raise UnknownGuidelineSourceError(
                f"'{document_id}' is not a registered guideline source."
            )

        try:
            result = await ingest_guideline(source)
        except GuidelineIngestionError as exc:
            raise GuidelineIngestionFailedError(
                f"Ingestion failed for '{document_id}': {exc}"
            ) from exc

        chunks = chunk_document(result.document, DEFAULT_CHUNKING_CONFIG)

        try:
            repository = DocumentRepository(self._session)
            await repository.upsert_document_and_chunks(
                result.document, chunks, sha256=result.provenance.sha256
            )
            await self._session.commit()
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise DocumentPersistenceError(
                f"'{document_id}' was ingested but could not be persisted."
            ) from exc

        return DocumentIngestionResult(
            document_id=result.document.metadata.document_id,
            status=result.status.value,
            chunk_count=len(chunks),
            section_count=result.provenance.section_count,
        )
