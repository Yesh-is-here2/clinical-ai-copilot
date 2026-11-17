"""``GET /documents`` and ``POST /documents/ingest`` (Milestone F,
section 10).

``POST /documents/ingest`` only ever accepts a ``document_id`` that must
resolve to an entry in ``app.retrieval.sources.GUIDELINE_SOURCES`` -
never an arbitrary URL. That, plus reusing
``app.retrieval.ingestion.ingest_guideline`` unchanged (which is where
the actual SSRF/allowed-hosts enforcement lives - see
``GuidelineSource.allowed_hosts`` / ``app.retrieval.downloader``), is
what preserves Phase 3's source allowlisting without this route
re-implementing or weakening it.

``get_current_actor`` is wired in as a placeholder (see app.api.deps)
so Phase 7 can add a real "must be an admin" check to this route later
without restructuring it - this milestone deliberately does not enforce
anything based on it.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import (
    Actor,
    get_current_actor,
    get_document_ingestion_service,
    get_document_repository,
)
from app.db.repositories.document_repository import DocumentRepository
from app.schemas.clinical import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentSummary,
)
from app.services.document_ingestion_service import (
    DocumentIngestionService,
    DocumentPersistenceError,
    GuidelineIngestionFailedError,
    UnknownGuidelineSourceError,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get(
    "",
    response_model=list[DocumentSummary],
    summary="List ingested clinical guideline documents",
)
async def list_documents(
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
) -> list[DocumentSummary]:
    rows = await repository.list_documents_with_chunk_counts()

    return [
        DocumentSummary(
            document_id=document.document_id,
            title=document.title,
            organization=document.organization,
            category=document.category,
            version=document.version,
            publication_date=document.publication_date,
            access_date=document.access_date,
            chunk_count=chunk_count,
        )
        for document, chunk_count in rows
    ]


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    summary="Ingest one registered guideline source and persist its chunks",
    description=(
        "Administrative capability, to be protected by RBAC in Phase 7. "
        "`document_id` must be a pre-registered, allowlisted guideline "
        "source (see app.retrieval.sources.GUIDELINE_SOURCES) - arbitrary "
        "URLs are never accepted. Does not generate or persist "
        "embeddings; that remains a separate step."
    ),
)
async def ingest_document(
    payload: DocumentIngestRequest,
    service: Annotated[DocumentIngestionService, Depends(get_document_ingestion_service)],
    _actor: Annotated[Actor | None, Depends(get_current_actor)],
) -> DocumentIngestResponse:
    try:
        result = await service.ingest(payload.document_id)
    except UnknownGuidelineSourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except GuidelineIngestionFailedError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc
    except DocumentPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return DocumentIngestResponse(
        document_id=result.document_id,
        status=result.status,
        chunk_count=result.chunk_count,
        section_count=result.section_count,
        warnings=result.warnings,
    )
