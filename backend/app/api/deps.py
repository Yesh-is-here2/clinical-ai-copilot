"""FastAPI dependency providers (Milestone F, section 12/13).

Every dependency in this module is lazy and import-safe: constructing
any of them (and importing this module) never touches the network,
never loads a CrossEncoder model, and never requires ``OPENAI_API_KEY``
to be set - only the moment a *real* call is attempted inside
``OpenAIEmbeddingProvider``/``OpenAIStructuredLLMProvider``/
``CrossEncoderReranker`` does that. This is what lets ``GET /health`` and
every non-generation route keep working with no API key configured
(section 13), and what lets every dependency below be trivially
overridden with a deterministic fake in tests (section 12/20) via
FastAPI's ``app.dependency_overrides``.

``get_hybrid_retrieval_service`` is the one provider that does real,
non-trivial work: it loads every currently-indexed chunk from Postgres
and builds a fresh in-memory ``BM25Index`` from them on every request.
This is a deliberate, documented scope decision for Milestone F (see its
docstring below) - a production system would cache/persist that index
rather than rebuild it per request, but that is a performance concern
out of scope for "does the API work correctly end-to-end", which is
this milestone's job.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.clinical.audit import AuditLogWriter
from app.clinical.graph import (
    ClinicalGraphDependencies,
    confidence_config_from_settings,
    evidence_sufficiency_config_from_settings,
)
from app.clinical.llm import OpenAIStructuredLLMProvider, StructuredLLMProvider
from app.core.config import Settings, settings as _default_settings
from app.core.security import TokenValidationError, decode_access_token
from app.db.repositories.audit_repository import (
    AuditLogRepository,
    SqlAlchemyAuditLogWriter,
)
from app.db.repositories.document_repository import DocumentRepository
from app.db.repositories.evaluation_repository import EvaluationRepository
from app.db.repositories.query_repository import ClinicalQueryRepository
from app.db.repositories.user_repository import UserRepository
from app.db.session import get_db_session
from app.models.document import ClinicalDocumentRecord, DocumentChunkRecord
from app.models.user import User, UserRole
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import HybridRetrievalService
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_search import VectorSearchService
from app.schemas.document import DocumentChunk
from app.services.clinical_query_service import ClinicalQueryService
from app.services.document_ingestion_service import DocumentIngestionService
from app.services.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
    get_embedding_provider,
)
from app.services.patient import PatientService, patient_service as _default_patient_service


# --- Cross-cutting -----------------------------------------------------


def get_settings() -> Settings:
    return _default_settings


def get_patient_service() -> PatientService:
    return _default_patient_service


async def get_db(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> AsyncGenerator[AsyncSession, None]:
    """Thin alias over ``app.db.session.get_db_session`` so every route
    in this package depends on one name from ``app.api.deps`` - the
    single place tests override the DB boundary."""

    yield session


_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Actor:
    """The authenticated identity carried through a request (Phase 7).

    Built once per request by ``get_current_actor`` from a validated
    token + a live database row, so a role change or deactivation
    takes effect on the next request - the token's role claim is
    never trusted on its own. (``user_id`` is the user's UUID.)"""

    user_id: uuid.UUID
    email: str
    role: UserRole


def _unauthorized(detail: str) -> HTTPException:
    """401 with a WWW-Authenticate header, as Bearer auth expects."""

    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)
    ],
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Validate the Bearer token and return the live user row.

    Raises 401 when no token is presented, the token is expired,
    malformed, or wrongly signed, or the user does not exist or is
    deactivated. The token's email/role claims are informational
    only - the returned row is the authority."""

    if credentials is None:
        raise _unauthorized("Not authenticated.")
    try:
        payload = decode_access_token(
            credentials.credentials,
            secret=settings.jwt_secret,
            algorithm=settings.jwt_algorithm,
        )
    except TokenValidationError as exc:
        raise _unauthorized(str(exc)) from exc
    user = await UserRepository(session).get_by_id(uuid.UUID(payload.subject))
    if user is None or not user.is_active:
        raise _unauthorized("User not found or inactive.")
    return user


async def get_current_actor(
    current_user: Annotated[User, Depends(get_current_user)],
) -> Actor:
    """The request actor: id, email, and *current database* role."""

    return Actor(
        user_id=current_user.id,
        email=current_user.email,
        role=current_user.role,
    )


def require_roles(*roles: UserRole) -> Callable[[Actor], Actor]:
    """Dependency factory: allow only the given roles.

    Authentication still runs first (401 for a bad/missing token);
    a valid token with the wrong role gets 403. Usage::

        @router.post("/documents/ingest")
        async def ingest(actor: Actor = Depends(require_roles(UserRole.ADMIN))):
            ...
    """

    async def _guard(
        actor: Annotated[Actor, Depends(get_current_actor)],
    ) -> Actor:
        if actor.role not in roles:
            allowed = ", ".join(role.value for role in roles)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {allowed}.",
            )
        return actor

    return _guard


# --- Repositories --------------------------------------------------------


def get_query_repository(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> ClinicalQueryRepository:
    return ClinicalQueryRepository(session)


def get_document_repository(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentRepository:
    return DocumentRepository(session)


def get_evaluation_repository(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> EvaluationRepository:
    return EvaluationRepository(session)


def get_audit_writer(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> AuditLogWriter:
    return SqlAlchemyAuditLogWriter(session)


def get_audit_repository(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> AuditLogRepository:
    """Read-only audit-log queries for GET /audit (Phase 7, section
    7.7)."""

    return AuditLogRepository(session)


# --- Retrieval / LLM (real application composition, section 13) --------


async def _load_indexed_chunks(session: AsyncSession) -> list[DocumentChunk]:
    stmt = select(DocumentChunkRecord, ClinicalDocumentRecord).join(
        ClinicalDocumentRecord,
        DocumentChunkRecord.document_id == ClinicalDocumentRecord.document_id,
    )
    rows = (await session.execute(stmt)).all()

    return [
        DocumentChunk(
            chunk_id=chunk_row.chunk_id,
            document_id=chunk_row.document_id,
            document_title=doc_row.title,
            organization=doc_row.organization,
            source_url=doc_row.source_url,
            category=doc_row.category,
            version=doc_row.version,
            publication_date=doc_row.publication_date,
            access_date=doc_row.access_date,
            section_id=chunk_row.section_id,
            section_heading=chunk_row.section_heading,
            chunk_number=chunk_row.chunk_number,
            text=chunk_row.text,
            token_count=chunk_row.token_count,
        )
        for chunk_row, doc_row in rows
    ]


async def get_hybrid_retrieval_service(
    session: Annotated[AsyncSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> HybridRetrievalService | None:
    """Build a real ``HybridRetrievalService`` from whatever is currently
    ingested, or ``None`` if there is nothing safe to retrieve from yet
    (no ingested chunks, and no configured embedding provider either).

    Returning ``None`` rather than raising lets the caller
    (``app.services.clinical_query_service.ClinicalQueryService``)
    decide how to fail (503, not an unhandled exception) - this
    dependency stays a pure "can we build one?" question.
    """

    chunks = await _load_indexed_chunks(session)
    bm25_index = BM25Index(chunks) if chunks else None

    embedding_provider: EmbeddingProvider | None = None
    vector_search: VectorSearchService | None = None

    if settings.openai_api_key:
        try:
            embedding_provider = get_embedding_provider()
        except EmbeddingProviderError:
            # Any provider-construction failure (not just a missing
            # key) degrades to BM25-only retrieval rather than failing
            # the whole request: vector search is one component, and
            # HybridRetrievalService already reports that degradation
            # explicitly. Never fall back to fake embeddings here.
            embedding_provider = None

        if embedding_provider is not None:
            vector_search = VectorSearchService(
                session,
                model=settings.embedding_model,
                default_top_k=settings.semantic_top_k,
            )

    if vector_search is None and bm25_index is None:
        return None

    reranker = (
        CrossEncoderReranker(model_name=settings.reranker_model)
        if settings.reranker_enabled
        else None
    )

    return HybridRetrievalService(
        vector_search=vector_search,
        embedding_provider=embedding_provider,
        bm25_index=bm25_index,
        reranker=reranker,
        reranker_enabled=settings.reranker_enabled,
        rrf_k=settings.rrf_k,
        semantic_top_k=settings.semantic_top_k,
        bm25_top_k=settings.bm25_top_k,
        reranker_top_k=settings.reranker_top_k,
    )


def get_llm_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> StructuredLLMProvider:
    """Always constructible with no API key (see ``OpenAIStructuredLLM
    Provider``'s own docstring) - a missing key only surfaces as
    ``LLMNotConfiguredError`` the moment a real generation call is
    attempted, which ``app.clinical.graph`` already turns into a safe
    ``error_generation_failed`` guarded response, never a 500."""

    return OpenAIStructuredLLMProvider(
        api_key=settings.openai_api_key,
        model=settings.generation_model,
        default_timeout_seconds=settings.generation_timeout_seconds,
    )


def get_clinical_graph_dependencies(
    hybrid_retrieval: Annotated[
        HybridRetrievalService | None, Depends(get_hybrid_retrieval_service)
    ],
    llm_provider: Annotated[StructuredLLMProvider, Depends(get_llm_provider)],
    audit_writer: Annotated[AuditLogWriter, Depends(get_audit_writer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ClinicalGraphDependencies | None:
    """``None`` when retrieval isn't configured yet (see
    ``get_hybrid_retrieval_service``) - the caller maps that to a clear
    503, never constructs a graph with nothing to retrieve from."""

    if hybrid_retrieval is None:
        return None

    return ClinicalGraphDependencies(
        hybrid_retrieval=hybrid_retrieval,
        llm_provider=llm_provider,
        audit_writer=audit_writer,
        evidence_sufficiency_config=evidence_sufficiency_config_from_settings(settings),
        confidence_config=confidence_config_from_settings(settings),
        generation_temperature=settings.generation_temperature,
        generation_timeout_seconds=settings.generation_timeout_seconds,
        max_regeneration_attempts=settings.max_regeneration_attempts,
    )


# --- Services (section 6/12) --------------------------------------------


def get_clinical_query_service(
    session: Annotated[AsyncSession, Depends(get_db)],
    patient_service: Annotated[PatientService, Depends(get_patient_service)],
    graph_dependencies: Annotated[
        ClinicalGraphDependencies | None, Depends(get_clinical_graph_dependencies)
    ],
) -> ClinicalQueryService:
    return ClinicalQueryService(
        session=session,
        patient_service=patient_service,
        graph_dependencies=graph_dependencies,
    )


def get_document_ingestion_service(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentIngestionService:
    return DocumentIngestionService(session=session)
