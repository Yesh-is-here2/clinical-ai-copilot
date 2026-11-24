"""Integration tests requiring a live PostgreSQL + pgvector database.

Excluded from the default test run by backend/pytest.ini
(`addopts = -m "not integration"`). Run explicitly with:

    docker compose exec backend pytest -m integration -q

These are NOT run in the sandbox this code was developed in (no route to
a real Postgres instance there) - do not report them as passing, and do
not weaken/skip-with-a-mock any of the assertions below to make that
happen. See LOCAL_INTEGRATION_PENDING.md.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.constants import DEFAULT_EMBEDDING_DIMENSIONS
from app.db.repositories.embedding_repository import SqlAlchemyEmbeddingRepository
from app.models.document import (
    ClinicalDocumentRecord,
    DocumentChunkRecord,
    DocumentEmbeddingRecord,
)
from app.models.query import QueryRecord
from app.retrieval.vector_search import VectorSearchService
from app.services.embedding_persistence import (
    EmbeddingPersistenceService,
    EmbeddingRecordInput,
)
from app.services.embeddings import FakeDeterministicEmbeddingProvider

# Two markers on every test in this module:
#   - `integration`  -> excluded by default (backend/pytest.ini), run
#     explicitly with `pytest -m integration`.
#   - `anyio`        -> required by the project's existing async test
#     plugin (anyio's pytest plugin, already used by every other async
#     test in this suite - see tests/conftest.py's `anyio_backend`
#     fixture and e.g. tests/test_guideline_ingestion.py). Async `def
#     test_...` functions are not natively runnable by pytest itself;
#     something has to claim them, and this project's convention is
#     anyio, not pytest-asyncio - adding a second, competing async
#     plugin would risk exactly the kind of marker/collection ambiguity
#     this fix is removing.
pytestmark = [pytest.mark.integration, pytest.mark.anyio]


# Every row this test module creates uses a document_id starting with this
# prefix, and nothing else in the (real, migrated) database ever should -
# that's what makes cleanup safe to run unconditionally against the same
# database docker-compose uses for the app itself, without touching
# unrelated development data.
_TEST_DOCUMENT_ID_PREFIX = "itest-"


async def _delete_test_owned_rows(session: AsyncSession) -> None:
    """Delete only rows this test module could have created.

    `document_chunks` and `document_embeddings` both have
    `ON DELETE CASCADE` foreign keys back to `clinical_documents` (see
    app/models/document.py), so deleting the parent row is sufficient -
    no separate chunk/embedding cleanup, and no risk of leaving orphans.

    Queries go FIRST: citations.chunk_id is ondelete=RESTRICT by design,
    so a leftover citation from another module's run blocks deleting
    its chunk. Deleting itest- queries cascades through
    generated_responses to citations, and only then is it safe to
    delete the documents.

    Deliberately does NOT drop/recreate any table, downgrade Alembic, or
    touch anything outside the `itest-` id space: the schema here is the
    same one `docker compose exec backend alembic upgrade head` already
    created, and this suite must not assume ownership of it.
    """
    await session.execute(
        delete(QueryRecord).where(
            QueryRecord.request_id.like(f"{_TEST_DOCUMENT_ID_PREFIX}%")
        )
    )
    await session.execute(
        delete(ClinicalDocumentRecord).where(
            ClinicalDocumentRecord.document_id.like(f"{_TEST_DOCUMENT_ID_PREFIX}%")
        )
    )
    await session.commit()


@pytest.fixture
async def db_session():
    """A Postgres session whose engine is scoped to this one test.

    Deliberately builds its own ``AsyncEngine`` here rather than reusing
    the application's module-global ``app.db.session.engine``: that
    engine is constructed once (at import time) and pools real asyncpg
    connections. AnyIO gives each test function its own event loop, so a
    second integration test reusing connections pooled under a
    *different* test's loop fails with "Future attached to a different
    loop" / "Event loop is closed" - not a Postgres problem, a
    pool-lifetime-vs-event-loop-lifetime mismatch.

    ``poolclass=NullPool`` means every checkout opens a real connection
    and every checkin closes it - nothing is ever pooled, so nothing can
    outlive the event loop it was created on. This engine, and NullPool,
    are scoped to this fixture only; the application's real engine
    (``app.db.session.engine``) keeps its normal pooled configuration
    for production use.
    """
    test_engine = create_async_engine(
        settings.database_url,
        poolclass=NullPool,
        echo=False,
    )
    test_session_factory = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with test_session_factory() as session:
        # Clean up first too, in case a previous run crashed mid-test and
        # left rows behind - tests must not depend on run order or on the
        # previous run having exited cleanly.
        await _delete_test_owned_rows(session)
        try:
            yield session
        finally:
            await _delete_test_owned_rows(session)

    await test_engine.dispose()


async def _seed_document_and_chunk(session: AsyncSession) -> str:
    document = ClinicalDocumentRecord(
        document_id="itest-doc",
        title="Integration Test Guideline",
        organization="Test Org",
        source_url="https://example.org/guideline",
        category="hypertension",
        access_date=date(2026, 9, 9),
    )
    chunk = DocumentChunkRecord(
        chunk_id="itest-doc:s1:0:abc123",
        document_id="itest-doc",
        section_id="s1",
        section_heading="Overview",
        chunk_number=0,
        text="Confirm hypertension diagnosis with ambulatory monitoring.",
        token_count=8,
    )
    session.add(document)
    session.add(chunk)
    await session.commit()

    return chunk.chunk_id


class TestPgvectorSemanticSearch:
    async def test_embedding_roundtrip_and_cosine_search(
        self, db_session: AsyncSession
    ) -> None:
        chunk_id = await _seed_document_and_chunk(db_session)

        # `document_embeddings.embedding` is a real, fixed VECTOR(1536)
        # column (see app/models/document.py /
        # app.core.constants.DEFAULT_EMBEDDING_DIMENSIONS) - the fake
        # provider must match it exactly, or PostgreSQL correctly rejects
        # the insert ("expected 1536 dimensions, not N"). A 1536-element
        # fake vector is still fully offline/deterministic; it does not
        # require OpenAI or network access.
        provider = FakeDeterministicEmbeddingProvider(
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS
        )
        repository = SqlAlchemyEmbeddingRepository(db_session)

        vector = (await provider.embed_batch(["query text"]))[0]

        await repository.upsert_many(
            [
                EmbeddingRecordInput(
                    chunk_id=chunk_id,
                    model=provider.model,
                    dimensions=provider.dimensions,
                    embedding=vector,
                )
            ]
        )

        stored = (
            await db_session.execute(
                select(DocumentEmbeddingRecord).where(
                    DocumentEmbeddingRecord.chunk_id == chunk_id
                )
            )
        ).scalar_one()

        assert stored.dimensions == DEFAULT_EMBEDDING_DIMENSIONS
        assert len(stored.embedding) == DEFAULT_EMBEDDING_DIMENSIONS

        service = VectorSearchService(db_session, model=provider.model, default_top_k=5)
        results = await service.search(vector, top_k=5)

        assert results
        assert results[0].chunk_id == chunk_id
        # An exact-match query vector should be (near) zero cosine
        # distance from itself, i.e. score close to 1.0.
        assert results[0].score > 0.99

    async def test_embedding_persistence_service_skips_existing_on_reingestion(
        self, db_session: AsyncSession
    ) -> None:
        from app.retrieval.chunking import chunk_document, DEFAULT_CHUNKING_CONFIG
        from app.schemas.document import (
            ClinicalDocument,
            ClinicalDocumentCategory,
            ClinicalDocumentMetadata,
            ClinicalSection,
        )

        document = ClinicalDocument(
            metadata=ClinicalDocumentMetadata(
                document_id="itest-doc-2",
                title="Doc 2",
                organization="Test Org",
                source_url="https://example.org/guideline2",
                access_date=date(2026, 9, 9),
                category=ClinicalDocumentCategory.HYPERTENSION,
            ),
            sections=[
                ClinicalSection(
                    section_id="s1",
                    heading="Overview",
                    text="Evidence text for re-ingestion test.",
                    order=0,
                )
            ],
        )
        chunks = chunk_document(document, DEFAULT_CHUNKING_CONFIG)

        provider = FakeDeterministicEmbeddingProvider(
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS
        )
        repository = SqlAlchemyEmbeddingRepository(db_session)
        service = EmbeddingPersistenceService(provider, repository)

        parent = ClinicalDocumentRecord(
            document_id=document.metadata.document_id,
            title=document.metadata.title,
            organization=document.metadata.organization,
            source_url=str(document.metadata.source_url),
            category=document.metadata.category.value,
            access_date=document.metadata.access_date,
        )
        db_session.add(parent)
        for chunk in chunks:
            db_session.add(
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
        await db_session.commit()

        first_run = await service.embed_and_persist_chunks(chunks)
        assert first_run.embedded_count == len(chunks)

        second_run = await service.embed_and_persist_chunks(chunks)
        assert second_run.embedded_count == 0
        assert second_run.skipped_existing_count == len(chunks)
