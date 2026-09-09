from datetime import date

import pytest

from app.retrieval.chunking import DEFAULT_CHUNKING_CONFIG, chunk_document
from app.schemas.document import (
    ClinicalDocument,
    ClinicalDocumentCategory,
    ClinicalDocumentMetadata,
    ClinicalSection,
)
from app.services.embedding_persistence import (
    EmbeddingPersistenceService,
    EmbeddingRecordInput,
)
from app.services.embeddings import FakeDeterministicEmbeddingProvider


class InMemoryEmbeddingRepository:
    """Fake implementing the EmbeddingRepository protocol - no database.

    This is exactly what the repository/service boundary (section 9) is
    for: the persistence service is tested against this fake, and the
    real SqlAlchemyEmbeddingRepository is exercised only by the
    Postgres-only integration test.
    """

    def __init__(self) -> None:
        self.stored: dict[tuple[str, str], EmbeddingRecordInput] = {}
        self.upsert_calls: list[list[EmbeddingRecordInput]] = []

    async def get_existing_chunk_ids(
        self, *, model: str, chunk_ids
    ) -> set[str]:
        return {
            chunk_id
            for (chunk_id, stored_model) in self.stored
            if stored_model == model and chunk_id in chunk_ids
        }

    async def upsert_many(self, records) -> int:
        records = list(records)
        self.upsert_calls.append(records)

        for record in records:
            self.stored[(record.chunk_id, record.model)] = record

        return len(records)


def make_document(document_id: str = "doc-1") -> ClinicalDocument:
    return ClinicalDocument(
        metadata=ClinicalDocumentMetadata(
            document_id=document_id,
            title="Test Guideline",
            organization="Example Org",
            source_url="https://example.org/guideline",
            access_date=date(2026, 9, 9),
            category=ClinicalDocumentCategory.HYPERTENSION,
        ),
        sections=[
            ClinicalSection(
                section_id="s1",
                heading="Overview",
                text=" ".join(f"word{i}" for i in range(30)),
                order=0,
            ),
        ],
    )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_embed_and_persist_new_chunks() -> None:
    chunks = chunk_document(make_document(), DEFAULT_CHUNKING_CONFIG)
    provider = FakeDeterministicEmbeddingProvider(dimensions=8)
    repository = InMemoryEmbeddingRepository()
    service = EmbeddingPersistenceService(provider, repository)

    result = await service.embed_and_persist_chunks(chunks)

    assert result.total_chunks == len(chunks)
    assert result.embedded_count == len(chunks)
    assert result.skipped_existing_count == 0
    assert len(repository.stored) == len(chunks)


@pytest.mark.anyio
async def test_reingesting_same_chunks_skips_existing() -> None:
    chunks = chunk_document(make_document(), DEFAULT_CHUNKING_CONFIG)
    provider = FakeDeterministicEmbeddingProvider(dimensions=8)
    repository = InMemoryEmbeddingRepository()
    service = EmbeddingPersistenceService(provider, repository)

    first = await service.embed_and_persist_chunks(chunks)
    second = await service.embed_and_persist_chunks(chunks)

    assert first.embedded_count == len(chunks)
    assert second.embedded_count == 0
    assert second.skipped_existing_count == len(chunks)
    assert set(second.skipped_chunk_ids) == {c.chunk_id for c in chunks}


@pytest.mark.anyio
async def test_force_reembeds_even_when_already_persisted() -> None:
    chunks = chunk_document(make_document(), DEFAULT_CHUNKING_CONFIG)
    provider = FakeDeterministicEmbeddingProvider(dimensions=8)
    repository = InMemoryEmbeddingRepository()
    service = EmbeddingPersistenceService(provider, repository)

    await service.embed_and_persist_chunks(chunks)
    forced = await service.embed_and_persist_chunks(chunks, force=True)

    assert forced.embedded_count == len(chunks)
    assert forced.skipped_existing_count == 0


@pytest.mark.anyio
async def test_empty_chunk_list_is_a_no_op() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=8)
    repository = InMemoryEmbeddingRepository()
    service = EmbeddingPersistenceService(provider, repository)

    result = await service.embed_and_persist_chunks([])

    assert result.total_chunks == 0
    assert result.embedded_count == 0
    assert repository.upsert_calls == []


@pytest.mark.anyio
async def test_batching_splits_into_multiple_upsert_calls() -> None:
    # Two documents worth of chunks, batch_size=1 forces multiple
    # embed+upsert round trips - verifies batching doesn't drop or
    # duplicate chunks.
    document = ClinicalDocument(
        metadata=ClinicalDocumentMetadata(
            document_id="doc-multi",
            title="Multi-section Guideline",
            organization="Example Org",
            source_url="https://example.org/guideline",
            access_date=date(2026, 9, 9),
            category=ClinicalDocumentCategory.HYPERTENSION,
        ),
        sections=[
            ClinicalSection(
                section_id=f"s{i}",
                heading=f"Section {i}",
                text=" ".join(f"word{j}" for j in range(20)),
                order=i,
            )
            for i in range(3)
        ],
    )
    chunks = chunk_document(document, DEFAULT_CHUNKING_CONFIG)

    provider = FakeDeterministicEmbeddingProvider(dimensions=8)
    repository = InMemoryEmbeddingRepository()
    service = EmbeddingPersistenceService(provider, repository, batch_size=1)

    result = await service.embed_and_persist_chunks(chunks)

    assert result.embedded_count == len(chunks)
    assert len(repository.upsert_calls) == len(chunks)
    assert all(len(call) == 1 for call in repository.upsert_calls)


def test_batch_size_must_be_positive() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=8)
    repository = InMemoryEmbeddingRepository()

    with pytest.raises(ValueError):
        EmbeddingPersistenceService(provider, repository, batch_size=0)
