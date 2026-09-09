"""Unit tests for SqlAlchemyEmbeddingRepository's dimension-validation
boundary (no database - see tests/integration/test_postgres_integration.py
for the real Postgres/pgvector round trip).

``document_embeddings.embedding`` is a fixed VECTOR(DEFAULT_EMBEDDING_
DIMENSIONS) column. The repository must reject a mismatched embedding
with a clear ``EmbeddingDimensionMismatchError`` *before* it ever calls
``session.execute`` - these tests assert exactly that boundary using a
minimal fake session that records whether it was touched at all, rather
than a real AsyncSession.
"""

from __future__ import annotations

import pytest

from app.core.constants import DEFAULT_EMBEDDING_DIMENSIONS
from app.db.repositories.embedding_repository import (
    EmbeddingDimensionMismatchError,
    SqlAlchemyEmbeddingRepository,
)
from app.services.embedding_persistence import EmbeddingRecordInput


class _FakeAsyncSession:
    """Records whether the repository ever reached the database layer.

    Intentionally not a real AsyncSession/engine - the point of these
    tests is that a dimension mismatch is rejected *before* any of this
    is touched, so a real database is never needed to prove it.
    """

    def __init__(self) -> None:
        self.executed: list[object] = []
        self.committed = False

    async def execute(self, stmt: object) -> None:
        self.executed.append(stmt)

    async def commit(self) -> None:
        self.committed = True


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_record(*, dimensions: int, vector_length: int) -> EmbeddingRecordInput:
    return EmbeddingRecordInput(
        chunk_id="chunk-1",
        model="fake-deterministic",
        dimensions=dimensions,
        embedding=[0.1] * vector_length,
    )


@pytest.mark.anyio
async def test_correct_dimension_is_accepted_and_reaches_the_session() -> None:
    session = _FakeAsyncSession()
    repository = SqlAlchemyEmbeddingRepository(session)  # type: ignore[arg-type]
    record = _make_record(
        dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
        vector_length=DEFAULT_EMBEDDING_DIMENSIONS,
    )

    count = await repository.upsert_many([record])

    assert count == 1
    assert session.executed  # the insert statement was actually issued
    assert session.committed is True


@pytest.mark.anyio
async def test_wrong_declared_dimension_is_rejected_before_touching_session() -> None:
    session = _FakeAsyncSession()
    repository = SqlAlchemyEmbeddingRepository(session)  # type: ignore[arg-type]
    # 8 != DEFAULT_EMBEDDING_DIMENSIONS (1536) - the column can't store this.
    record = _make_record(dimensions=8, vector_length=8)

    with pytest.raises(EmbeddingDimensionMismatchError):
        await repository.upsert_many([record])

    assert session.executed == []
    assert session.committed is False


@pytest.mark.anyio
async def test_vector_length_inconsistent_with_declared_dimensions_is_rejected() -> None:
    session = _FakeAsyncSession()
    repository = SqlAlchemyEmbeddingRepository(session)  # type: ignore[arg-type]
    # Declares the right dimensionality but the actual vector disagrees -
    # a buggy/misbehaving provider, not a config mismatch.
    record = _make_record(
        dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
        vector_length=DEFAULT_EMBEDDING_DIMENSIONS - 1,
    )

    with pytest.raises(EmbeddingDimensionMismatchError):
        await repository.upsert_many([record])

    assert session.executed == []
    assert session.committed is False


@pytest.mark.anyio
async def test_one_bad_record_in_a_batch_rejects_the_whole_batch_before_any_write() -> None:
    # Batches are validated up front, not partially applied - a mismatch
    # anywhere in the batch must not silently persist the rest of it.
    session = _FakeAsyncSession()
    repository = SqlAlchemyEmbeddingRepository(session)  # type: ignore[arg-type]
    good = _make_record(
        dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
        vector_length=DEFAULT_EMBEDDING_DIMENSIONS,
    )
    bad = _make_record(dimensions=8, vector_length=8)

    with pytest.raises(EmbeddingDimensionMismatchError):
        await repository.upsert_many([good, bad])

    assert session.executed == []
    assert session.committed is False
