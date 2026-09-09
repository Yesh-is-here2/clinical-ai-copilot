from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.embeddings import (
    EmbeddingProviderNotConfiguredError,
    EmbeddingProviderRequestError,
    FakeDeterministicEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)


# ---------------------------------------------------------------------
# FakeDeterministicEmbeddingProvider
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_fake_provider_is_deterministic() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=8)

    first = await provider.embed_batch(["hypertension evidence"])
    second = await provider.embed_batch(["hypertension evidence"])

    assert first == second


@pytest.mark.anyio
async def test_fake_provider_produces_correct_dimensionality() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=16)

    vectors = await provider.embed_batch(["a", "b"])

    assert len(vectors) == 2
    assert all(len(vector) == 16 for vector in vectors)


@pytest.mark.anyio
async def test_fake_provider_different_text_gives_different_vectors() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=8)

    vectors = await provider.embed_batch(["diagnosis", "management"])

    assert vectors[0] != vectors[1]


@pytest.mark.anyio
async def test_fake_provider_empty_batch_returns_empty_list() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=8)

    assert await provider.embed_batch([]) == []


@pytest.mark.anyio
async def test_fake_provider_rejects_blank_text() -> None:
    provider = FakeDeterministicEmbeddingProvider(dimensions=8)

    with pytest.raises(ValueError):
        await provider.embed_batch(["valid text", "   "])


# ---------------------------------------------------------------------
# OpenAIEmbeddingProvider: missing key
# ---------------------------------------------------------------------


def test_openai_provider_requires_api_key() -> None:
    with pytest.raises(EmbeddingProviderNotConfiguredError):
        OpenAIEmbeddingProvider(
            api_key=None,
            model="text-embedding-3-small",
            dimensions=1536,
        )


def test_openai_provider_rejects_blank_api_key() -> None:
    with pytest.raises(EmbeddingProviderNotConfiguredError):
        OpenAIEmbeddingProvider(
            api_key="   ",
            model="text-embedding-3-small",
            dimensions=1536,
        )


def test_openai_provider_rejects_empty_string_api_key() -> None:
    # Matches a real .env with `OPENAI_API_KEY=` left blank - pydantic-
    # settings loads that as "" (not None), so this must be treated as
    # "not configured" too, not passed through to the OpenAI client.
    with pytest.raises(EmbeddingProviderNotConfiguredError):
        OpenAIEmbeddingProvider(
            api_key="",
            model="text-embedding-3-small",
            dimensions=1536,
        )


def test_get_embedding_provider_raises_without_configured_key() -> None:
    with patch("app.core.config.settings") as mock_settings:
        mock_settings.openai_api_key = None
        mock_settings.embedding_model = "text-embedding-3-small"

        with pytest.raises(EmbeddingProviderNotConfiguredError):
            get_embedding_provider()


# ---------------------------------------------------------------------
# OpenAIEmbeddingProvider: mocked success/failure (never touches network)
# ---------------------------------------------------------------------


class _FakeEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_FakeEmbeddingItem(vector) for vector in vectors]


@pytest.mark.anyio
async def test_openai_provider_embed_batch_success() -> None:
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test-not-a-real-key",
        model="text-embedding-3-small",
        dimensions=4,
    )

    fake_vectors = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    provider._client.embeddings.create = AsyncMock(
        return_value=_FakeEmbeddingResponse(fake_vectors)
    )

    result = await provider.embed_batch(["first chunk", "second chunk"])

    assert result == fake_vectors
    provider._client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-3-small",
        input=["first chunk", "second chunk"],
    )


@pytest.mark.anyio
async def test_openai_provider_empty_batch_never_calls_api() -> None:
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test-not-a-real-key",
        model="text-embedding-3-small",
        dimensions=4,
    )
    provider._client.embeddings.create = AsyncMock()

    result = await provider.embed_batch([])

    assert result == []
    provider._client.embeddings.create.assert_not_awaited()


@pytest.mark.anyio
async def test_openai_provider_dimension_mismatch_raises() -> None:
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test-not-a-real-key",
        model="text-embedding-3-small",
        dimensions=1536,
    )
    # Server "returns" a wrong-sized vector - simulates a model/config
    # drift, which must surface as an explicit error, not silently
    # corrupt the pgvector column.
    provider._client.embeddings.create = AsyncMock(
        return_value=_FakeEmbeddingResponse([[0.1, 0.2]])
    )

    with pytest.raises(EmbeddingProviderRequestError, match="dimensional"):
        await provider.embed_batch(["chunk text"])


@pytest.mark.anyio
async def test_openai_provider_translates_timeout_error() -> None:
    from openai import APITimeoutError

    provider = OpenAIEmbeddingProvider(
        api_key="sk-test-not-a-real-key",
        model="text-embedding-3-small",
        dimensions=4,
    )
    provider._client.embeddings.create = AsyncMock(
        side_effect=APITimeoutError(request=MagicMock())
    )

    with pytest.raises(EmbeddingProviderRequestError, match="timed out"):
        await provider.embed_batch(["chunk text"])


@pytest.mark.anyio
async def test_openai_provider_rejects_blank_text_in_batch() -> None:
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test-not-a-real-key",
        model="text-embedding-3-small",
        dimensions=4,
    )
    provider._client.embeddings.create = AsyncMock()

    with pytest.raises(ValueError):
        await provider.embed_batch(["valid", ""])

    provider._client.embeddings.create.assert_not_awaited()
