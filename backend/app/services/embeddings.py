"""Embedding provider abstraction (Milestone C, section 5).

Nothing outside this module talks to OpenAI directly for embeddings - API
routes and retrieval code depend on the ``EmbeddingProvider`` protocol,
never on ``openai`` or a hardcoded API key. That keeps embedding calls in
exactly one place, makes them trivially mockable in tests, and means a
future second provider (a local model, a different vendor) is a new class
implementing the same protocol, not a rewrite of every caller.

Two implementations:

* ``OpenAIEmbeddingProvider`` - the real thing, used in production. Never
  constructed unless an API key is actually configured (constructing it
  with no key raises immediately, rather than deferring the failure to
  the first embedding call).
* ``FakeDeterministicEmbeddingProvider`` - test-only. Produces the same
  vector for the same text every time (hash-derived, not random) with no
  network access, so retrieval/persistence tests are fast and
  reproducible without ever touching a real API key.

Nothing in this module calls the network at import time - constructing
``OpenAIEmbeddingProvider`` builds a client lazily; it doesn't validate
the key against OpenAI until the first real ``embed_batch`` call.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence, runtime_checkable


class EmbeddingProviderError(RuntimeError):
    """Base class for all embedding-provider failures."""


class EmbeddingProviderNotConfiguredError(EmbeddingProviderError):
    """Raised when a real provider is requested but not configured
    (e.g. no OPENAI_API_KEY). Callers must handle this explicitly and
    fail safely - never fall back to fake embeddings in application
    runtime code (only tests may use the fake provider)."""


class EmbeddingProviderRequestError(EmbeddingProviderError):
    """Raised when the provider's API call itself fails (timeout, rate
    limit, malformed response, dimension mismatch, ...)."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that can turn a batch of text chunks into vectors of a
    known, fixed dimensionality."""

    model: str
    dimensions: int

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        ...


def _validate_batch_input(texts: Sequence[str]) -> None:
    for text in texts:
        if not text or not text.strip():
            raise ValueError(
                "embed_batch received an empty/blank text input; "
                "chunking should never produce empty chunks (see "
                "app.retrieval.chunking), so this indicates a bug "
                "upstream rather than a provider failure."
            )


class OpenAIEmbeddingProvider:
    """Real embedding provider backed by the OpenAI embeddings API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        dimensions: int,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise EmbeddingProviderNotConfiguredError(
                "OPENAI_API_KEY is not configured. Real embedding calls "
                "require it; do not silently substitute fake vectors."
            )

        # Imported lazily so importing this module (e.g. at FastAPI
        # startup, or for the fake provider in tests) never requires the
        # `openai` package to touch the network, and so a missing key
        # fails before any client object is even built.
        from openai import AsyncOpenAI

        self.model = model
        self.dimensions = dimensions
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_seconds)

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        _validate_batch_input(texts)

        from openai import APIError, APITimeoutError

        try:
            response = await self._client.embeddings.create(
                model=self.model,
                input=list(texts),
            )
        except APITimeoutError as exc:
            raise EmbeddingProviderRequestError(
                f"Embedding request to '{self.model}' timed out."
            ) from exc
        except APIError as exc:
            raise EmbeddingProviderRequestError(
                f"Embedding request to '{self.model}' failed: {exc}"
            ) from exc

        vectors = [item.embedding for item in response.data]

        for vector in vectors:
            if len(vector) != self.dimensions:
                raise EmbeddingProviderRequestError(
                    f"Expected {self.dimensions}-dimensional vectors from "
                    f"'{self.model}', got {len(vector)}. The configured "
                    "embedding model and the pgvector column dimension "
                    "(app.core.constants.DEFAULT_EMBEDDING_DIMENSIONS) "
                    "have drifted apart."
                )

        return vectors


class FakeDeterministicEmbeddingProvider:
    """Test-only provider. Deterministic, offline, no API key required.

    Never used by application runtime code - only by tests and (if ever
    needed) local fixture generation. The vector for a given text is
    always the same across runs/processes, which is what makes
    "duplicate re-ingestion is a no-op" tests possible without touching a
    real API.
    """

    model = "fake-deterministic"

    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        _validate_batch_input(texts)

        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()

        return [
            (digest[i % len(digest)] / 255.0) * 2.0 - 1.0
            for i in range(self.dimensions)
        ]


def get_embedding_provider() -> EmbeddingProvider:
    """Build the real embedding provider from application settings.

    This is the one place application code (routes, ingestion services)
    should call to get a provider - never construct
    ``OpenAIEmbeddingProvider`` directly elsewhere, and never call this
    at module import time (it must stay lazy so the app can start, and
    every non-embedding route can work, with no API key configured).
    Raises ``EmbeddingProviderNotConfiguredError`` if no key is set;
    callers must translate that into a safe, explicit API failure rather
    than falling back to fake embeddings.
    """

    from app.core.config import settings
    from app.core.constants import DEFAULT_EMBEDDING_DIMENSIONS

    return OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
    )
