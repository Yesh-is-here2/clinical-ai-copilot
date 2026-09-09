"""Reranker abstraction for Milestone D hybrid retrieval (section 5/6).

Two implementations:

* ``CrossEncoderReranker`` - the real thing, backed by a
  sentence-transformers cross-encoder (default model:
  cross-encoder/ms-marco-MiniLM-L-6-v2). ``sentence-transformers`` (and
  the ``torch`` it pulls in) is a genuinely heavy dependency, so it is
  deliberately NOT in ``requirements.txt`` - it lives in the optional
  ``backend/requirements-reranker.txt`` extra (see backend/Dockerfile's
  ``INSTALL_RERANKER`` build arg) and is imported lazily, inside
  ``_load_model()`` only, only when a real rerank is actually attempted.
  Constructing this class, or importing this module, never touches the
  network, the filesystem, or the ``sentence_transformers`` package.
* ``FakeReranker`` / ``FailingReranker`` - test-only. Deterministic (no
  ML, no network), so hybrid-retrieval orchestration tests can exercise
  both "reranker succeeded" and "reranker failed" paths without ever
  importing sentence-transformers or downloading a model - see
  app.services.embeddings.FakeDeterministicEmbeddingProvider for the
  same pattern applied to embeddings.

Nothing in this module calls the network or loads a model at import
time, and nothing here silently relabels a fallback/heuristic score as
though it came from the real cross-encoder.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Protocol, Sequence, runtime_checkable

from app.core.constants import DEFAULT_RERANKER_MODEL


class RerankerError(RuntimeError):
    """Base class for all reranker failures. Callers (see
    app.retrieval.hybrid.HybridRetrievalService) must catch exactly this
    - never a bare Exception - so an unrelated bug inside a Reranker
    implementation is not silently reinterpreted as "reranker
    unavailable, fall back gracefully"."""


class RerankerNotAvailableError(RerankerError):
    """Raised when the reranker cannot be used at all: the optional
    sentence-transformers/torch dependency isn't installed, or the
    configured model failed to load. Callers must treat this exactly
    like a scoring failure - fall back to the fused (pre-rerank)
    ordering with explicit degradation metadata, never crash the whole
    retrieval request over a missing optional dependency."""


class RerankerScoringError(RerankerError):
    """Raised when the model loaded successfully but scoring an actual
    batch of (query, text) pairs failed."""


@runtime_checkable
class Reranker(Protocol):
    """Anything that can score a query against a batch of candidate
    texts, in the same order they were given. Higher score = more
    relevant. Implementations must never silently substitute a
    fallback/heuristic score and report it as a real reranker score - if
    scoring can't happen, raise a ``RerankerError`` and let the caller
    handle the fallback explicitly."""

    model_name: str

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        ...


class CrossEncoderReranker:
    """Real reranker backed by ``sentence_transformers.CrossEncoder``.

    Model loading is lazy and cached: the first call to ``score()``
    imports ``sentence_transformers`` and constructs the model; every
    later call on this instance reuses it. Constructing this class only
    assigns attributes - it never imports ``sentence_transformers``,
    touches the network, or reads from disk.
    """

    def __init__(self, *, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerNotAvailableError(
                "sentence-transformers is not installed, so the "
                f"cross-encoder reranker ('{self.model_name}') is not "
                "available. Install the optional "
                "backend/requirements-reranker.txt extra (Docker: build "
                "with --build-arg INSTALL_RERANKER=true) to enable real "
                "reranking, or set reranker_enabled=False to skip "
                "reranking by design instead of hitting this fallback on "
                "every request."
            ) from exc

        try:
            self._model = CrossEncoder(self.model_name)
        except Exception as exc:  # pragma: no cover - depends on network/model cache
            raise RerankerNotAvailableError(
                f"Failed to load cross-encoder model '{self.model_name}': {exc}"
            ) from exc

        return self._model

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []

        model = self._load_model()
        pairs = [(query, text) for text in texts]

        try:
            # CrossEncoder.predict is synchronous, CPU/GPU-bound work -
            # run it off the event loop rather than blocking it.
            raw_scores = await asyncio.to_thread(model.predict, pairs)
        except Exception as exc:
            raise RerankerScoringError(
                f"Cross-encoder scoring failed for model "
                f"'{self.model_name}': {exc}"
            ) from exc

        return [float(score) for score in raw_scores]


class FakeReranker:
    """Deterministic, offline, test-only reranker.

    Never used by application runtime code. The default scorer is a
    trivial query/text term-overlap ratio - not a real relevance signal,
    just something deterministic and text-dependent enough to produce a
    non-trivial (and non-random) reordering in tests. A custom
    ``scorer`` callable can be passed to control exact scores in a
    specific test.
    """

    model_name = "fake-reranker"

    def __init__(self, *, scorer: Callable[[str, str], float] | None = None) -> None:
        self._scorer = scorer or self._default_scorer

    @staticmethod
    def _default_scorer(query: str, text: str) -> float:
        query_terms = set(query.lower().split())
        text_terms = text.lower().split()

        if not text_terms:
            return 0.0

        overlap = sum(1 for term in text_terms if term in query_terms)

        return overlap / len(text_terms)

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [self._scorer(query, text) for text in texts]


class FailingReranker:
    """Test-only reranker that always raises - for exercising the
    reranker-failure fallback path directly, without depending on
    CrossEncoderReranker being unavailable for the right reason."""

    model_name = "failing-reranker"

    def __init__(self, *, error: RerankerError | None = None) -> None:
        self._error = error or RerankerScoringError("synthetic failure for tests")

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        raise self._error
