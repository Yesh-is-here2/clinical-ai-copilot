"""Unit tests for app.retrieval.reranker.

No network, no model download: CrossEncoderReranker's lazy-loading
boundary is tested against the fact that ``sentence-transformers`` is
genuinely not installed in this environment (see
backend/requirements-reranker.txt - it's an optional extra, not in
requirements.txt), so calling .score() here exercises the real
"dependency not available" fallback path, not a mock of it.
"""

from __future__ import annotations

import sys

import pytest

from app.retrieval.reranker import (
    CrossEncoderReranker,
    FailingReranker,
    FakeReranker,
    RerankerError,
    RerankerNotAvailableError,
    RerankerScoringError,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------
# Lazy loading boundary
# ---------------------------------------------------------------------


def test_constructing_cross_encoder_reranker_does_not_import_sentence_transformers() -> None:
    assert "sentence_transformers" not in sys.modules

    reranker = CrossEncoderReranker()

    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert "sentence_transformers" not in sys.modules


def test_model_name_is_configurable() -> None:
    reranker = CrossEncoderReranker(model_name="some-other/cross-encoder")

    assert reranker.model_name == "some-other/cross-encoder"


@pytest.mark.anyio
async def test_scoring_with_the_dependency_missing_raises_not_available() -> None:
    # sentence-transformers is genuinely not installed in this sandbox -
    # this proves the real fallback path, not a mocked stand-in for it.
    reranker = CrossEncoderReranker()

    with pytest.raises(RerankerNotAvailableError):
        await reranker.score("hypertension", ["some candidate text"])


@pytest.mark.anyio
async def test_scoring_empty_texts_returns_empty_without_loading_a_model() -> None:
    reranker = CrossEncoderReranker()

    result = await reranker.score("hypertension", [])

    assert result == []
    assert "sentence_transformers" not in sys.modules


def test_not_available_error_is_a_reranker_error() -> None:
    assert issubclass(RerankerNotAvailableError, RerankerError)


def test_scoring_error_is_a_reranker_error() -> None:
    assert issubclass(RerankerScoringError, RerankerError)


# ---------------------------------------------------------------------
# FakeReranker (test-only, used by hybrid orchestration tests)
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_fake_reranker_default_scorer_is_deterministic() -> None:
    reranker = FakeReranker()

    first = await reranker.score("hypertension diagnosis", ["hypertension evidence"])
    second = await reranker.score("hypertension diagnosis", ["hypertension evidence"])

    assert first == second


@pytest.mark.anyio
async def test_fake_reranker_default_scorer_favors_term_overlap() -> None:
    reranker = FakeReranker()

    scores = await reranker.score(
        "hypertension diagnosis",
        ["hypertension diagnosis evidence", "unrelated diabetes content"],
    )

    assert scores[0] > scores[1]


@pytest.mark.anyio
async def test_fake_reranker_accepts_a_custom_scorer() -> None:
    reranker = FakeReranker(scorer=lambda query, text: float(len(text)))

    scores = await reranker.score("q", ["ab", "abcdef"])

    assert scores == [2.0, 6.0]


@pytest.mark.anyio
async def test_fake_reranker_preserves_model_name_distinct_from_real_one() -> None:
    reranker = FakeReranker()

    assert reranker.model_name != CrossEncoderReranker().model_name
    assert reranker.model_name == "fake-reranker"


# ---------------------------------------------------------------------
# FailingReranker (test-only, used to exercise failure fallback)
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_failing_reranker_raises_by_default() -> None:
    reranker = FailingReranker()

    with pytest.raises(RerankerScoringError):
        await reranker.score("q", ["text"])


@pytest.mark.anyio
async def test_failing_reranker_accepts_a_custom_error() -> None:
    reranker = FailingReranker(error=RerankerNotAvailableError("custom"))

    with pytest.raises(RerankerNotAvailableError, match="custom"):
        await reranker.score("q", ["text"])
