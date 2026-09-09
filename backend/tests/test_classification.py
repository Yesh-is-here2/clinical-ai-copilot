"""Unit tests for app.clinical.classification (Milestone E, section 4)."""

from __future__ import annotations

import pytest

from app.clinical.classification import (
    DeterministicQuestionClassifier,
    LLMQuestionClassifier,
    QuestionCategory,
    _LLMClassificationResponse,
)
from app.clinical.llm import (
    FailingStructuredLLMProvider,
    FakeStructuredLLMProvider,
    LLMProviderError,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "question,expected_category",
    [
        (
            "Is lisinopril contraindicated for a patient allergic to ACE inhibitors?",
            QuestionCategory.CONTRAINDICATION_SAFETY,
        ),
        (
            "My A1C lab result came back elevated, what does that mean?",
            QuestionCategory.LAB_INTERPRETATION,
        ),
        (
            "What is the starting dose for this medication?",
            QuestionCategory.MEDICATION,
        ),
        (
            "How can screening help with prevention of cardiovascular disease?",
            QuestionCategory.PREVENTION,
        ),
        (
            "What causes hypertension in adults?",
            QuestionCategory.CONDITION,
        ),
        (
            "What do the guidelines recommend for blood pressure targets?",
            QuestionCategory.GUIDELINE,
        ),
        (
            "The patient's doctor discussed general health today.",
            QuestionCategory.GENERAL_CLINICAL_INFORMATION,
        ),
        (
            "What's the weather like in Paris tomorrow?",
            QuestionCategory.UNSUPPORTED_OUT_OF_SCOPE,
        ),
    ],
)
async def test_deterministic_classifier_categories(
    question: str, expected_category: QuestionCategory
) -> None:
    classifier = DeterministicQuestionClassifier()

    result = await classifier.classify(question)

    assert result.category == expected_category
    assert result.method == "deterministic"


@pytest.mark.anyio
async def test_blank_question_is_out_of_scope() -> None:
    classifier = DeterministicQuestionClassifier()

    result = await classifier.classify("")

    assert result.category == QuestionCategory.UNSUPPORTED_OUT_OF_SCOPE
    assert result.confidence == 1.0


@pytest.mark.anyio
async def test_more_specific_category_wins_priority_order() -> None:
    # Mentions both a medication term and a contraindication-safety term
    # - contraindication_safety is checked first in the priority list
    # and should win over medication.
    classifier = DeterministicQuestionClassifier()

    result = await classifier.classify(
        "Is this medication contraindicated for patients with kidney disease?"
    )

    assert result.category == QuestionCategory.CONTRAINDICATION_SAFETY


@pytest.mark.anyio
async def test_matched_keywords_are_reported() -> None:
    classifier = DeterministicQuestionClassifier()

    result = await classifier.classify("What is the recommended dosage for this drug?")

    assert "dosage" in result.matched_keywords


@pytest.mark.anyio
async def test_llm_classifier_uses_llm_result_on_success() -> None:
    response = _LLMClassificationResponse(
        category=QuestionCategory.MEDICATION, confidence=0.95
    )
    fake_llm = FakeStructuredLLMProvider(response=response)
    classifier = LLMQuestionClassifier(fake_llm)

    result = await classifier.classify("What is the dosage?")

    assert result.category == QuestionCategory.MEDICATION
    assert result.confidence == 0.95
    assert result.method == "llm"
    assert len(fake_llm.calls) == 1


@pytest.mark.anyio
async def test_llm_classifier_falls_back_to_deterministic_on_llm_error() -> None:
    failing_llm = FailingStructuredLLMProvider(error=LLMProviderError("boom"))
    classifier = LLMQuestionClassifier(failing_llm)

    result = await classifier.classify(
        "What do the guidelines recommend for hypertension?"
    )

    # Falls back to the deterministic classifier rather than raising -
    # never blocks the whole request over an unavailable/misbehaving LLM.
    assert result.method == "deterministic"
    assert result.category == QuestionCategory.GUIDELINE


@pytest.mark.anyio
async def test_llm_classifier_accepts_a_custom_fallback() -> None:
    failing_llm = FailingStructuredLLMProvider()
    custom_fallback = DeterministicQuestionClassifier()
    classifier = LLMQuestionClassifier(failing_llm, fallback=custom_fallback)

    result = await classifier.classify("What causes diabetes?")

    assert result.category == QuestionCategory.CONDITION
