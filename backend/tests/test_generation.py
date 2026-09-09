"""Unit tests for app.clinical.generation (Milestone E, section 8)."""

from __future__ import annotations

import pytest

from app.clinical.generation import GroundedAnswer, generate_grounded_answer
from app.clinical.llm import (
    FailingStructuredLLMProvider,
    FakeStructuredLLMProvider,
    LLMProviderError,
)
from app.clinical.prompting import build_generation_prompt


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _answer(**overrides) -> GroundedAnswer:
    defaults = dict(
        patient_summary="No patient context was provided.",
        clinical_question="What is the target blood pressure?",
        answer="Guidelines recommend a target below 130/80 mmHg [C1].",
        key_points=["Target below 130/80 mmHg"],
        citation_ids=["C1"],
        warnings=[],
        disclaimer="Educational and research use only.",
    )
    defaults.update(overrides)
    return GroundedAnswer(**defaults)


@pytest.mark.anyio
async def test_generate_grounded_answer_delegates_to_the_llm_provider() -> None:
    expected = _answer()
    fake_llm = FakeStructuredLLMProvider(response=expected)
    prompt = build_generation_prompt(
        question="What is the target blood pressure?", patient_context=None, citations=[]
    )

    result = await generate_grounded_answer(fake_llm, prompt, temperature=0.1, timeout_seconds=5.0)

    assert result == expected
    system_prompt, user_prompt, response_model = fake_llm.calls[0]
    assert system_prompt == prompt.system_prompt
    assert user_prompt == prompt.user_prompt
    assert response_model is GroundedAnswer


@pytest.mark.anyio
async def test_generate_grounded_answer_propagates_llm_errors() -> None:
    failing_llm = FailingStructuredLLMProvider(error=LLMProviderError("provider down"))
    prompt = build_generation_prompt(question="Q", patient_context=None, citations=[])

    with pytest.raises(LLMProviderError):
        await generate_grounded_answer(failing_llm, prompt)


def test_grounded_answer_forbids_extra_fields() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        GroundedAnswer(
            patient_summary="x",
            clinical_question="x",
            answer="x",
            disclaimer="x",
            unexpected_field="not allowed",
        )
