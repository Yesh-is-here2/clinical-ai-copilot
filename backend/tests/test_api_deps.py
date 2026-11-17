"""Unit tests for ``app.api.deps`` (Milestone F, sections 12/13).

Covers the "lazy provider" contract explicitly required by the spec:
constructing the LLM provider, and asking for hybrid retrieval with
nothing ingested, must never touch the network or require
``OPENAI_API_KEY`` - and ``get_current_actor`` must stay a true
placeholder (always ``None``) until Phase 7.
"""

from __future__ import annotations

import pytest

from app.api.deps import (
    Actor,
    get_current_actor,
    get_hybrid_retrieval_service,
    get_llm_provider,
)
from app.clinical.llm import OpenAIStructuredLLMProvider
from app.core.config import Settings


class _FakeResult:
    def all(self):
        return []


class _FakeAsyncSession:
    async def execute(self, stmt):
        return _FakeResult()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _settings(**overrides) -> Settings:
    base = {
        "database_url": "postgresql+asyncpg://user:pass@localhost:5432/db",
        "openai_api_key": None,
    }
    base.update(overrides)
    return Settings(**base)


def test_get_llm_provider_never_requires_an_api_key() -> None:
    provider = get_llm_provider(_settings(openai_api_key=None))

    assert isinstance(provider, OpenAIStructuredLLMProvider)
    # Constructing it must not have built a real OpenAI client yet.
    assert provider._client is None  # noqa: SLF001 - white-box lazy-init check


@pytest.mark.anyio
async def test_hybrid_retrieval_is_none_with_nothing_ingested_and_no_api_key() -> None:
    result = await get_hybrid_retrieval_service(
        _FakeAsyncSession(),  # type: ignore[arg-type]
        _settings(openai_api_key=None),
    )

    assert result is None


@pytest.mark.anyio
async def test_get_current_actor_is_always_none_until_phase_7() -> None:
    actor = await get_current_actor()

    assert actor is None


def test_actor_placeholder_model_exists_for_future_phase_7_wiring() -> None:
    actor = Actor(id="user-1", role="clinician")

    assert actor.id == "user-1"
    assert actor.role == "clinician"
