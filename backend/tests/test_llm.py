"""Unit tests for app.clinical.llm (Milestone E, section 7).

Covers the safety-critical property this module exists for: importing
it, and constructing ``OpenAIStructuredLLMProvider``, must never touch
the network or require an API key - only an actual call does, and only
then does a missing key raise a typed, catchable error. The real
provider's request/response plumbing is exercised with a monkeypatched
client (still no network) so the JSON-parsing and schema-validation
branches get real coverage without depending on OpenAI's exception
constructors.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.clinical.llm import (
    FailingStructuredLLMProvider,
    FakeStructuredLLMProvider,
    LLMMalformedResponseError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMResponseValidationError,
    OpenAIStructuredLLMProvider,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _EchoResponse(BaseModel):
    text: str


# ---------------------------------------------------------------------
# OpenAIStructuredLLMProvider - construction / no-key behavior
# ---------------------------------------------------------------------


def test_construction_never_requires_a_key() -> None:
    provider = OpenAIStructuredLLMProvider(api_key=None, model="gpt-4o-mini")

    assert provider.model == "gpt-4o-mini"


def test_construction_never_touches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # If construction imported/instantiated the OpenAI client eagerly,
    # this would need a real network-capable environment; it must not.
    OpenAIStructuredLLMProvider(api_key="sk-fake", model="gpt-4o-mini")


@pytest.mark.anyio
async def test_missing_key_raises_not_configured_only_when_called() -> None:
    provider = OpenAIStructuredLLMProvider(api_key=None, model="gpt-4o-mini")

    with pytest.raises(LLMNotConfiguredError):
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=_EchoResponse,
        )


@pytest.mark.anyio
async def test_blank_key_also_raises_not_configured() -> None:
    provider = OpenAIStructuredLLMProvider(api_key="   ", model="gpt-4o-mini")

    with pytest.raises(LLMNotConfiguredError):
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=_EchoResponse,
        )


def test_not_configured_error_is_an_llm_error() -> None:
    assert issubclass(LLMNotConfiguredError, Exception)
    from app.clinical.llm import LLMError

    assert issubclass(LLMNotConfiguredError, LLMError)


# ---------------------------------------------------------------------
# OpenAIStructuredLLMProvider - request/response plumbing, via a
# monkeypatched client (no real network call).
# ---------------------------------------------------------------------


class _FakeChoiceMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeChoiceMessage(content)


class _FakeCompletionResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeCompletionResponse(self._content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeAsyncOpenAIClient:
    def __init__(self, content: str) -> None:
        self.chat = _FakeChat(content)


def _provider_with_fake_client(content: str) -> OpenAIStructuredLLMProvider:
    provider = OpenAIStructuredLLMProvider(api_key="sk-fake", model="gpt-4o-mini")
    # Bypass the lazy `_get_client()` (which would require a real key
    # check + `openai.AsyncOpenAI` construction) by setting the
    # already-constructed client directly - the same object
    # `_get_client()` would otherwise cache.
    provider._client = _FakeAsyncOpenAIClient(content)
    return provider


@pytest.mark.anyio
async def test_valid_json_response_is_parsed_and_validated() -> None:
    provider = _provider_with_fake_client('{"text": "hello"}')

    result = await provider.generate_structured(
        system_prompt="system",
        user_prompt="user",
        response_model=_EchoResponse,
    )

    assert result == _EchoResponse(text="hello")


@pytest.mark.anyio
async def test_malformed_json_raises_typed_error() -> None:
    provider = _provider_with_fake_client("not json at all {{{")

    with pytest.raises(LLMMalformedResponseError):
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=_EchoResponse,
        )


@pytest.mark.anyio
async def test_json_not_matching_schema_raises_typed_validation_error() -> None:
    provider = _provider_with_fake_client('{"wrong_field": 123}')

    with pytest.raises(LLMResponseValidationError):
        await provider.generate_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=_EchoResponse,
        )


@pytest.mark.anyio
async def test_request_is_sent_with_json_object_response_format() -> None:
    provider = _provider_with_fake_client('{"text": "hi"}')

    await provider.generate_structured(
        system_prompt="sys",
        user_prompt="usr",
        response_model=_EchoResponse,
        temperature=0.2,
        timeout_seconds=15.0,
    )

    call = provider._client.chat.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["temperature"] == 0.2
    assert call["timeout"] == 15.0
    assert call["messages"][0] == {"role": "system", "content": "sys"}
    assert call["messages"][1] == {"role": "user", "content": "usr"}


# ---------------------------------------------------------------------
# FakeStructuredLLMProvider / FailingStructuredLLMProvider
# ---------------------------------------------------------------------


def test_fake_provider_requires_response_or_responder() -> None:
    with pytest.raises(ValueError):
        FakeStructuredLLMProvider()


@pytest.mark.anyio
async def test_fake_provider_returns_fixed_response() -> None:
    response = _EchoResponse(text="fixed")
    provider = FakeStructuredLLMProvider(response=response)

    result = await provider.generate_structured(
        system_prompt="s", user_prompt="u", response_model=_EchoResponse
    )

    assert result is response
    assert provider.calls == [("s", "u", _EchoResponse)]


@pytest.mark.anyio
async def test_fake_provider_responder_can_vary_by_prompt() -> None:
    def responder(system_prompt, user_prompt, response_model):
        return _EchoResponse(text=user_prompt.upper())

    provider = FakeStructuredLLMProvider(responder=responder)

    result = await provider.generate_structured(
        system_prompt="s", user_prompt="abc", response_model=_EchoResponse
    )

    assert result.text == "ABC"


@pytest.mark.anyio
async def test_fake_provider_raises_if_response_type_mismatches_request() -> None:
    class _OtherResponse(BaseModel):
        value: int

    provider = FakeStructuredLLMProvider(response=_EchoResponse(text="x"))

    with pytest.raises(LLMResponseValidationError):
        await provider.generate_structured(
            system_prompt="s", user_prompt="u", response_model=_OtherResponse
        )


@pytest.mark.anyio
async def test_failing_provider_raises_default_error() -> None:
    provider = FailingStructuredLLMProvider()

    with pytest.raises(LLMProviderError):
        await provider.generate_structured(
            system_prompt="s", user_prompt="u", response_model=_EchoResponse
        )


@pytest.mark.anyio
async def test_failing_provider_raises_custom_error() -> None:
    from app.clinical.llm import LLMTimeoutError

    provider = FailingStructuredLLMProvider(error=LLMTimeoutError("timed out"))

    with pytest.raises(LLMTimeoutError):
        await provider.generate_structured(
            system_prompt="s", user_prompt="u", response_model=_EchoResponse
        )
