"""Structured LLM call abstraction (Milestone E, section 7).

Nothing outside this module talks to OpenAI directly for clinical
generation/classification - graph nodes depend on the
``StructuredLLMProvider`` protocol, never on ``openai`` or a hardcoded
API key. That keeps LLM calls in exactly one place and makes them
trivially fakeable in tests (see ``FakeStructuredLLMProvider``/
``FailingStructuredLLMProvider``).

Unlike ``app.services.embeddings.OpenAIEmbeddingProvider`` (which fails
immediately at construction if no key is configured), this provider is
constructible with no key at all - the app must be able to start, and
every non-generation code path must keep working, with no
``OPENAI_API_KEY`` set. The key is only required, and only checked, at
the moment a real structured call is actually attempted
(``generate_structured``), matching section 7's "No API key needed until
a real call is attempted."
"""

from __future__ import annotations

import json
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError as PydanticValidationError


class LLMError(RuntimeError):
    """Base class for all structured-LLM-call failures."""


class LLMNotConfiguredError(LLMError):
    """Raised when a real call is attempted but no API key is configured.
    Callers must handle this explicitly (e.g. fail the generation step
    safely) - never silently fabricate a response."""


class LLMTimeoutError(LLMError):
    """Raised when the provider call exceeds its timeout."""


class LLMProviderError(LLMError):
    """Raised when the provider's API call itself fails (rate limit,
    malformed request, transport error, ...)."""


class LLMMalformedResponseError(LLMError):
    """Raised when the provider's response is not valid JSON at all."""


class LLMResponseValidationError(LLMError):
    """Raised when the response is valid JSON but does not match the
    requested ``response_model`` schema."""


ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


@runtime_checkable
class StructuredLLMProvider(Protocol):
    """Anything that can turn a (system prompt, user prompt) pair into a
    validated instance of a given Pydantic response model."""

    model: str

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModelT],
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
    ) -> ResponseModelT:
        ...


class OpenAIStructuredLLMProvider:
    """Real structured-LLM provider backed by the OpenAI chat completions
    API, using JSON-object response formatting + Pydantic validation.

    The client is constructed lazily on the first real call, exactly
    like ``OpenAIEmbeddingProvider`` - importing this module, or
    constructing this class, never touches the network or requires an
    API key.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        default_timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self._default_timeout_seconds = default_timeout_seconds
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self._api_key or not self._api_key.strip():
            raise LLMNotConfiguredError(
                "OPENAI_API_KEY is not configured. Real structured LLM "
                "calls require it; do not silently substitute a fake "
                "response."
            )

        # Imported lazily so importing this module never requires the
        # `openai` package to touch the network, and a missing key fails
        # before any client object is even built.
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=self._api_key)

        return self._client

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModelT],
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
    ) -> ResponseModelT:
        client = self._get_client()

        from openai import APIError, APITimeoutError

        try:
            response = await client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                timeout=timeout_seconds,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                f"Structured LLM request to '{self.model}' timed out "
                f"after {timeout_seconds}s."
            ) from exc
        except APIError as exc:
            raise LLMProviderError(
                f"Structured LLM request to '{self.model}' failed: {exc}"
            ) from exc

        raw_content = response.choices[0].message.content or ""

        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise LLMMalformedResponseError(
                f"Structured LLM response from '{self.model}' was not "
                "valid JSON."
            ) from exc

        try:
            return response_model.model_validate(data)
        except PydanticValidationError as exc:
            raise LLMResponseValidationError(
                f"Structured LLM response from '{self.model}' did not "
                f"match {response_model.__name__}: {exc}"
            ) from exc


class FakeStructuredLLMProvider:
    """Deterministic, offline, test-only provider.

    Either pass a fixed ``response`` (returned for every call - fine
    when a test only calls this once), or a ``responder`` callable for
    tests that need to vary the response by prompt/response_model.
    Never used by application runtime code.
    """

    model = "fake-structured-llm"

    def __init__(
        self,
        *,
        response: BaseModel | None = None,
        responder=None,
    ) -> None:
        if response is None and responder is None:
            raise ValueError(
                "FakeStructuredLLMProvider needs either `response` or "
                "`responder`."
            )

        self._response = response
        self._responder = responder
        self.calls: list[tuple[str, str, type[BaseModel]]] = []

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModelT],
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
    ) -> ResponseModelT:
        self.calls.append((system_prompt, user_prompt, response_model))

        result = (
            self._responder(system_prompt, user_prompt, response_model)
            if self._responder is not None
            else self._response
        )

        if not isinstance(result, response_model):
            raise LLMResponseValidationError(
                f"FakeStructuredLLMProvider was configured to return "
                f"{type(result).__name__}, but {response_model.__name__} "
                "was requested."
            )

        return result


class FailingStructuredLLMProvider:
    """Test-only provider that always raises - for exercising LLM-failure
    fallback paths directly."""

    model = "failing-structured-llm"

    def __init__(self, *, error: LLMError | None = None) -> None:
        self._error = error or LLMProviderError("synthetic failure for tests")

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModelT],
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
    ) -> ResponseModelT:
        raise self._error
