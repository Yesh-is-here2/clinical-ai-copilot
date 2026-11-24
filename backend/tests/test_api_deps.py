"""Unit tests for ``app.api.deps`` (Milestone F, sections 12/13; Phase 7).

Covers the "lazy provider" contract explicitly required by the spec:
constructing the LLM provider, and asking for hybrid retrieval with
nothing ingested, must never touch the network or require
``OPENAI_API_KEY`` - and Phase 7's real actor: ``get_current_actor``
validates a Bearer JWT and loads the live user row (401 otherwise),
and ``require_roles`` gates roles (403). The token's role claim is
never trusted on its own: the actor's role always comes from the
database row.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api.deps import (
    Actor,
    get_current_actor,
    get_current_user,
    get_hybrid_retrieval_service,
    get_llm_provider,
    require_roles,
)
from app.clinical.llm import OpenAIStructuredLLMProvider
from app.core.config import Settings
from app.core.security import create_access_token
from app.models.user import User, UserRole


class _FakeResult:
    def all(self):
        return []


class _FakeAsyncSession:
    def __init__(self, user: User | None = None) -> None:
        self._user = user

    async def execute(self, stmt):
        return _FakeResult()

    async def get(self, model, user_id):
        # What UserRepository.get_by_id calls. Returns the configured
        # user regardless of the requested id - the tests only ever
        # configure one user, and the point here is the auth logic,
        # not the repository's SQL.
        return self._user


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


def _user(**overrides) -> User:
    # Explicit id: SQLAlchemy column defaults only fire at INSERT, so
    # a User built without a session has id None - a persisted user
    # always has one.
    values = {
        "id": uuid.uuid4(),
        "email": "clinician@example.com",
        "hashed_password": "hashed",
        "role": UserRole.CLINICIAN,
        "is_active": True,
    }
    values.update(overrides)
    return User(**values)


def _token_for(user: User, settings: Settings, **overrides) -> str:
    claims = {
        "subject": str(user.id),
        "email": user.email,
        "role": user.role.value,
        "secret": settings.jwt_secret,
        "expires_minutes": settings.access_token_expire_minutes,
    }
    claims.update(overrides)
    return create_access_token(**claims)


def _credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


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
async def test_get_current_user_rejects_missing_token() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(None, _settings(), _FakeAsyncSession())

    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_get_current_user_returns_live_row_for_valid_token() -> None:
    settings = _settings()
    user = _user()
    token = _token_for(user, settings)

    found = await get_current_user(_credentials(token), settings, _FakeAsyncSession(user))

    assert found is user


@pytest.mark.anyio
async def test_get_current_user_rejects_expired_token() -> None:
    settings = _settings()
    user = _user()
    token = _token_for(user, settings, expires_minutes=-1)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(_credentials(token), settings, _FakeAsyncSession(user))

    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_get_current_user_rejects_wrong_signature() -> None:
    settings = _settings()
    user = _user()
    token = _token_for(user, settings, secret="a-different-secret-that-is-long-enough")

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(_credentials(token), settings, _FakeAsyncSession(user))

    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_get_current_user_rejects_unknown_user() -> None:
    settings = _settings()
    user = _user()
    token = _token_for(user, settings)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(_credentials(token), settings, _FakeAsyncSession(None))

    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_get_current_user_rejects_deactivated_user() -> None:
    settings = _settings()
    user = _user(is_active=False)
    token = _token_for(user, settings)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(_credentials(token), settings, _FakeAsyncSession(user))

    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_get_current_actor_role_comes_from_database_not_token() -> None:
    # The actor is built from the live database row, never from the
    # token's role claim: a role change (or a forged claim) takes
    # effect on the next request.
    user = _user(role=UserRole.CLINICIAN)

    actor = await get_current_actor(user)

    assert actor.user_id == user.id
    assert actor.email == user.email
    assert actor.role is UserRole.CLINICIAN


@pytest.mark.anyio
async def test_require_roles_allows_matching_role() -> None:
    guard = require_roles(UserRole.CLINICIAN, UserRole.ADMIN)
    actor = Actor(
        user_id=uuid.uuid4(), email="c@example.com", role=UserRole.CLINICIAN
    )

    assert await guard(actor) is actor


@pytest.mark.anyio
async def test_require_roles_rejects_other_role() -> None:
    guard = require_roles(UserRole.ADMIN)
    actor = Actor(
        user_id=uuid.uuid4(), email="c@example.com", role=UserRole.CLINICIAN
    )

    with pytest.raises(HTTPException) as exc_info:
        await guard(actor)

    assert exc_info.value.status_code == 403


def test_actor_carries_id_email_and_role() -> None:
    user_id = uuid.uuid4()
    actor = Actor(user_id=user_id, email="r@example.com", role=UserRole.REVIEWER)

    assert actor.user_id == user_id
    assert actor.email == "r@example.com"
    assert actor.role is UserRole.REVIEWER
