"""Unit tests for Phase 7 security: password hashing / JWT primitives
(``app.core.security``), the login rate limiter
(``app.core.rate_limit``), ``AuthService``, the ``/auth/*`` routes,
production secret validation, and the middleware controls (security
headers, ``no-store`` on ``/auth/*``, 1 MB body limit).

No real Postgres anywhere: the auth service and routes run against a
small in-memory fake session, and tokens are real HS256 JWTs signed
with a test secret - never fakes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import jwt
from fastapi.testclient import TestClient

import app.main as main_module
from app.api.deps import Actor, get_current_actor, get_current_user, get_db, get_settings
from app.core.config import Settings
from app.core.rate_limit import LoginRateLimiter
from app.core.security import (
    TokenExpiredError,
    TokenInvalidError,
    create_access_token,
    decode_access_token,
    hash_password,
    validate_security_settings,
    verify_password,
)
from app.models.user import User, UserRole
from app.services.auth_service import AuthService, DuplicateEmailError

_TEST_SECRET = "test-secret-that-is-long-enough-for-tests-0123456789"
_USER_ID = uuid.uuid4()


def _settings(**overrides) -> Settings:
    base = {
        "database_url": "postgresql+asyncpg://user:pass@localhost:5432/db",
        "jwt_secret": _TEST_SECRET,
    }
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------
# Fake session (in-memory stand-in for an AsyncSession)
# ---------------------------------------------------------------------


class _FakeResult:
    def __init__(self, user: User | None = None, users: list[User] | None = None) -> None:
        self._user = user
        self._users = users or []

    def scalar_one_or_none(self) -> User | None:
        return self._user

    def scalar_one(self) -> int:
        return len(self._users)

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list[User]:
        return self._users


class _FakeSession:
    """Enough of an AsyncSession for the auth paths: ``get_by_email``
    (a ``users.email = '<email>'`` equality), ``get_by_id``,
    ``list_users``/``count_users``, ``add``/``flush``/``commit``."""

    def __init__(self, users: list[User] | None = None) -> None:
        self.users: dict[str, User] = {u.email: u for u in (users or [])}
        self.added: list[object] = []
        self.committed = False

    async def execute(self, stmt):
        where = getattr(stmt, "whereclause", None)
        if (
            where is not None
            and hasattr(where, "right")
            and hasattr(where.right, "value")
        ):
            return _FakeResult(user=self.users.get(where.right.value))
        return _FakeResult(users=list(self.users.values()))

    async def get(self, model, user_id):
        for user in self.users.values():
            if user.id == user_id:
                return user
        return None

    def add(self, instance: object) -> None:
        self.added.append(instance)

    async def flush(self) -> None:
        # Mirror what a real flush does: fire the Python-side column
        # defaults (id, is_active) that only exist after INSERT.
        for obj in self.added:
            if isinstance(obj, User):
                if obj.id is None:
                    obj.id = uuid.uuid4()
                if obj.is_active is None:
                    obj.is_active = True
        return None

    async def commit(self) -> None:
        self.committed = True
        for obj in self.added:
            if isinstance(obj, User):
                self.users[obj.email] = obj

    async def rollback(self) -> None:
        return None


def _make_user(email: str = "clinician@example.com", password: str = "correct horse", **overrides) -> User:
    # Explicit id: SQLAlchemy column defaults only fire at INSERT, so
    # a User built without a session has id None - a persisted user
    # always has one.
    values = {
        "id": uuid.uuid4(),
        "email": email,
        "hashed_password": hash_password(password),
        "role": UserRole.CLINICIAN,
        "is_active": True,
    }
    values.update(overrides)
    return User(**values)


def teardown_function() -> None:
    main_module.app.dependency_overrides.clear()
    if hasattr(main_module.app.state, "login_rate_limiter"):
        del main_module.app.state.login_rate_limiter


# ---------------------------------------------------------------------
# Password hashing / JWT primitives
# ---------------------------------------------------------------------


def test_hash_and_verify_round_trip() -> None:
    hashed = hash_password("correct horse battery staple")

    assert hashed != "correct horse battery staple"
    assert verify_password("correct horse battery staple", hashed) is True
    assert verify_password("wrong password", hashed) is False


def test_verify_password_never_raises_on_bad_hash() -> None:
    assert verify_password("anything", "not-a-bcrypt-hash") is False


def test_hash_password_rejects_empty_and_overlong() -> None:
    with pytest.raises(ValueError):
        hash_password("")
    with pytest.raises(ValueError):
        hash_password("x" * 73)


def test_token_round_trip() -> None:
    token = create_access_token(
        subject=str(_USER_ID),
        email="c@example.com",
        role="clinician",
        secret=_TEST_SECRET,
        expires_minutes=30,
    )

    payload = decode_access_token(token, secret=_TEST_SECRET)

    assert payload.subject == str(_USER_ID)
    assert payload.email == "c@example.com"
    assert payload.role == "clinician"


def test_token_rejects_wrong_secret() -> None:
    token = create_access_token(
        subject=str(_USER_ID),
        email="c@example.com",
        role="clinician",
        secret=_TEST_SECRET,
        expires_minutes=30,
    )

    with pytest.raises(TokenInvalidError):
        decode_access_token(token, secret="a-different-long-enough-secret-0123456789")


def test_token_rejects_expired() -> None:
    token = create_access_token(
        subject=str(_USER_ID),
        email="c@example.com",
        role="clinician",
        secret=_TEST_SECRET,
        expires_minutes=-1,
    )

    with pytest.raises(TokenExpiredError):
        decode_access_token(token, secret=_TEST_SECRET)


def test_token_rejects_malformed_and_wrong_type() -> None:
    with pytest.raises(TokenInvalidError):
        decode_access_token("not-a-token", secret=_TEST_SECRET)

    # A well-formed, unexpired token whose "type" claim is not
    # "access" must not authenticate - the type check, not expiry.
    now = datetime.now(UTC)
    tampered = jwt.encode(
        {
            "sub": str(_USER_ID),
            "email": "c@example.com",
            "role": "clinician",
            "type": "refresh",
            "iat": now,
            "exp": now + timedelta(minutes=30),
        },
        _TEST_SECRET,
        algorithm="HS256",
    )
    with pytest.raises(TokenInvalidError):
        decode_access_token(tampered, secret=_TEST_SECRET)


# ---------------------------------------------------------------------
# Production secret validation
# ---------------------------------------------------------------------


def test_validate_security_settings_allows_dev_placeholder() -> None:
    validate_security_settings(_settings(app_env="development"))


def test_validate_security_settings_rejects_production_placeholder() -> None:
    with pytest.raises(RuntimeError):
        validate_security_settings(
            _settings(app_env="production", jwt_secret="dev-only-insecure-change-me-in-production")
        )


def test_validate_security_settings_rejects_production_short_secret() -> None:
    with pytest.raises(RuntimeError):
        validate_security_settings(
            _settings(app_env="production", jwt_secret="too-short")
        )


def test_validate_security_settings_accepts_production_long_secret() -> None:
    validate_security_settings(
        _settings(app_env="production", jwt_secret="a" * 40)
    )


def test_validate_security_settings_rejects_production_bad_algorithm() -> None:
    with pytest.raises(RuntimeError):
        validate_security_settings(
            _settings(app_env="production", jwt_secret="a" * 40, jwt_algorithm="none")
        )


# ---------------------------------------------------------------------
# Login rate limiter
# ---------------------------------------------------------------------


def test_rate_limiter_allows_up_to_max_then_blocks() -> None:
    limiter = LoginRateLimiter(max_attempts=2, window_seconds=60)

    assert limiter.is_allowed("k") is True
    limiter.record_failure("k")
    assert limiter.is_allowed("k") is True
    limiter.record_failure("k")
    assert limiter.is_allowed("k") is False


def test_rate_limiter_success_clears_failures() -> None:
    limiter = LoginRateLimiter(max_attempts=1, window_seconds=60)

    limiter.record_failure("k")
    assert limiter.is_allowed("k") is False
    limiter.record_success("k")
    assert limiter.is_allowed("k") is True


def test_rate_limiter_window_expires(monkeypatch) -> None:
    limiter = LoginRateLimiter(max_attempts=1, window_seconds=60)
    now = [1000.0]
    monkeypatch.setattr("app.core.rate_limit.time.monotonic", lambda: now[0])

    limiter.record_failure("k")
    assert limiter.is_allowed("k") is False
    now[0] = 1000.0 + 61  # past the window
    assert limiter.is_allowed("k") is True


# ---------------------------------------------------------------------
# AuthService (fake session)
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_authenticate_succeeds_with_valid_credentials() -> None:
    user = _make_user()
    service = AuthService(_FakeSession([user]), _settings())

    assert await service.authenticate(email=user.email, password="correct horse") is user


@pytest.mark.anyio
async def test_authenticate_rejects_wrong_password_unknown_email_and_inactive() -> None:
    user = _make_user()
    inactive = _make_user(email="off@example.com", is_active=False)
    service = AuthService(_FakeSession([user, inactive]), _settings())

    assert await service.authenticate(email=user.email, password="nope") is None
    assert await service.authenticate(email="nobody@example.com", password="correct horse") is None
    assert await service.authenticate(email=inactive.email, password="correct horse") is None


@pytest.mark.anyio
async def test_authenticate_normalizes_email_case() -> None:
    user = _make_user()
    service = AuthService(_FakeSession([user]), _settings())

    assert (
        await service.authenticate(email="  CLINICIAN@EXAMPLE.COM ", password="correct horse")
        is user
    )


@pytest.mark.anyio
async def test_create_user_normalizes_email_and_commits() -> None:
    session = _FakeSession()
    service = AuthService(session, _settings())

    user = await service.create_user(
        email="  New@Example.COM ",
        password="correct horse",
        full_name="New Clinician",
        role=UserRole.REVIEWER,
    )

    assert user.email == "new@example.com"
    assert user.role is UserRole.REVIEWER
    assert session.committed is True


@pytest.mark.anyio
async def test_create_user_rejects_duplicate_email() -> None:
    user = _make_user()
    service = AuthService(_FakeSession([user]), _settings())

    with pytest.raises(DuplicateEmailError):
        await service.create_user(
            email="CLINICIAN@example.com",
            password="correct horse",
            full_name=None,
            role=UserRole.CLINICIAN,
        )


@pytest.mark.anyio
async def test_issue_token_round_trips_through_decode() -> None:
    user = _make_user()
    settings = _settings()
    service = AuthService(_FakeSession([user]), settings)

    token = service.issue_token(user)
    payload = decode_access_token(token, secret=settings.jwt_secret)

    assert payload.subject == str(user.id)
    assert payload.role == "clinician"


# ---------------------------------------------------------------------
# /auth/* routes
# ---------------------------------------------------------------------


def _override_session(session: _FakeSession) -> None:
    async def _override_get_db():
        yield session

    main_module.app.dependency_overrides[get_db] = _override_get_db


def _override_settings(settings: Settings) -> None:
    main_module.app.dependency_overrides[get_settings] = lambda: settings


def _as_actor(actor: Actor) -> None:
    main_module.app.dependency_overrides[get_current_actor] = lambda: actor


def test_login_success_returns_decodable_token() -> None:
    user = _make_user()
    settings = _settings()
    _override_session(_FakeSession([user]))
    _override_settings(settings)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/auth/login",
            json={"email": user.email, "password": "correct horse"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == settings.access_token_expire_minutes * 60
    payload = decode_access_token(body["access_token"], secret=settings.jwt_secret)
    assert payload.subject == str(user.id)
    assert "Cache-Control" in response.headers
    assert response.headers["Cache-Control"] == "no-store"


def test_login_wrong_password_returns_401() -> None:
    user = _make_user()
    _override_session(_FakeSession([user]))
    _override_settings(_settings())

    with TestClient(main_module.app) as client:
        response = client.post(
            "/auth/login", json={"email": user.email, "password": "nope"}
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."


def test_login_unknown_email_returns_same_401() -> None:
    _override_session(_FakeSession([]))
    _override_settings(_settings())

    with TestClient(main_module.app) as client:
        response = client.post(
            "/auth/login", json={"email": "nobody@example.com", "password": "whatever"}
        )

    assert response.status_code == 401


def test_login_rate_limit_blocks_after_max_failures_and_success_resets() -> None:
    user = _make_user()
    settings = _settings(login_rate_limit_max_attempts=2)
    _override_session(_FakeSession([user]))
    _override_settings(settings)

    with TestClient(main_module.app) as client:
        # The lifespan built a limiter from the real settings (max 10);
        # drop it so the first login builds one from the overridden
        # settings (max 2).
        if hasattr(main_module.app.state, "login_rate_limiter"):
            del main_module.app.state.login_rate_limiter

        assert (
            client.post("/auth/login", json={"email": user.email, "password": "nope"}).status_code
            == 401
        )
        # A good login clears the failure count.
        assert (
            client.post(
                "/auth/login", json={"email": user.email, "password": "correct horse"}
            ).status_code
            == 200
        )
        assert (
            client.post("/auth/login", json={"email": user.email, "password": "nope"}).status_code
            == 401
        )
        assert (
            client.post("/auth/login", json={"email": user.email, "password": "nope"}).status_code
            == 401
        )
        blocked = client.post(
            "/auth/login", json={"email": user.email, "password": "nope"}
        )
        assert blocked.status_code == 429
        assert "Retry-After" in blocked.headers


def test_auth_me_returns_current_user() -> None:
    user = _make_user()
    main_module.app.dependency_overrides[get_current_user] = lambda: user

    with TestClient(main_module.app) as client:
        response = client.get("/auth/me")

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == user.email
    assert body["role"] == "clinician"
    assert "hashed_password" not in body


def test_create_user_admin_only_and_duplicate_is_409() -> None:
    session = _FakeSession()
    _override_session(session)
    _override_settings(_settings())
    _as_actor(
        Actor(user_id=uuid.uuid4(), email="admin@example.com", role=UserRole.ADMIN)
    )

    with TestClient(main_module.app) as client:
        created = client.post(
            "/auth/users",
            json={
                "email": "new@example.com",
                "password": "correct horse",
                "full_name": "New Reviewer",
                "role": "reviewer",
            },
        )
        assert created.status_code == 201
        assert created.json()["email"] == "new@example.com"
        assert "hashed_password" not in created.json()

        duplicate = client.post(
            "/auth/users",
            json={"email": "new@example.com", "password": "correct horse"},
        )
        assert duplicate.status_code == 409


def test_create_user_rejects_clinician_and_unauthenticated() -> None:
    _override_session(_FakeSession())
    _override_settings(_settings())
    _as_actor(
        Actor(user_id=uuid.uuid4(), email="c@example.com", role=UserRole.CLINICIAN)
    )

    with TestClient(main_module.app) as client:
        assert (
            client.post(
                "/auth/users",
                json={"email": "new@example.com", "password": "correct horse"},
            ).status_code
            == 403
        )

    main_module.app.dependency_overrides.pop(get_current_actor, None)
    with TestClient(main_module.app) as client:
        assert (
            client.post(
                "/auth/users",
                json={"email": "new@example.com", "password": "correct horse"},
            ).status_code
            == 401
        )


def test_list_users_admin_only() -> None:
    user = _make_user()
    _override_session(_FakeSession([user]))
    _override_settings(_settings())
    _as_actor(
        Actor(user_id=uuid.uuid4(), email="admin@example.com", role=UserRole.ADMIN)
    )

    with TestClient(main_module.app) as client:
        response = client.get("/auth/users")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["users"][0]["email"] == user.email


def test_login_rejects_short_password_with_422() -> None:
    _override_session(_FakeSession([]))
    _override_settings(_settings())

    with TestClient(main_module.app) as client:
        response = client.post(
            "/auth/login", json={"email": "a@example.com", "password": ""}
        )

    assert response.status_code == 422


def test_login_never_leaks_token_or_password_into_logs(caplog) -> None:
    user = _make_user()
    _override_session(_FakeSession([user]))
    _override_settings(_settings())

    with caplog.at_level("INFO", logger="clinical_ai"):
        with TestClient(main_module.app) as client:
            response = client.post(
                "/auth/login",
                json={"email": user.email, "password": "correct horse"},
            )

    assert response.status_code == 200
    token = response.json()["access_token"]
    assert token not in caplog.text
    assert "correct horse" not in caplog.text


# ---------------------------------------------------------------------
# Middleware controls
# ---------------------------------------------------------------------


def test_security_headers_present_on_response() -> None:
    # GET / is the one route that touches no database (unlike
    # /health, which pings Postgres and needs a live one).
    with TestClient(main_module.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["X-Request-ID"]
    assert "Cache-Control" not in response.headers  # no-store is /auth/* only


def test_body_over_1mb_returns_413() -> None:
    with TestClient(main_module.app) as client:
        response = client.post(
            "/clinical/query",
            content=b'{"question": "' + b"x" * (1_048_577) + b'"}',
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413
