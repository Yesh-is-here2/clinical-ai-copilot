"""Password hashing, JWT handling, and production secret validation
(Phase 7, sections 7.1/7.6).

Deliberately small and dependency-light: ``bcrypt`` for password
hashing (maintained; ``passlib`` is not) and ``PyJWT`` for tokens.
No fake JWT logic anywhere - tokens are real HS256-signed JWTs, and
anything that fails validation is rejected, never "best-effort"
accepted.

Nothing in this module ever logs a password, a password hash, or a
token.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from app.core.config import Settings

# The development placeholder shipped in .env.example / Settings.
# Production startup rejects it (see validate_security_settings).
DEV_ONLY_SECRET_PLACEHOLDER = "dev-only-insecure-change-me-in-production"

# Anything a human might plausibly leave in place of a real secret.
INSECURE_SECRETS = frozenset(
    {
        "",
        "change-me",
        "changeme",
        "secret",
        "test",
        "test-secret",
        "dev-secret",
        "development",
        "password",
        "replace-with-a-secure-random-secret",
        DEV_ONLY_SECRET_PLACEHOLDER,
    }
)

MIN_SECRET_LENGTH = 32
SUPPORTED_JWT_ALGORITHM = "HS256"
MAX_PASSWORD_BYTES = 72  # bcrypt's own limit; longer input is rejected, never truncated.
MIN_PASSWORD_LENGTH = 8


class TokenValidationError(ValueError):
    """Base class for every token rejection: expired, malformed, wrong
    signature, wrong type, or a subject that is not a user id."""


class TokenExpiredError(TokenValidationError):
    """The token's ``exp`` claim is in the past."""


class TokenInvalidError(TokenValidationError):
    """The token is malformed, has a bad signature, was signed with an
    unsupported algorithm, or carries unexpected claims."""


@dataclass(frozen=True)
class TokenPayload:
    """The claims a validated access token carries. ``subject`` is the
    user's UUID as a string; ``role`` is the ``UserRole`` value."""

    subject: str
    email: str
    role: str


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt.

    Raises ``ValueError`` on an empty password or one longer than
    bcrypt's 72-byte limit (truncating would silently accept two
    different passwords as the same one). Never returns or logs the
    input.
    """

    if not password:
        raise ValueError("Password must not be empty.")
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes."
        )
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Bcrypt check. Returns ``False`` (never raises) for a wrong
    password, an overlong input, or a malformed stored hash, so a bad
    row can never turn a login attempt into a 500."""

    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except (ValueError, TypeError):
        return False


def create_access_token(
    *,
    subject: str,
    email: str,
    role: str,
    secret: str,
    algorithm: str = SUPPORTED_JWT_ALGORITHM,
    expires_minutes: int,
) -> str:
    """Issue a signed access token. ``subject`` is the user's UUID."""

    if algorithm != SUPPORTED_JWT_ALGORITHM:
        raise ValueError(f"Unsupported JWT algorithm: {algorithm!r}.")

    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "email": email,
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_access_token(
    token: str, *, secret: str, algorithm: str = SUPPORTED_JWT_ALGORITHM
) -> TokenPayload:
    """Validate a token and return its claims. Raises
    ``TokenExpiredError`` / ``TokenInvalidError`` - never returns a
    partial payload."""

    if algorithm != SUPPORTED_JWT_ALGORITHM:
        raise TokenInvalidError(f"Unsupported JWT algorithm: {algorithm!r}.")

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[algorithm],
            options={"require": ["exp", "iat", "sub", "type"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpiredError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenInvalidError("Token is invalid.") from exc

    if payload.get("type") != "access":
        raise TokenInvalidError("Token is not an access token.")

    subject = payload.get("sub")
    email = payload.get("email")
    role = payload.get("role")
    if not isinstance(subject, str) or not subject:
        raise TokenInvalidError("Token has no subject.")
    try:
        uuid.UUID(subject)
    except (ValueError, AttributeError) as exc:
        raise TokenInvalidError("Token subject is not a user id.") from exc
    if not isinstance(email, str) or not email:
        raise TokenInvalidError("Token has no email.")
    if not isinstance(role, str) or not role:
        raise TokenInvalidError("Token has no role.")

    return TokenPayload(subject=subject, email=email, role=role)


def validate_security_settings(settings: Settings) -> None:
    """Reject placeholder or unsafe secrets - production only.

    Called from the application lifespan (``app.main``). In
    development/testing the insecure default is tolerated so the app
    and the test suite work with no ``.env`` at all; in production a
    missing, placeholder, or short secret (or an unsupported
    algorithm) is a startup error, not a warning - the app must not
    start and silently sign tokens with a guessable key.
    """

    if settings.app_env != "production":
        return

    secret = settings.jwt_secret
    if secret in INSECURE_SECRETS or len(secret) < MIN_SECRET_LENGTH:
        raise RuntimeError(
            "Production startup refused: JWT secret is missing, a known "
            "placeholder, or shorter than "
            f"{MIN_SECRET_LENGTH} characters. Set a long random secret."
        )
    if settings.jwt_algorithm != SUPPORTED_JWT_ALGORITHM:
        raise RuntimeError(
            "Production startup refused: unsupported jwt_algorithm "
            f"{settings.jwt_algorithm!r} (only HS256 is supported)."
        )
