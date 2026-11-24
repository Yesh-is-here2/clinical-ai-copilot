"""Authentication and admin user management (Phase 7, sections 7.1-7.2).

The service owns the user transaction: it commits after a successful
create, and rolls back (via the caller's session lifecycle) on any
failure. It never logs a password or a token.

Security notes:

- Emails are normalized (stripped, lowercased) before lookup and
  storage, so ``Clinician@Example.com`` and ``clinician@example.com``
  are the same account.
- ``authenticate`` returns ``None`` for an unknown email, a wrong
  password, *and* a deactivated account - the caller cannot tell
  which, so the endpoint cannot be used to enumerate accounts.
  A dummy bcrypt check runs for unknown emails so the timing does
  not leak whether the account exists.
- Duplicate emails raise ``DuplicateEmailError`` (mapped to 409 by
  the auth route); the database unique constraint is the backstop
  for a check-then-insert race.
"""

from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.db.repositories.user_repository import UserRepository
from app.models.user import User, UserRole

# Bcrypt hash of a value no real user ever has. Checked (and
# discarded) when the email is unknown, so a login attempt against a
# nonexistent account costs the same bcrypt work as a real one.
# This is a hash, not a secret - it authenticates nobody.
_DUMMY_HASH = "$2b$12$JAlPMwvuEjal.d1AUau4LOXTX8nGHNYwXxYeswyA1eaCu3mMH6hZy"


class DuplicateEmailError(ValueError):
    """An account with this email already exists."""


class AuthService:
    """Login, token issuance, and admin user management."""

    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._users = UserRepository(session)

    @staticmethod
    def normalize_email(email: str) -> str:
        """Strip and lowercase: the canonical form for storage and
        lookup."""

        return email.strip().lower()

    async def authenticate(self, *, email: str, password: str) -> User | None:
        """Return the user when the credentials are valid and the
        account is active; ``None`` otherwise (unknown email, wrong
        password, or deactivated account - indistinguishable)."""

        user = await self._users.get_by_email(self.normalize_email(email))
        if user is None:
            verify_password(password, _DUMMY_HASH)
            return None
        if not user.is_active:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def issue_token(self, user: User) -> str:
        """Issue a signed access token for an authenticated user."""

        return create_access_token(
            subject=str(user.id),
            email=user.email,
            role=user.role.value,
            secret=self._settings.jwt_secret,
            algorithm=self._settings.jwt_algorithm,
            expires_minutes=self._settings.access_token_expire_minutes,
        )

    async def create_user(
        self,
        *,
        email: str,
        password: str,
        full_name: str | None,
        role: UserRole,
    ) -> User:
        """Create a user. Raises ``DuplicateEmailError`` when the email
        is taken, ``ValueError`` from ``hash_password`` when the
        password is empty or longer than bcrypt's 72-byte limit."""

        normalized = self.normalize_email(email)
        existing = await self._users.get_by_email(normalized)
        if existing is not None:
            raise DuplicateEmailError(
                f"An account already exists for {normalized}."
            )
        user = await self._users.create_user(
            email=normalized,
            hashed_password=hash_password(password),
            full_name=full_name,
            role=role,
        )
        try:
            await self._session.commit()
        except IntegrityError as exc:
            await self._session.rollback()
            raise DuplicateEmailError(
                f"An account already exists for {normalized}."
            ) from exc
        return user

    async def list_users(
        self, *, limit: int = 100, offset: int = 0
    ) -> tuple[list[User], int]:
        """All users plus the total count, for the admin list."""

        users = await self._users.list_users(limit=limit, offset=offset)
        total = await self._users.count_users()
        return users, total
