"""User persistence (Phase 7, section 7.1).

Thin async repository over ``app.models.user.User``. It never commits:
the caller (``app.services.auth_service.AuthService``) owns the
transaction, so creating a user stays atomic with whatever else the
service does in the same request.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User, UserRole


class UserRepository:
    """User lookups and creation."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_email(self, email: str) -> User | None:
        """Fetch a user by email. Callers pass the already-normalized
        (lowercased, stripped) address - see AuthService."""

        stmt = select(User).where(User.email == email)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        """Fetch a user by primary key."""

        return await self._session.get(User, user_id)

    async def create_user(
        self,
        *,
        email: str,
        hashed_password: str,
        full_name: str | None,
        role: UserRole,
    ) -> User:
        """Insert a user and flush so the returned row carries its id.
        The caller commits."""

        user = User(
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            role=role,
        )
        self._session.add(user)
        await self._session.flush()
        return user

    async def list_users(
        self, *, limit: int = 100, offset: int = 0
    ) -> list[User]:
        """All users, oldest first (stable order for pagination)."""

        stmt = (
            select(User)
            .order_by(User.created_at, User.email)
            .limit(limit)
            .offset(offset)
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def count_users(self) -> int:
        """Total user count, for the admin list response."""

        stmt = select(func.count()).select_from(User)
        return int((await self._session.execute(stmt)).scalar_one())
