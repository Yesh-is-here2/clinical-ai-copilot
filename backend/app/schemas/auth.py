"""Auth request/response schemas (Phase 7, section 7.2).

Passwords are validated here, not just in the service: creation
requires at least 8 characters and at most bcrypt's 72-byte limit
(a multi-byte password can be 72 bytes in fewer than 72
characters). Login accepts any non-empty password - a wrong one
simply fails authentication.

No schema ever carries a password hash.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
)

from app.core.security import MAX_PASSWORD_BYTES, MIN_PASSWORD_LENGTH
from app.models.user import UserRole


class LoginRequest(BaseModel):
    """POST /auth/login body. JSON, not a form - see the route."""

    email: EmailStr
    password: str = Field(min_length=1)


class LoginResponse(BaseModel):
    """Issued access token (OAuth2-style fields, JSON body)."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds.")


class UserCreate(BaseModel):
    """POST /auth/users body (admin only)."""

    email: EmailStr
    password: str = Field(min_length=MIN_PASSWORD_LENGTH)
    full_name: str | None = Field(default=None, max_length=255)
    role: UserRole = UserRole.CLINICIAN

    @field_validator("password")
    @classmethod
    def _password_fits_bcrypt(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_PASSWORD_BYTES:
            raise ValueError(
                f"Password must be at most {MAX_PASSWORD_BYTES} bytes."
            )
        return value


class UserOut(BaseModel):
    """A user as the API returns it - never includes a hash."""

    id: uuid.UUID
    email: str
    full_name: str | None
    role: UserRole
    is_active: bool
    created_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class UserListResponse(BaseModel):
    """GET /auth/users response."""

    users: list[UserOut]
    total: int
