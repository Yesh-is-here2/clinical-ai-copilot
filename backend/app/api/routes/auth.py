"""Authentication endpoints (Phase 7, section 7.2).

- ``POST /auth/login`` - JSON body (email + password), returns a
  signed access token. Failed attempts are rate-limited per
  (client IP, email); a successful login clears that key.
- ``GET /auth/me`` - the caller's own profile.
- ``POST /auth/users`` / ``GET /auth/users`` - admin-only user
  management. No public registration exists by design.

No endpoint ever returns a password hash, and no endpoint logs a
password or a token.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Actor, get_current_user, get_db, get_settings, require_roles
from app.core.config import Settings
from app.core.rate_limit import LoginRateLimiter
from app.models.user import User, UserRole
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    UserCreate,
    UserListResponse,
    UserOut,
)
from app.services.auth_service import AuthService, DuplicateEmailError

router = APIRouter(prefix="/auth", tags=["auth"])


def _get_login_rate_limiter(
    request: Request, settings: Settings
) -> LoginRateLimiter:
    """The app-wide limiter, created in the lifespan (``app.main``).

    Falls back to building one from settings when the lifespan never
    ran (e.g. a test client that never entered the context manager),
    so login still rate-limits instead of silently skipping it."""

    limiter = getattr(request.app.state, "login_rate_limiter", None)
    if limiter is None:
        limiter = LoginRateLimiter(
            max_attempts=settings.login_rate_limit_max_attempts,
            window_seconds=settings.login_rate_limit_window_seconds,
        )
        request.app.state.login_rate_limiter = limiter
    return limiter


def _client_ip(request: Request) -> str:
    """Best-effort client address for the rate-limit key. No
    X-Forwarded-For trust: behind a proxy the direct peer is the
    honest value unless the deployment is configured otherwise."""

    return request.client.host if request.client is not None else "unknown"


@router.post("/login", response_model=LoginResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> LoginResponse:
    """Authenticate and issue an access token.

    A wrong email, a wrong password, and a deactivated account all
    produce the same 401 - the response never reveals which one."""

    service = AuthService(session, settings)
    limiter = _get_login_rate_limiter(request, settings)
    key = f"{_client_ip(request)}|{service.normalize_email(payload.email)}"

    if settings.login_rate_limit_enabled and not limiter.is_allowed(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Try again later.",
            headers={"Retry-After": str(settings.login_rate_limit_window_seconds)},
        )

    user = await service.authenticate(
        email=payload.email, password=payload.password
    )
    if user is None:
        if settings.login_rate_limit_enabled:
            limiter.record_failure(key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if settings.login_rate_limit_enabled:
        limiter.record_success(key)
    return LoginResponse(
        access_token=service.issue_token(user),
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.get("/me", response_model=UserOut)
async def read_current_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """The caller's own profile (any authenticated user)."""

    return current_user


@router.post(
    "/users", response_model=UserOut, status_code=status.HTTP_201_CREATED
)
async def create_user(
    payload: UserCreate,
    actor: Annotated[Actor, Depends(require_roles(UserRole.ADMIN))],
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Create a user (admin only). 409 when the email is taken."""

    del actor  # The dependency is the guard; the actor is not used.
    try:
        return await AuthService(session, settings).create_user(
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name,
            role=payload.role,
        )
    except DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.get("/users", response_model=UserListResponse)
async def list_users(
    actor: Annotated[Actor, Depends(require_roles(UserRole.ADMIN))],
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> UserListResponse:
    """List users (admin only), oldest first."""

    del actor  # The dependency is the guard; the actor is not used.
    users, total = await AuthService(session, settings).list_users(
        limit=limit, offset=offset
    )
    return UserListResponse(users=users, total=total)
