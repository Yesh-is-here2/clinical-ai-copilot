import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.auth import router as auth_router
from app.api.routes.clinical import router as clinical_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.patients import router as patients_router
from app.api.routes.queries import router as queries_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.core.rate_limit import LoginRateLimiter
from app.core.security import validate_security_settings
from app.db.session import close_database


configure_logging()

logger = logging.getLogger("clinical_ai.application")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast on an unsafe production secret before serving
    # anything (Phase 7, section 7.6): the app must not start and
    # silently sign tokens with a guessable key.
    validate_security_settings(settings)
    app.state.login_rate_limiter = LoginRateLimiter(
        max_attempts=settings.login_rate_limit_max_attempts,
        window_seconds=settings.login_rate_limit_window_seconds,
    )

    logger.info(
        "Starting %s version %s",
        settings.app_name,
        settings.app_version,
    )

    yield

    logger.info("Shutting down %s", settings.app_name)
    await close_database()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Educational and research clinical AI decision-support prototype. "
        "Only synthetic, public, or de-identified data may be used."
    ),
    lifespan=lifespan,
)

app.add_middleware(RequestContextMiddleware)

# Browser cross-origin access is closed by default (Phase 7, section
# 7.6): CORSMiddleware is only installed when cors_allow_origins is
# configured, and then only for the named origins. Added after
# RequestContextMiddleware so it runs first (outermost) and preflight
# requests are answered before any other middleware sees them.
if settings.cors_allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(patients_router)
app.include_router(clinical_router)
app.include_router(queries_router)
app.include_router(documents_router)


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "docs": "/docs",
        "health": "/health",
    }

