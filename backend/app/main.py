import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.clinical import router as clinical_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.patients import router as patients_router
from app.api.routes.queries import router as queries_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.db.session import close_database


configure_logging()

logger = logging.getLogger("clinical_ai.application")


@asynccontextmanager
async def lifespan(_: FastAPI):
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

app.include_router(health_router)
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

