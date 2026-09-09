from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.db.session import engine
from app.schemas.health import HealthResponse


router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health_check() -> HealthResponse:
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

            result = await connection.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM pg_extension
                        WHERE extname = 'vector'
                    )
                    """
                )
            )

            pgvector_available = bool(result.scalar())

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database health check failed.",
        ) from exc

    if not pgvector_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="pgvector extension is unavailable.",
        )

    return HealthResponse(
        status="healthy",
        service="clinical-ai-copilot-api",
        environment=settings.app_env,
        database="connected",
        pgvector="available",
        timestamp=datetime.now(UTC),
    )
