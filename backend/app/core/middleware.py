import logging
import time
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


logger = logging.getLogger("clinical_ai.request")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid4())

        request.state.request_id = request_id
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round(
                (time.perf_counter() - start_time) * 1000,
                2,
            )

            logger.exception(
                "Unhandled request exception",
                extra={
                    "request_id": request_id,
                    "duration_ms": duration_ms,
                },
            )
            raise

        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "%s %s completed with status %s",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "request_id": request_id,
                "duration_ms": duration_ms,
            },
        )

        return response
