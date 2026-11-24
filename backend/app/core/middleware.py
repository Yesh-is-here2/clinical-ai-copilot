"""Request-context middleware (Milestone F, section 13; Phase 7,
section 7.6).

Assigns every request an id (``X-Request-ID``, echoed back on the
response), logs method/path/status/duration, and - since Phase 7 -
enforces the security controls that apply to all traffic:

- a 1 MB request-body limit, enforced before the route ever sees the
  body (a declared-huge ``Content-Length`` is rejected without
  reading a byte; a chunked body is capped while streaming, reading
  at most limit + 1 bytes);
- response security headers on every response (``nosniff``, ``DENY``
  framing, ``no-referrer``), plus ``no-store`` on ``/auth/*`` so
  tokens are never cached.

The accepted body is cached on the request (see
``_read_bounded_body``) so downstream routes still read it normally:
Starlette's ``BaseHTTPMiddleware`` replays a set ``request._body``
to the downstream app, while a stream read to completion *without*
caching it would hand downstream an empty body.
"""

import logging
import time
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response


logger = logging.getLogger("clinical_ai.request")

# Phase 7, section 7.6: the largest legitimate body this API accepts
# is a clinical question, so 1 MB is generous and still stops a
# client from forcing the server to buffer an unbounded upload.
MAX_REQUEST_BODY_BYTES = 1_048_576

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


def _security_headers(request: Request) -> dict[str, str]:
    """Security headers for a response. ``/auth/*`` additionally gets
    ``no-store``: tokens must never sit in any cache."""

    headers = dict(_SECURITY_HEADERS)
    if request.url.path.startswith("/auth"):
        headers["Cache-Control"] = "no-store"
    return headers


async def _read_bounded_body(request: Request) -> bytes | None:
    """Read the request body, bounded by ``MAX_REQUEST_BODY_BYTES``.

    Returns the body when it fits, or ``None`` when it exceeds the
    limit (the caller answers 413). The body is cached on the
    request (``request._body``) - see the module docstring for why
    that assignment is load-bearing."""

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return None
        except ValueError:
            pass  # Malformed header: fall through to the streaming cap.

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_REQUEST_BODY_BYTES:
            return None
        chunks.append(chunk)
    body = b"".join(chunks)
    request._body = body  # noqa: SLF001 - see module docstring.
    return body


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid4())

        request.state.request_id = request_id
        start_time = time.perf_counter()

        if await _read_bounded_body(request) is None:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large."},
                headers={"X-Request-ID": request_id, **_security_headers(request)},
            )

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
        response.headers.update(_security_headers(request))

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
