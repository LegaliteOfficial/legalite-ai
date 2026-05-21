"""Observability middleware: request id + latency + structured context.

Each request gets a UUID that is bound into structlog's contextvars, so every
log line emitted while handling the request carries `request_id`. Also emits
an `http.request` log with method, path, status, and latency.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "http.request_failed",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
            raise
        latency_ms = int((time.perf_counter() - t0) * 1000)
        response.headers["x-request-id"] = request_id
        logger.info(
            "http.request",
            status=response.status_code,
            latency_ms=latency_ms,
        )
        return response
