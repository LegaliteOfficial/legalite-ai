"""Per-tenant / per-IP rate limiting (audit Item 6).

The /ask endpoint costs OpenAI tokens; the upload endpoints cost CPU,
Qdrant writes, and storage. Without rate limiting a single tenant — or a
single forgetful client — can burn the LLM budget or wedge the worker
queue.

We use slowapi (the FastAPI port of Flask-Limiter). Limits are computed
per the most specific identity we have:

1. ``x-organization-id`` if present (post-HMAC, since :func:`tenant_key`
   reads the SAME header the dependency does — they cannot disagree).
2. Otherwise the remote IP address.

Defaults live in :class:`Settings` (``RATE_LIMIT_*``). The middleware can
be disabled in tests via ``RATE_LIMIT_ENABLED=false``.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.core.config import get_settings


def tenant_key(request: Request) -> str:
    """Identity used by the limiter to bucket requests.

    Prefers the gateway's ``x-organization-id`` header so noisy tenants
    rate-limit themselves without affecting other tenants. Falls back to
    the client IP for anonymous / legacy traffic.
    """
    org_id = request.headers.get("x-organization-id")
    if org_id:
        return f"org:{org_id.strip().lower()}"
    # request.client may be None in some ASGI test transports.
    client = getattr(request.client, "host", None)
    return f"ip:{client or 'unknown'}"


limiter = Limiter(key_func=tenant_key, default_limits=[])


def configure_rate_limiting(app: FastAPI) -> None:
    """Attach the rate-limit middleware + exception handler to the app.

    Safe to call when ``RATE_LIMIT_ENABLED=false`` — we simply leave the
    middleware off so tests / dev don't hit limits. The ``Limiter``
    object stays initialised because route decorators reference it at
    import time.
    """
    settings = get_settings()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    if settings.rate_limit_enabled:
        app.add_middleware(SlowAPIMiddleware)
