"""GET /metrics — Prometheus exposition endpoint.

Mounted unconditionally; gated by ``settings.prometheus_enabled`` only at
collection time (the endpoint is always there so monitoring tooling can
still confirm reachability).

Deployment note
---------------
In production, expose this on the **internal** network only. The metrics
include per-tenant labels (``organization_id``) that should not be public.
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from app.core.config import get_settings
from app.core.telemetry import render_metrics

router = APIRouter(prefix="", tags=["telemetry"])


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    settings = get_settings()
    if not settings.prometheus_enabled:
        return Response(
            content="# metrics disabled\n",
            media_type="text/plain; version=0.0.4",
        )
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)
