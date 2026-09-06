"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.logging import get_logger
from app.schemas.documents import HealthResponse

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    # The app itself being reachable is reported as "ok" regardless of
    # Qdrant's state — a transient vector-store outage shouldn't make
    # orchestrators (Railway, Docker) restart an otherwise-healthy process.
    # The `qdrant` field carries the actual dependency status for monitoring.
    qdrant_status = "ok"
    try:
        from app.vector_store.qdrant_client import qdrant_is_reachable

        if not qdrant_is_reachable():
            qdrant_status = "unavailable"
    except Exception:  # noqa: BLE001
        logger.exception("health.qdrant_check_failed")
        qdrant_status = "unavailable"

    return HealthResponse(status="ok", qdrant=qdrant_status)
