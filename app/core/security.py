"""Security helpers.

Kept intentionally small. Authentication (JWT / API keys) is stubbed here so
routes can depend on it from day one; swap in a real provider later without
touching call sites.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.config import get_settings


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reserved for future protected endpoints.

    Currently a no-op in development. Wire it into route dependencies once an
    auth strategy is chosen (API key, JWT, OAuth, ...).
    """
    settings = get_settings()
    if not settings.is_production:
        return
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )
