"""FastAPI dependencies that turn gateway headers into a ``TenantContext``.

Three flavors:

- :func:`get_optional_tenant_context` — accepts requests with or without
  headers. Missing headers return ``TenantContext.anonymous()``. Use this on
  legacy routes (`/ask`, `/upload-law`) so the existing client keeps working.

- :func:`get_required_tenant_context` — 401 if no ``x-organization-id``.
  Use on tenant-scoped endpoints (`/documents/private-upload`,
  `/admin/upload-law`).

- :func:`require_admin` — additionally requires the role be ``owner`` or
  ``admin``. Use on admin-only routes.

Service-to-service auth (audit Item 8): when
``LEGALITE_GATEWAY_SECRET`` is set, every request that asserts a tenant
identity MUST also carry a valid HMAC signature
(:mod:`app.core.service_auth`). Anonymous requests are unaffected.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.service_auth import verify_gateway_signature
from app.core.tenant_context import (
    TenantContext,
    parse_role,
    parse_uuid,
)
from app.db.enums import GLOBAL_INGEST_ROLES


async def get_optional_tenant_context(
    x_organization_id: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_legalite_signature: str | None = Header(default=None),
    x_legalite_timestamp: str | None = Header(default=None),
) -> TenantContext:
    """Build a TenantContext from headers, falling back to anonymous.

    Anonymous = no org_id header at all → retrieval scoped to GLOBAL only.
    Used by the legacy /ask and /upload-law endpoints for backward compat.
    """
    verify_gateway_signature(
        timestamp=x_legalite_timestamp,
        signature=x_legalite_signature,
        organization_id=x_organization_id,
        user_id=x_user_id,
        role=x_role,
    )
    org_id = parse_uuid(x_organization_id)
    if org_id is None:
        return TenantContext.anonymous()
    return TenantContext(
        organization_id=org_id,
        user_id=parse_uuid(x_user_id),
        role=parse_role(x_role),
    )


async def get_required_tenant_context(
    x_organization_id: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_legalite_signature: str | None = Header(default=None),
    x_legalite_timestamp: str | None = Header(default=None),
) -> TenantContext:
    """Require an organization_id header; 401 otherwise."""
    verify_gateway_signature(
        timestamp=x_legalite_timestamp,
        signature=x_legalite_signature,
        organization_id=x_organization_id,
        user_id=x_user_id,
        role=x_role,
    )
    org_id = parse_uuid(x_organization_id)
    if org_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing x-organization-id header — tenant context required",
        )
    user_id = parse_uuid(x_user_id)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing x-user-id header — user identity required",
        )
    return TenantContext(
        organization_id=org_id,
        user_id=user_id,
        role=parse_role(x_role),
    )


async def require_admin(
    x_organization_id: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_legalite_signature: str | None = Header(default=None),
    x_legalite_timestamp: str | None = Header(default=None),
) -> TenantContext:
    """Require an admin/owner role. Use on /admin/* endpoints."""
    ctx = await get_required_tenant_context(
        x_organization_id,
        x_user_id,
        x_role,
        x_legalite_signature,
        x_legalite_timestamp,
    )
    if ctx.role not in GLOBAL_INGEST_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Admin role required (got {ctx.role.value!r})",
        )
    return ctx
