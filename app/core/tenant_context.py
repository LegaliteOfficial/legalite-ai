"""TenantContext — the authenticated org/user/role triple for a single request.

The org context arrives via headers set by the ``legalite-backend`` gateway:

    x-organization-id: <uuid>     # required for any private endpoint
    x-user-id:         <uuid>     # who initiated the request
    x-role:            <role>     # owner | admin | lawyer | associate | viewer

Trust model
-----------
This service trusts the gateway headers (deployment-level network isolation
is assumed — the FastAPI app is not exposed publicly; only the backend can
call it). If that assumption ever changes, swap the dependency in
``app/api/dependencies/tenant.py`` for a JWT- or HMAC-verifying one without
touching the rest of the codebase.

Rules enforced here
-------------------
- A TenantContext with ``organization_id == None`` represents the legacy /
  public-only path. Retrieval through such a context returns only GLOBAL
  knowledge.
- A TenantContext with ``organization_id`` set MUST have a user_id too;
  org-without-user is treated as missing and rejected at the dependency layer.
- ``role`` is parsed into the strict ``TenantRole`` enum where possible;
  unknown roles collapse to ``VIEWER`` (least-privileged) so a bad header
  cannot grant elevated access.

Why a contextvar
----------------
Some code paths (background workers, structlog binding) need the tenant
identity without explicit threading. ``current_tenant`` exposes it; the
request middleware will set it before dispatch and clear it after.
"""

from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass

from app.db.enums import TenantRole


@dataclass(slots=True, frozen=True)
class TenantContext:
    """Immutable per-request tenant identity.

    ``organization_id == None`` means "no tenant scope" — only GLOBAL
    knowledge is retrievable. This is the legacy / anonymous path.
    """

    organization_id: uuid.UUID | None
    user_id: uuid.UUID | None
    role: TenantRole

    # ---- shorthand -------------------------------------------------------

    @property
    def is_anonymous(self) -> bool:
        return self.organization_id is None

    @property
    def is_admin(self) -> bool:
        return self.role in (TenantRole.OWNER, TenantRole.ADMIN)

    @classmethod
    def anonymous(cls) -> TenantContext:
        """Legacy / no-auth context. Retrieval scoped to GLOBAL only."""
        return cls(organization_id=None, user_id=None, role=TenantRole.VIEWER)


# Set by the FastAPI dependency at the start of a request, read by services
# that don't want to thread the context through every signature (logging,
# background-job-launching, observability spans).
current_tenant: contextvars.ContextVar[TenantContext | None] = contextvars.ContextVar(
    "current_tenant", default=None
)


def set_current_tenant(ctx: TenantContext | None) -> contextvars.Token[TenantContext | None]:
    return current_tenant.set(ctx)


def reset_current_tenant(token: contextvars.Token[TenantContext | None]) -> None:
    current_tenant.reset(token)


def parse_role(raw: str | None) -> TenantRole:
    """Map a header value to a TenantRole. Unknowns collapse to VIEWER."""
    if not raw:
        return TenantRole.VIEWER
    normalized = raw.strip().lower()
    try:
        return TenantRole(normalized)
    except ValueError:
        return TenantRole.VIEWER


def parse_uuid(raw: str | None) -> uuid.UUID | None:
    """Lenient UUID parse — returns None on missing/blank/malformed values.

    We never raise here; the dependency layer decides whether a missing UUID
    is fatal (private endpoints) or just means "anonymous" (legacy).
    """
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    try:
        return uuid.UUID(candidate)
    except (ValueError, AttributeError):
        return None
