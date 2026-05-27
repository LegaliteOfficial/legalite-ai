"""HMAC verification for gateway headers (audit Item 8).

The Phase 4 trust model accepted ``x-organization-id`` / ``x-user-id`` /
``x-role`` headers at face value. That is correct only if this service is
unreachable except via the backend gateway. Service-to-service auth
removes that "only if" — the backend signs the tenant tuple, we verify.

Wire format
-----------
The backend includes two extra headers on every request to this service::

    x-legalite-timestamp: 1748390401            # unix seconds
    x-legalite-signature: <hex sha256 hmac>

The signed message is::

    f"{timestamp}.{organization_id}.{user_id}.{role}"

with empty string ("") substituted for any missing value (e.g. anonymous
requests have org/user/role all empty). Both sides MUST normalise UUIDs
to lowercase. Timestamps older than
``GATEWAY_SIGNATURE_MAX_SKEW_SECONDS`` (default 300) are rejected to
prevent replay.

Mode of operation
-----------------
- If ``LEGALITE_GATEWAY_SECRET`` is unset → verification is bypassed
  (Phase 4 trust-the-network mode). A warning is logged once per process.
- If set → every authenticated request MUST carry a valid signature.
  Requests with no tenant headers at all are treated as anonymous and do
  not require a signature, preserving the legacy /ask / /upload-law path.
"""

from __future__ import annotations

import hmac
import time
from hashlib import sha256

from fastapi import HTTPException, status

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def signature_required() -> bool:
    """True iff a shared secret is configured. When False, this module is a no-op."""
    return bool(get_settings().legalite_gateway_secret)


def compute_signature(
    *,
    timestamp: str | int,
    organization_id: str | None,
    user_id: str | None,
    role: str | None,
    secret: str | None = None,
) -> str:
    """Return the canonical hex-digest signature for the given tenant tuple.

    Exposed so the backend's test suite (and any operator script) can
    sign requests with the same primitive — preventing the kind of
    "implemented differently on each side" drift that silently breaks
    HMAC.
    """
    s = secret if secret is not None else get_settings().legalite_gateway_secret or ""
    msg = f"{timestamp}.{organization_id or ''}.{user_id or ''}.{role or ''}"
    return hmac.new(s.encode("utf-8"), msg.encode("utf-8"), sha256).hexdigest()


def verify_gateway_signature(
    *,
    timestamp: str | None,
    signature: str | None,
    organization_id: str | None,
    user_id: str | None,
    role: str | None,
) -> None:
    """Verify the HMAC + freshness, or raise HTTP 401.

    Called by the tenant FastAPI dependencies BEFORE constructing a
    ``TenantContext``. If the secret isn't configured the function
    returns silently — the deployment is in legacy network-trust mode.

    Anonymous requests (no organization_id) MAY skip the signature; we
    only enforce when a tenant identity is being asserted. This keeps the
    legacy /ask + /upload-law contracts unbroken.
    """
    settings = get_settings()
    secret = settings.legalite_gateway_secret
    if not secret:
        return  # legacy trust mode

    # Anonymous request → no signature needed.
    if not organization_id:
        if signature or timestamp:
            # Operator sent partial credentials; refuse rather than half-trust.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="signature headers present without organization_id",
            )
        return

    if not signature or not timestamp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing x-legalite-signature or x-legalite-timestamp",
        )

    # Freshness — reject replays.
    try:
        ts_int = int(timestamp)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="x-legalite-timestamp must be an integer unix timestamp",
        ) from exc
    skew = abs(time.time() - ts_int)
    if skew > settings.gateway_signature_max_skew_seconds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"x-legalite-timestamp out of skew (|Δ|={int(skew)}s)",
        )

    # Normalise so the backend and this service compute the same digest.
    expected = compute_signature(
        timestamp=ts_int,
        organization_id=(organization_id or "").lower(),
        user_id=(user_id or "").lower(),
        role=(role or "").lower(),
        secret=secret,
    )
    # Constant-time compare prevents timing side-channels.
    if not hmac.compare_digest(expected, signature.lower()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid gateway signature",
        )
