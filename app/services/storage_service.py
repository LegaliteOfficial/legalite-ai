"""Supabase Storage adapter for original-PDF persistence.

Why a thin adapter instead of the supabase-py SDK
-------------------------------------------------
The full ``supabase`` PyPI package drags in postgrest, gotrue, realtime,
and a websocket client just so we can call three storage REST endpoints.
Our auth is the service-role key (a static bearer) and our payloads are
binary PDFs — the abstraction layer adds nothing but install size and
release cadence. ``httpx`` is already a project dependency for the
qdrant client, so we reuse it.

What the service does
---------------------
- :meth:`upload` — PUT object to ``{bucket}/{key}``. Idempotent (uses
  ``x-upsert: true`` so re-running ingestion overwrites cleanly).
- :meth:`signed_url` — POST to ``/storage/v1/object/sign/{bucket}/{key}``
  and return a time-limited URL the browser can fetch directly.
- :meth:`delete` — DELETE for cleanup paths (failed-ingestion rollback,
  tenant document removal).

Graceful degradation
--------------------
If ``SUPABASE_URL`` or ``SUPABASE_SERVICE_KEY`` is unset the service
constructs but every method returns the no-op equivalent (``None`` /
``False``). Callers can call methods unconditionally without an
``if storage.enabled:`` dance — the only thing they MUST check is the
return value before persisting a key. This keeps storage optional in
dev environments without polluting business logic with feature flags.

Path conventions
----------------
Within the bucket:

    global/<doc_id>/<safe_filename>           — public-record law
    tenant/<org_id>/<doc_id>/<safe_filename>  — firm-private uploads

Putting scope at the top of the path makes it trivial to ``list``,
back up, or migrate one slice independently (e.g. "copy all global docs
to R2 for CDN"). The doc_id segment isolates per-document overwrites.
"""

from __future__ import annotations

import re
import uuid
from typing import Literal

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ``httpx`` calls in serving paths are short; long-running uploads of
# large PDFs need extra slack. 60s is generous for a single 50 MB PUT
# over a Railway → Supabase round trip.
_REQUEST_TIMEOUT_SECONDS = 60.0


class StorageService:
    """Talks to Supabase Storage via REST."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        service_key: str | None = None,
        bucket: str | None = None,
    ) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.supabase_url or "").rstrip("/")
        self._service_key = service_key or settings.supabase_service_key
        self._bucket = bucket or settings.supabase_storage_bucket
        self._enabled = bool(self._base_url and self._service_key)

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ---- public API --------------------------------------------------------

    def build_key(
        self,
        *,
        scope: Literal["global", "tenant"],
        document_id: uuid.UUID,
        filename: str,
        organization_id: uuid.UUID | None = None,
    ) -> str:
        """Construct the in-bucket object path for a document.

        ``filename`` is sanitised — Supabase Storage accepts most printable
        ASCII but rejects ``/`` and any non-UTF-8 byte. We collapse runs
        of unsafe characters into ``_`` and clamp length.
        """
        safe = _sanitise_filename(filename)
        if scope == "tenant":
            if organization_id is None:
                raise ValueError("tenant scope requires organization_id")
            return f"tenant/{organization_id}/{document_id}/{safe}"
        return f"global/{document_id}/{safe}"

    def upload(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str = "application/pdf",
    ) -> bool:
        """Upload ``data`` to ``{bucket}/{key}``.

        Returns True on success, False on any failure (logged at error).
        Idempotent — re-uploading the same key overwrites cleanly. We
        deliberately don't raise so ingestion can survive a transient
        storage outage with ``remote_storage_key`` left NULL; users can
        still query the doc, they just can't see the original PDF.
        """
        if not self._enabled:
            return False

        url = f"{self._base_url}/storage/v1/object/{self._bucket}/{key}"
        headers = {
            "Authorization": f"Bearer {self._service_key}",
            "Content-Type": content_type,
            # Without x-upsert the second call to the same key returns
            # 409; idempotent overwrites are the behavior we want from
            # this layer.
            "x-upsert": "true",
        }
        try:
            with httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                response = client.post(url, content=data, headers=headers)
        except httpx.HTTPError:
            logger.exception("storage.upload_network_failed", key=key)
            return False

        if response.status_code >= 400:
            logger.error(
                "storage.upload_failed",
                key=key,
                status=response.status_code,
                body=response.text[:500],
            )
            return False
        logger.info("storage.uploaded", key=key, bytes=len(data))
        return True

    def signed_url(self, key: str, *, expires_in_seconds: int) -> str | None:
        """Mint a short-lived URL the client can fetch directly.

        Supabase's response is ``{"signedURL": "/object/sign/..."}`` —
        we prepend the project base URL to give callers a full
        navigable URL. Returns None on any failure so the API can
        respond text-only without breaking.
        """
        if not self._enabled:
            return None
        url = f"{self._base_url}/storage/v1/object/sign/{self._bucket}/{key}"
        headers = {
            "Authorization": f"Bearer {self._service_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                response = client.post(
                    url,
                    headers=headers,
                    json={"expiresIn": int(expires_in_seconds)},
                )
        except httpx.HTTPError:
            logger.exception("storage.sign_network_failed", key=key)
            return None

        if response.status_code >= 400:
            logger.error(
                "storage.sign_failed",
                key=key,
                status=response.status_code,
                body=response.text[:500],
            )
            return None
        body = response.json()
        signed_path = body.get("signedURL") or body.get("signedUrl")
        if not signed_path:
            logger.error("storage.sign_missing_field", body=str(body)[:200])
            return None
        # Supabase returns a path beginning with ``/object/sign/...``;
        # convert to absolute. Newer projects sometimes return
        # ``/storage/v1/object/sign/...`` directly — handle both.
        if signed_path.startswith("/storage/v1"):
            return f"{self._base_url}{signed_path}"
        return f"{self._base_url}/storage/v1{signed_path}"

    def delete(self, key: str) -> bool:
        """Delete a single object. Returns False on failure (logged)."""
        if not self._enabled:
            return False
        url = f"{self._base_url}/storage/v1/object/{self._bucket}/{key}"
        headers = {"Authorization": f"Bearer {self._service_key}"}
        try:
            with httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                response = client.delete(url, headers=headers)
        except httpx.HTTPError:
            logger.exception("storage.delete_network_failed", key=key)
            return False
        if response.status_code >= 400:
            logger.error(
                "storage.delete_failed",
                key=key,
                status=response.status_code,
                body=response.text[:500],
            )
            return False
        return True


# ---- helpers ---------------------------------------------------------------


_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitise_filename(name: str) -> str:
    """Reduce a filename to a safe slug for Supabase Storage paths.

    Two goals: (1) eliminate path separators and shell metachars that
    would break the URL, (2) cap length so the total path stays under
    Supabase's 1024-char limit even for deeply-nested tenant paths.
    """
    base = name.rsplit("/", 1)[-1].strip() or "document.pdf"
    cleaned = _UNSAFE_FILENAME_RE.sub("_", base).strip("._-") or "document.pdf"
    # Most filesystems handle 255 bytes; the path also includes a UUID
    # prefix, so leave headroom.
    return cleaned[:160]
