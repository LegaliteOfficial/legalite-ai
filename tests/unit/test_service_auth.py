"""Item 8 — HMAC gateway signature verification."""

from __future__ import annotations

import time
import uuid

import pytest
from fastapi import HTTPException

from app.core.config import get_settings
from app.core.service_auth import compute_signature, verify_gateway_signature


@pytest.fixture(autouse=True)
def _enable_signature(monkeypatch):
    """Force the gateway secret on for these tests, regardless of env."""
    settings = get_settings()
    monkeypatch.setattr(settings, "legalite_gateway_secret", "test-shared-secret")
    monkeypatch.setattr(settings, "gateway_signature_max_skew_seconds", 60)
    yield


def _sign(org: uuid.UUID, user: uuid.UUID, role: str, ts: int | None = None) -> tuple[str, str]:
    """Helper — produce the (signature, timestamp) pair the backend would send."""
    ts = ts or int(time.time())
    sig = compute_signature(
        timestamp=ts,
        organization_id=str(org).lower(),
        user_id=str(user).lower(),
        role=role.lower(),
        secret="test-shared-secret",
    )
    return sig, str(ts)


def test_valid_signature_accepted() -> None:
    org, user = uuid.uuid4(), uuid.uuid4()
    sig, ts = _sign(org, user, "lawyer")
    # No exception → accepted.
    verify_gateway_signature(
        timestamp=ts,
        signature=sig,
        organization_id=str(org),
        user_id=str(user),
        role="lawyer",
    )


def test_missing_signature_when_secret_set_rejects_tenant_request() -> None:
    with pytest.raises(HTTPException) as ei:
        verify_gateway_signature(
            timestamp=None,
            signature=None,
            organization_id=str(uuid.uuid4()),
            user_id=str(uuid.uuid4()),
            role="lawyer",
        )
    assert ei.value.status_code == 401


def test_anonymous_request_does_not_need_signature() -> None:
    """Legacy /ask / /upload-law with no tenant headers must still work."""
    verify_gateway_signature(
        timestamp=None, signature=None,
        organization_id=None, user_id=None, role=None,
    )


def test_partial_credentials_rejected() -> None:
    """Sending a signature but no org_id is a sign of a bug or attack — refuse."""
    with pytest.raises(HTTPException):
        verify_gateway_signature(
            timestamp=str(int(time.time())),
            signature="deadbeef",
            organization_id=None,
            user_id=None,
            role=None,
        )


def test_stale_timestamp_rejected() -> None:
    org, user = uuid.uuid4(), uuid.uuid4()
    stale_ts = int(time.time()) - 10_000
    sig, ts = _sign(org, user, "lawyer", ts=stale_ts)
    with pytest.raises(HTTPException) as ei:
        verify_gateway_signature(
            timestamp=ts,
            signature=sig,
            organization_id=str(org),
            user_id=str(user),
            role="lawyer",
        )
    assert ei.value.status_code == 401
    assert "skew" in ei.value.detail.lower()


def test_tampered_signature_rejected() -> None:
    org, user = uuid.uuid4(), uuid.uuid4()
    _, ts = _sign(org, user, "lawyer")
    with pytest.raises(HTTPException):
        verify_gateway_signature(
            timestamp=ts,
            signature="0" * 64,  # well-formed hex but wrong
            organization_id=str(org),
            user_id=str(user),
            role="lawyer",
        )


def test_tampered_role_rejected() -> None:
    """Backend signs role=lawyer; attacker tries to escalate to admin."""
    org, user = uuid.uuid4(), uuid.uuid4()
    sig, ts = _sign(org, user, "lawyer")
    with pytest.raises(HTTPException):
        verify_gateway_signature(
            timestamp=ts,
            signature=sig,
            organization_id=str(org),
            user_id=str(user),
            role="admin",  # would-be escalation
        )


def test_secret_unset_bypasses_verification(monkeypatch) -> None:
    """Legacy network-trust mode: when no secret is configured, the function is a noop."""
    settings = get_settings()
    monkeypatch.setattr(settings, "legalite_gateway_secret", None)
    # Even invalid signatures should pass without raising.
    verify_gateway_signature(
        timestamp="not-a-timestamp",
        signature="invalid",
        organization_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        role="admin",
    )
