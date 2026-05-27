"""Item 6 — rate limiter key function picks org_id when present, IP otherwise."""

from __future__ import annotations

from types import SimpleNamespace

from app.core.rate_limit import tenant_key


def _request(headers: dict[str, str], client_host: str | None = "1.2.3.4"):
    """Minimal stand-in for a starlette Request used by slowapi."""
    client = SimpleNamespace(host=client_host) if client_host else None
    return SimpleNamespace(headers=headers, client=client)


def test_key_prefers_organization_id_header() -> None:
    req = _request({"x-organization-id": "ORG-ABC"})
    assert tenant_key(req) == "org:org-abc"  # lowercased + prefixed


def test_key_falls_back_to_ip_when_anonymous() -> None:
    req = _request({}, client_host="9.9.9.9")
    assert tenant_key(req) == "ip:9.9.9.9"


def test_key_handles_missing_client() -> None:
    """In some ASGI test transports request.client can be None."""
    req = _request({}, client_host=None)
    assert tenant_key(req) == "ip:unknown"


def test_key_distinguishes_orgs() -> None:
    a = tenant_key(_request({"x-organization-id": "ORG-A"}))
    b = tenant_key(_request({"x-organization-id": "ORG-B"}))
    assert a != b
    assert a.startswith("org:")
    assert b.startswith("org:")
