"""StorageService — graceful degradation, key construction, sanitisation.

We don't hit Supabase in unit tests. Those interactions are exercised
in integration tests against a real bucket; here we lock down the
deterministic surface: how the service behaves with no credentials,
how it builds in-bucket paths, and how the filename sanitiser handles
adversarial input.
"""

from __future__ import annotations

import uuid

import pytest

from app.services.storage_service import StorageService, _sanitise_filename


def test_disabled_when_credentials_missing() -> None:
    s = StorageService(base_url=None, service_key=None, bucket="legal-documents")
    assert s.enabled is False
    assert s.upload(key="anything.pdf", data=b"x") is False
    assert s.signed_url("anything.pdf", expires_in_seconds=60) is None
    assert s.delete("anything.pdf") is False


def test_enabled_when_both_credentials_present() -> None:
    s = StorageService(
        base_url="https://example.supabase.co",
        service_key="svc-role-key",
        bucket="legal-documents",
    )
    assert s.enabled is True


def test_build_key_global_scope() -> None:
    s = StorageService(base_url="https://x.supabase.co", service_key="k", bucket="b")
    doc_id = uuid.UUID("c49af58e-008f-41bd-9acb-118a5ca5e670")
    key = s.build_key(scope="global", document_id=doc_id, filename="Constitution.pdf")
    assert key == f"global/{doc_id}/Constitution.pdf"


def test_build_key_tenant_scope_requires_org() -> None:
    s = StorageService(base_url="https://x.supabase.co", service_key="k", bucket="b")
    doc_id = uuid.uuid4()
    with pytest.raises(ValueError):
        s.build_key(scope="tenant", document_id=doc_id, filename="memo.pdf")


def test_build_key_tenant_scope_includes_org() -> None:
    s = StorageService(base_url="https://x.supabase.co", service_key="k", bucket="b")
    doc_id = uuid.UUID("c49af58e-008f-41bd-9acb-118a5ca5e670")
    org_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    key = s.build_key(
        scope="tenant",
        document_id=doc_id,
        filename="firm-memo.pdf",
        organization_id=org_id,
    )
    assert key == f"tenant/{org_id}/{doc_id}/firm-memo.pdf"


def test_sanitise_strips_path_separators() -> None:
    # Adversarial: a user-supplied filename like ``../etc/passwd`` would
    # let an attacker escape the document folder if we naively
    # concatenated. The sanitiser must collapse to a leaf name.
    assert _sanitise_filename("../../etc/passwd") == "passwd"
    assert _sanitise_filename("foo/bar/baz.pdf") == "baz.pdf"


def test_sanitise_replaces_unsafe_chars() -> None:
    assert _sanitise_filename("Acquah v. Kwesi (2024).pdf") == "Acquah_v._Kwesi_2024_.pdf"


def test_sanitise_handles_empty() -> None:
    assert _sanitise_filename("") == "document.pdf"
    # Pure separators are also collapsed to the default.
    assert _sanitise_filename("///") == "document.pdf"


def test_sanitise_clamps_length() -> None:
    long = "x" * 500 + ".pdf"
    assert len(_sanitise_filename(long)) <= 160
