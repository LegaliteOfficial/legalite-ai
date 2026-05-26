"""IngestionService boundary checks — the cheap rejections at the service
edge that prevent obvious tenancy errors from reaching the database."""

from __future__ import annotations

import uuid

import pytest

from app.core.tenant_context import TenantContext
from app.db.enums import DocumentOrigin, TenantRole, VisibilityScope
from app.services.ingestion_service import _resolve_org_id, _validate_origin_scope


def test_validate_rejects_private_origin_in_global_scope() -> None:
    with pytest.raises(ValueError, match="tenant-private"):
        _validate_origin_scope(VisibilityScope.GLOBAL, DocumentOrigin.SAMPLE_CASE)


def test_validate_rejects_global_origin_in_tenant_scope() -> None:
    with pytest.raises(ValueError, match="global-only"):
        _validate_origin_scope(VisibilityScope.TENANT_PRIVATE, DocumentOrigin.CONSTITUTION)


def test_validate_allows_matching_combos() -> None:
    # No exception:
    _validate_origin_scope(VisibilityScope.GLOBAL, DocumentOrigin.ACT)
    _validate_origin_scope(VisibilityScope.GLOBAL, DocumentOrigin.CASE_LAW)
    _validate_origin_scope(VisibilityScope.TENANT_PRIVATE, DocumentOrigin.INTERNAL_MEMO)
    _validate_origin_scope(VisibilityScope.TENANT_PRIVATE, DocumentOrigin.TEMPLATE)
    # No origin = no check applied.
    _validate_origin_scope(VisibilityScope.GLOBAL, None)
    _validate_origin_scope(VisibilityScope.TENANT_PRIVATE, None)


def test_resolve_org_global_ignores_tenant_header() -> None:
    """An admin uploading to GLOBAL must NOT have their org tag the document."""
    ctx = TenantContext(
        organization_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        role=TenantRole.ADMIN,
    )
    assert _resolve_org_id(VisibilityScope.GLOBAL, ctx) is None


def test_resolve_org_private_requires_tenant_context() -> None:
    with pytest.raises(ValueError, match="organization_id"):
        _resolve_org_id(VisibilityScope.TENANT_PRIVATE, None)

    anon = TenantContext.anonymous()
    with pytest.raises(ValueError, match="organization_id"):
        _resolve_org_id(VisibilityScope.TENANT_PRIVATE, anon)


def test_resolve_org_private_returns_tenant_org() -> None:
    org = uuid.uuid4()
    ctx = TenantContext(
        organization_id=org,
        user_id=uuid.uuid4(),
        role=TenantRole.LAWYER,
    )
    assert _resolve_org_id(VisibilityScope.TENANT_PRIVATE, ctx) == org
