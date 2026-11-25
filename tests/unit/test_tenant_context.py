"""TenantContext header parsing + role coercion."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import HTTPException

from app.api.dependencies.tenant import (
    get_optional_tenant_context,
    get_required_tenant_context,
    require_admin,
)
from app.core.tenant_context import TenantContext, parse_role, parse_uuid
from app.db.enums import TenantRole

# ---- pure parsing helpers --------------------------------------------------


def test_parse_uuid_handles_missing_and_invalid() -> None:
    assert parse_uuid(None) is None
    assert parse_uuid("") is None
    assert parse_uuid("   ") is None
    assert parse_uuid("not-a-uuid") is None

    real = uuid.uuid4()
    assert parse_uuid(str(real)) == real
    assert parse_uuid(f"  {real}  ") == real


def test_parse_role_collapses_unknown_to_viewer() -> None:
    assert parse_role(None) is TenantRole.VIEWER
    assert parse_role("") is TenantRole.VIEWER
    assert parse_role("supreme_overlord") is TenantRole.VIEWER
    assert parse_role("ADMIN") is TenantRole.ADMIN
    assert parse_role("  Owner ") is TenantRole.OWNER
    assert parse_role("lawyer") is TenantRole.LAWYER


# ---- TenantContext properties ---------------------------------------------


def test_anonymous_context_is_anonymous_and_not_admin() -> None:
    ctx = TenantContext.anonymous()
    assert ctx.is_anonymous
    assert not ctx.is_admin
    assert ctx.organization_id is None
    assert ctx.role is TenantRole.VIEWER


def test_admin_context_with_org_is_admin() -> None:
    ctx = TenantContext(
        organization_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        role=TenantRole.ADMIN,
    )
    assert not ctx.is_anonymous
    assert ctx.is_admin


# ---- FastAPI dependencies --------------------------------------------------


def test_optional_dep_anonymous_when_no_org_header() -> None:
    ctx = asyncio.run(
        get_optional_tenant_context(
            x_organization_id=None, x_user_id=None, x_role=None
        )
    )
    assert ctx.is_anonymous


def test_optional_dep_builds_context_when_headers_present() -> None:
    org = uuid.uuid4()
    user = uuid.uuid4()
    ctx = asyncio.run(
        get_optional_tenant_context(
            x_organization_id=str(org), x_user_id=str(user), x_role="admin"
        )
    )
    assert ctx.organization_id == org
    assert ctx.user_id == user
    assert ctx.role is TenantRole.ADMIN


def test_required_dep_raises_401_without_org() -> None:
    with pytest.raises(HTTPException) as ei:
        asyncio.run(
            get_required_tenant_context(
                x_organization_id=None, x_user_id=str(uuid.uuid4()), x_role="admin"
            )
        )
    assert ei.value.status_code == 401


def test_required_dep_raises_401_without_user() -> None:
    with pytest.raises(HTTPException) as ei:
        asyncio.run(
            get_required_tenant_context(
                x_organization_id=str(uuid.uuid4()), x_user_id=None, x_role="admin"
            )
        )
    assert ei.value.status_code == 401


def test_admin_dep_rejects_non_admin_role() -> None:
    with pytest.raises(HTTPException) as ei:
        asyncio.run(
            require_admin(
                x_organization_id=str(uuid.uuid4()),
                x_user_id=str(uuid.uuid4()),
                x_role="lawyer",
            )
        )
    assert ei.value.status_code == 403


def test_admin_dep_accepts_owner_or_admin() -> None:
    org = uuid.uuid4()
    user = uuid.uuid4()
    for role in ("admin", "owner"):
        ctx = asyncio.run(
            require_admin(
                x_organization_id=str(org), x_user_id=str(user), x_role=role
            )
        )
        assert ctx.is_admin
