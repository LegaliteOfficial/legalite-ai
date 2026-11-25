"""P0-2 — /ingestion/jobs/{id} visibility predicate."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.api.routes.jobs import _job_visible
from app.core.tenant_context import TenantContext
from app.db.enums import TenantRole


@dataclass
class _Job:
    organization_id: uuid.UUID | None


def _ctx(role: TenantRole = TenantRole.LAWYER, org: uuid.UUID | None = None) -> TenantContext:
    return TenantContext(organization_id=org, user_id=uuid.uuid4() if org else None, role=role)


def test_global_job_hidden_from_anonymous() -> None:
    """Audit-fix: admin-uploaded global jobs leaked error_message + paths to anyone."""
    assert _job_visible(_Job(organization_id=None), TenantContext.anonymous()) is False


def test_global_job_hidden_from_non_admin_tenant() -> None:
    assert (
        _job_visible(_Job(organization_id=None), _ctx(role=TenantRole.LAWYER, org=uuid.uuid4()))
        is False
    )


def test_global_job_visible_to_admin() -> None:
    assert (
        _job_visible(_Job(organization_id=None), _ctx(role=TenantRole.ADMIN, org=uuid.uuid4()))
        is True
    )


def test_global_job_visible_to_owner() -> None:
    assert (
        _job_visible(_Job(organization_id=None), _ctx(role=TenantRole.OWNER, org=uuid.uuid4()))
        is True
    )


def test_tenant_job_visible_to_same_org() -> None:
    org = uuid.uuid4()
    assert _job_visible(_Job(organization_id=org), _ctx(org=org)) is True


def test_tenant_job_hidden_from_other_org() -> None:
    assert _job_visible(_Job(organization_id=uuid.uuid4()), _ctx(org=uuid.uuid4())) is False


def test_tenant_job_hidden_from_anonymous() -> None:
    assert _job_visible(_Job(organization_id=uuid.uuid4()), TenantContext.anonymous()) is False
