"""GET /ingestion/jobs/{id} — status board for background ingestion.

Tenant visibility rules (audit fix P0-2):

- Tenant-private job (``organization_id`` set) → visible only when the
  caller's ``TenantContext.organization_id`` matches. Cross-tenant probes
  return 404 (not 403) to prevent ID enumeration.
- GLOBAL job (``organization_id`` is NULL) → these are admin uploads of
  Constitution / Acts / public case law. The ``error_message`` field can
  contain stack traces and file paths, so a 404 is returned to anyone who
  is not an admin (``role ∈ {owner, admin}``). Anonymous callers cannot
  see admin jobs.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies.tenant import get_optional_tenant_context
from app.core.tenant_context import TenantContext
from app.db.enums import GLOBAL_INGEST_ROLES
from app.db.repositories.job_repository import JobRepository
from app.db.session import get_db
from app.schemas.jobs import JobStatusResponse

router = APIRouter(prefix="/ingestion/jobs", tags=["ingestion"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_optional_tenant_context),
) -> JobStatusResponse:
    job = JobRepository(db).get(job_id)
    if job is None or not _job_visible(job, tenant):
        # Same 404 in both cases so callers can't probe existence
        # across tenant boundaries.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JobStatusResponse.model_validate(job, from_attributes=True)


def _job_visible(job: object, tenant: TenantContext) -> bool:
    """Return True iff the caller is allowed to see this job row.

    Rules:
    - Job with ``organization_id`` set → caller's org must match exactly.
    - Job with ``organization_id`` NULL (admin / global upload) → caller
      must have an admin role. Anonymous callers and non-admin tenants
      cannot see these — they could leak stack traces or paths from
      ``error_message``.
    """
    job_org = getattr(job, "organization_id", None)
    if job_org is None:
        return tenant.role in GLOBAL_INGEST_ROLES
    return tenant.organization_id is not None and tenant.organization_id == job_org
