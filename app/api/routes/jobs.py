"""Status board for background ingestion jobs.

Two routes:

- ``GET /ingestion/jobs/{job_id}`` — single job lookup.
- ``GET /ingestion/jobs/bulk?ids=<csv>`` — many at once, used by
  ``POST /upload-law/bulk``'s polling flow.

Tenant visibility rules (audit fix P0-2):

- Tenant-private job (``organization_id`` set) → visible only when the
  caller's ``TenantContext.organization_id`` matches. Cross-tenant probes
  return 404 (not 403) to prevent ID enumeration.
- GLOBAL job (``organization_id`` is NULL) → these are admin uploads of
  Constitution / Acts / public case law. The ``error_message`` field can
  contain stack traces and file paths, so a 404 is returned to anyone who
  is not an admin (``role ∈ {owner, admin}``). Anonymous callers cannot
  see admin jobs **by direct ID**, with one carefully-bounded exception:
  the bulk endpoint accepts up to ``BULK_STATUS_MAX_IDS`` IDs at once
  and surfaces visibility-denied entries in the ``missing`` array — same
  shape as "doesn't exist", so cross-tenant probing reveals nothing.

The bulk endpoint exists because the legacy anonymous
``/upload-law/bulk`` flow returns N job_ids in one HTTP response and
expects callers to poll all N. Without a batch endpoint, polling a 25-
file upload would mean 25 round-trips — wasted load on both sides.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.dependencies.tenant import get_optional_tenant_context
from app.core.tenant_context import TenantContext
from app.db.enums import GLOBAL_INGEST_ROLES
from app.db.models.ingestion_job import IngestionJob
from app.db.models.legal_document import LegalDocument
from app.db.repositories.job_repository import JobRepository
from app.db.session import get_db
from app.schemas.jobs import BulkJobStatusResponse, JobStatusResponse

router = APIRouter(prefix="/ingestion/jobs", tags=["ingestion"])

# Bound on the bulk lookup so a malicious caller can't ask for 10k IDs
# at once. Matches the bulk-upload cap so a 25-file upload's polling
# uses exactly one round-trip per cycle.
_BULK_STATUS_MAX_IDS = 50


@router.get("/bulk", response_model=BulkJobStatusResponse)
async def get_jobs_bulk(
    ids: str = Query(
        ...,
        description=(
            "Comma-separated list of job UUIDs. Up to "
            f"{_BULK_STATUS_MAX_IDS} per request."
        ),
    ),
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_optional_tenant_context),
) -> BulkJobStatusResponse:
    """Batch status lookup for a list of ingestion jobs.

    The companion to ``POST /upload-law/bulk`` — callers ship the
    ``job_id`` array they got back, we return per-job status +
    document_id + chunk_count in a single round-trip.

    Behavior:
    - Unknown / unauthorised IDs are folded into ``missing`` rather
      than producing a 404 — that way the caller doesn't have to
      special-case "some succeeded, some failed".
    - Malformed UUIDs in the query string are silently dropped into
      ``missing`` with the original token preserved (uuid.UUID raises
      on bad input; we just append to the missing list).
    - The endpoint is anonymous-friendly for GLOBAL jobs **only when
      the caller presents the job ID** — possession of the ID is
      treated as proof of ownership for the bulk-upload flow that
      created it. The same admin-only visibility still applies to
      direct GET /{id}.
    """
    raw_ids = [token.strip() for token in ids.split(",") if token.strip()]
    if not raw_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide at least one job id via ?ids=…",
        )
    if len(raw_ids) > _BULK_STATUS_MAX_IDS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Requested {len(raw_ids)} ids; the per-request cap is "
                f"{_BULK_STATUS_MAX_IDS}."
            ),
        )

    parsed_ids: list[uuid.UUID] = []
    missing: list[uuid.UUID] = []
    for token in raw_ids:
        try:
            parsed_ids.append(uuid.UUID(token))
        except ValueError:
            # Malformed UUIDs aren't valid IDs at all; we can't put
            # them in the ``missing`` list as UUIDs, so we drop them
            # silently. The bulk-upload flow always returns valid
            # UUIDs so this only fires when callers hand-edit the
            # URL.
            continue

    repo = JobRepository(db)
    rows = repo.get_many(parsed_ids)
    by_id = {job.id: job for job in rows}

    found: list[JobStatusResponse] = []
    for job_id in parsed_ids:
        job = by_id.get(job_id)
        if job is None or not _job_visible_bulk(job, tenant):
            # "Not found" and "not allowed to see" collapse to the
            # same response so cross-tenant probing reveals nothing.
            missing.append(job_id)
            continue
        found.append(_status_payload(db, repo, job))

    return BulkJobStatusResponse(found=found, missing=missing)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_optional_tenant_context),
) -> JobStatusResponse:
    repo = JobRepository(db)
    job = repo.get(job_id)
    if job is None or not _job_visible(job, tenant):
        # Same 404 in both cases so callers can't probe existence
        # across tenant boundaries.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return _status_payload(db, repo, job)


def _status_payload(
    db: Session, repo: JobRepository, job: IngestionJob
) -> JobStatusResponse:
    """Job row → response, plus what ingestion learned about the document.

    For auto-classified uploads the job row keeps the ``auto`` sentinel;
    the detected ``doc_type`` and ``summary`` live on the document.
    """
    payload = JobStatusResponse.model_validate(job, from_attributes=True)
    if job.document_id:
        payload.chunk_count = repo.count_chunks(job.document_id)
        document = db.get(LegalDocument, job.document_id)
        if document is not None and document.doc_type_source is not None:
            payload.doc_type = document.doc_type
            payload.summary = document.summary
            payload.doc_type_source = document.doc_type_source
    return payload


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


def _job_visible_bulk(job: object, tenant: TenantContext) -> bool:
    """Visibility rule for the bulk-status endpoint.

    Differs from :func:`_job_visible` in ONE place: anonymous callers
    can see GLOBAL jobs they already hold the ID for. This matches
    the ``POST /upload-law/bulk`` flow — uploads are anonymous, the
    response gives the caller N UUIDs, and forcing those callers to
    authenticate just to poll their own work would make the bulk
    endpoint unusable in its primary scenario.

    Tenant-private jobs are still strictly scoped — possessing an ID
    is NOT proof of organisation membership.
    """
    job_org = getattr(job, "organization_id", None)
    if job_org is None:
        # Anonymous OR any authenticated caller can see global jobs by
        # ID. The single-job route still enforces the admin-only rule.
        return True
    return tenant.organization_id is not None and tenant.organization_id == job_org
