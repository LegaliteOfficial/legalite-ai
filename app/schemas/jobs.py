"""Pydantic schemas for the ingestion-job API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


JobStatus = Literal["queued", "processing", "completed", "failed"]


class JobEnqueuedResponse(BaseModel):
    """Returned by /admin/upload-law and /documents/private-upload."""

    job_id: uuid.UUID
    status: JobStatus = "queued"
    document_id: uuid.UUID | None = None


class JobStatusResponse(BaseModel):
    id: uuid.UUID
    status: JobStatus
    progress: int
    document_id: uuid.UUID | None = None
    # Populated once ``status == "completed"``. Mirrors the legacy
    # ``/upload-law`` response's ``chunks_indexed`` so polling clients
    # can verify a doc actually produced text — not just that ingestion
    # didn't crash. 0 means the document row exists but no chunks
    # landed (empty PDF, OCR failure, etc.).
    chunk_count: int = 0
    visibility_scope: str
    document_origin: str | None = None
    title: str
    doc_type: str
    source_filename: str
    organization_id: uuid.UUID | None = None
    user_id: uuid.UUID | None = None
    error_message: str | None = None
    retry_count: int
    enqueued_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


# ─── Bulk upload ────────────────────────────────────────────────────────────


class BulkUploadJobResult(BaseModel):
    """One row in the ``jobs`` array — a file the server accepted.

    The ``filename`` echoes back the upload's ``UploadFile.filename``
    so the client can correlate the result with the original input.
    ``title`` is what the server derived from that filename (strip
    extension + sanitise). ``status`` is the job state immediately
    after enqueue — typically ``queued``; ``completed`` only when
    the inline-fallback path ran because the queue was unreachable.
    """

    filename: str
    title: str
    job_id: uuid.UUID
    status: JobStatus
    document_id: uuid.UUID | None = None


class BulkUploadRejection(BaseModel):
    """One row in the ``rejections`` array — a file the server refused.

    The whole batch is NOT failed when one file is rejected; the rest
    still go to the queue. This row tells the client which file was
    dropped and why so they can fix and re-upload that file alone.
    """

    filename: str
    reason: str


class BulkUploadSkipped(BaseModel):
    """One row in the ``skipped_duplicates`` array.

    A skipped file is NOT a rejection — it just means the server
    detected that the document already exists (or is being ingested)
    and the upload was redundant. We surface the existing
    ``document_id`` (or ``job_id`` for an in-flight match) so the
    client can immediately deep-link to the existing artefact without
    re-uploading.

    Skip reasons:

    - ``already_ingested`` — a completed ``legal_documents`` row exists
      with the same ``(title, doc_type)`` at GLOBAL scope. The
      ``existing_document_id`` field is set.
    - ``in_flight`` — an ``ingestion_jobs`` row in ``queued`` or
      ``processing`` status exists with the same ``(title, doc_type)``.
      The ``existing_job_id`` field is set and clients should poll
      that ID rather than the new one (which we never create).
    - ``duplicate_in_batch`` — two files in the SAME request resolve
      to the same title. The first one through wins; later ones are
      surfaced here with no existing references.
    """

    filename: str
    title: str
    reason: Literal["already_ingested", "in_flight", "duplicate_in_batch"]
    existing_document_id: uuid.UUID | None = None
    existing_job_id: uuid.UUID | None = None


class BulkUploadResponse(BaseModel):
    """Aggregate response from ``POST /upload-law/bulk``.

    Contract:
    - ``accepted + rejected + skipped == len(files)`` (no file vanishes).
    - When ``accepted > 0`` the HTTP status is **202 Accepted** —
      even if some files were rejected or skipped.
    - When ``accepted == 0`` AND ``skipped == 0`` the HTTP status is
      **422 Unprocessable Entity** — every file was a hard rejection.
    - When ``accepted == 0`` AND ``skipped > 0`` the HTTP status is
      **200 OK** — nothing new was ingested but the duplicates
      resolve to documents/jobs the caller can already see.
    - ``document_id`` and ``chunks_indexed`` are NOT in the per-job
      results because they don't exist yet — ingestion happens in
      the worker pool after the response returns. Clients should
      hit ``GET /ingestion/jobs/bulk?ids=<id1,id2,...>`` (or
      ``GET /ingestion/jobs/{id}`` for one at a time) to learn the
      ``document_id`` + ``chunk_count`` once each job lands.
    """

    accepted: int
    rejected: int
    skipped: int = 0
    max_files_per_request: int
    jobs: list[BulkUploadJobResult]
    rejections: list[BulkUploadRejection]
    skipped_duplicates: list[BulkUploadSkipped] = []
    # Convenience for clients: the URL pattern to poll. Embedded in
    # the response so callers don't have to hard-code the path; if
    # we move the polling endpoint later, the response will redirect
    # them transparently.
    poll_endpoint: str = "/ingestion/jobs/bulk?ids=<comma-separated>"


class BulkJobStatusResponse(BaseModel):
    """Aggregate response from ``GET /ingestion/jobs/bulk?ids=...``.

    ``found`` is one entry per matched ID; ``missing`` carries IDs the
    caller asked for but that don't exist (or that the caller can't
    see). We deliberately conflate "not found" and "not authorised"
    so cross-tenant probing returns no information.
    """

    found: list[JobStatusResponse]
    missing: list[uuid.UUID]
