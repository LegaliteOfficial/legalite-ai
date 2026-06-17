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


class BulkUploadResponse(BaseModel):
    """Aggregate response from ``POST /upload-law/bulk``.

    Contract:
    - ``accepted + rejected == len(files)`` (no file vanishes).
    - When ``accepted > 0`` the HTTP status is **202 Accepted** —
      even if some files were rejected.
    - When ``accepted == 0`` the HTTP status is **422
      Unprocessable Entity** — the request did nothing useful.
    - ``document_id`` and ``chunks_indexed`` are NOT in the per-job
      results because they don't exist yet — ingestion happens in
      the worker pool after the response returns. Clients should
      hit ``GET /ingestion/jobs/bulk?ids=<id1,id2,...>`` (or
      ``GET /ingestion/jobs/{id}`` for one at a time) to learn the
      ``document_id`` + ``chunk_count`` once each job lands.
    """

    accepted: int
    rejected: int
    max_files_per_request: int
    jobs: list[BulkUploadJobResult]
    rejections: list[BulkUploadRejection]
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
