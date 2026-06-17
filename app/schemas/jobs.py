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
    - Clients should poll ``GET /jobs/{job_id}`` for each accepted
      ``job_id`` to track ingestion progress.
    """

    accepted: int
    rejected: int
    max_files_per_request: int
    jobs: list[BulkUploadJobResult]
    rejections: list[BulkUploadRejection]
