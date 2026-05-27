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
