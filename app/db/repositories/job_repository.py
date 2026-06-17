"""Repository for background ingestion jobs (Phase 5 — M4).

Pure persistence. Worker writes lifecycle transitions through these helpers,
the API route reads via :meth:`get_job`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.enums import VisibilityScope
from app.db.models.ingestion_job import IngestionJob
from app.db.models.legal_chunk import LegalChunk


@dataclass(slots=True)
class JobCreateRequest:
    title: str
    doc_type: str
    source_filename: str
    storage_path: str
    visibility_scope: str = VisibilityScope.GLOBAL.value
    document_origin: str | None = None
    organization_id: uuid.UUID | None = None
    user_id: uuid.UUID | None = None


class JobRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, request: JobCreateRequest) -> IngestionJob:
        job = IngestionJob(
            organization_id=request.organization_id,
            user_id=request.user_id,
            visibility_scope=request.visibility_scope,
            document_origin=request.document_origin,
            title=request.title,
            doc_type=request.doc_type,
            source_filename=request.source_filename,
            storage_path=request.storage_path,
            status="queued",
            progress=0,
            enqueued_at=datetime.now(timezone.utc),
        )
        self._db.add(job)
        self._db.flush()
        return job

    def get(self, job_id: uuid.UUID) -> IngestionJob | None:
        return self._db.get(IngestionJob, job_id)

    def get_many(self, job_ids: list[uuid.UUID]) -> list[IngestionJob]:
        """Fetch many jobs in one round-trip.

        Used by the bulk-status endpoint so a 25-file batch's polling
        costs one SQL query instead of 25. Returns rows in arbitrary
        order; callers should re-sort if they need to mirror the input
        sequence.
        """
        if not job_ids:
            return []
        rows = self._db.execute(
            select(IngestionJob).where(IngestionJob.id.in_(job_ids))
        ).scalars()
        return list(rows)

    def count_chunks(self, document_id: uuid.UUID) -> int:
        """How many indexed chunks the document landed.

        Surfaces in the JobStatusResponse alongside the document_id so
        the bulk-upload caller can verify ingestion produced text —
        not just that it didn't crash. Cheap: single COUNT on the
        ``legal_chunks`` index.
        """
        count = self._db.execute(
            select(func.count())
            .select_from(LegalChunk)
            .where(LegalChunk.document_id == document_id)
        ).scalar()
        return int(count or 0)

    def mark_processing(self, job_id: uuid.UUID) -> None:
        job = self._db.get(IngestionJob, job_id)
        if job is None:
            return
        job.status = "processing"
        job.progress = max(job.progress, 5)
        job.started_at = datetime.now(timezone.utc)
        self._db.flush()

    def mark_completed(
        self, job_id: uuid.UUID, *, document_id: uuid.UUID
    ) -> None:
        job = self._db.get(IngestionJob, job_id)
        if job is None:
            return
        job.status = "completed"
        job.progress = 100
        job.document_id = document_id
        job.finished_at = datetime.now(timezone.utc)
        self._db.flush()

    def mark_failed(self, job_id: uuid.UUID, *, error: str) -> None:
        job = self._db.get(IngestionJob, job_id)
        if job is None:
            return
        job.status = "failed"
        job.error_message = error[:2000]
        job.finished_at = datetime.now(timezone.utc)
        self._db.flush()

    def bump_progress(self, job_id: uuid.UUID, *, progress: int) -> None:
        job = self._db.get(IngestionJob, job_id)
        if job is None:
            return
        job.progress = max(job.progress, min(100, progress))
        self._db.flush()

    def bump_retry(self, job_id: uuid.UUID) -> int:
        job = self._db.get(IngestionJob, job_id)
        if job is None:
            return 0
        job.retry_count += 1
        self._db.flush()
        return job.retry_count
