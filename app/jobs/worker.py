"""Arq worker for background PDF ingestion (Phase 5 — M4).

Run with::

    arq app.jobs.worker.WorkerSettings

The worker pulls ``ingest_pdf_task`` jobs off the configured Redis queue,
materializes a fresh database session per task, and delegates the actual
work to :class:`IngestionService`. The job row in ``ingestion_jobs`` is
updated through every state transition so the API can surface progress
without touching Redis directly.

Idempotency contract
--------------------
Each task receives a ``job_id`` (UUID string). The worker reads the
matching ``IngestionJob`` row to extract title / scope / origin / etc.
If the job's status is already ``completed`` it short-circuits — re-runs
caused by Arq retries do not double-ingest. If the document was partially
ingested (status = ``processing`` but no document_id), the worker rolls
the row back to ``queued`` and retries.

Retry policy
------------
On any unhandled exception:

  * The job's ``retry_count`` is bumped.
  * If retry_count <= ``INGESTION_MAX_RETRIES`` the worker raises
    ``arq.worker.Retry`` so Arq re-queues the task with a backoff. Arq
    only retries on ``Retry`` — re-raising the original exception makes
    Arq record the task as failed and never run it again, which used to
    leave the job row stuck at ``processing`` with no error message.
  * Otherwise the row is marked ``failed`` with the error message and
    the worker swallows the exception (the task is "done" from Arq's POV).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from arq.connections import RedisSettings
from arq.worker import Retry

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.telemetry import ingestion_job_state
from app.core.tenant_context import TenantContext
from app.db.enums import (
    ConfidentialityLevel,
    DocumentOrigin,
    TenantRole,
    VisibilityScope,
)
from app.db.repositories.job_repository import JobRepository
from app.db.session import SessionLocal
from app.services.ingestion_service import IngestionService

logger = get_logger(__name__)


async def startup(_ctx: dict[str, Any]) -> None:
    configure_logging()
    logger.info("worker.startup")


async def shutdown(_ctx: dict[str, Any]) -> None:
    logger.info("worker.shutdown")


async def ingest_pdf_task(
    _ctx: dict[str, Any],
    job_id_str: str,
    file_bytes: bytes,
) -> dict[str, Any]:
    """Run one ingestion. Returns the resulting document_id on success.

    All DB work happens inside a fresh session that we open per task —
    Arq workers are async + multi-task, sharing a Session would be unsafe.
    """
    settings = get_settings()
    job_id = uuid.UUID(job_id_str)

    db = SessionLocal()
    repo = JobRepository(db)

    try:
        job = repo.get(job_id)
        if job is None:
            logger.error("worker.job_missing", job_id=job_id_str)
            return {"status": "missing"}

        # Idempotency — never re-ingest a completed job.
        if job.status == "completed":
            logger.info("worker.skip_completed", job_id=job_id_str)
            return {
                "status": "already_completed",
                "document_id": str(job.document_id) if job.document_id else None,
            }

        repo.mark_processing(job_id)
        ingestion_job_state.labels(state="processing").inc()
        db.commit()
    except Exception:
        db.rollback()
        db.close()
        raise

    # Real work happens on a SEPARATE session so the status-update writes
    # above are durable even if ingestion itself fails inside the service.
    work_db = SessionLocal()
    try:
        scope = VisibilityScope(job.visibility_scope)
        origin: DocumentOrigin | None = None
        if job.document_origin:
            try:
                origin = DocumentOrigin(job.document_origin)
            except ValueError:
                origin = None

        tenant: TenantContext | None = None
        if job.organization_id is not None and job.user_id is not None:
            tenant = TenantContext(
                organization_id=job.organization_id,
                user_id=job.user_id,
                role=TenantRole.LAWYER,
            )

        service = IngestionService(work_db)
        document_id, _chunks = service.ingest_pdf(
            file_bytes=file_bytes if file_bytes else Path(job.storage_path).read_bytes(),
            original_filename=job.source_filename,
            title=job.title,
            doc_type=job.doc_type,
            visibility_scope=scope,
            document_origin=origin,
            confidentiality_level=ConfidentialityLevel.PUBLIC
            if scope is VisibilityScope.GLOBAL
            else ConfidentialityLevel.INTERNAL,
            tenant=tenant,
        )

        # Reopen the job session for the final transition.
        finalize_db = SessionLocal()
        try:
            JobRepository(finalize_db).mark_completed(job_id, document_id=document_id)
            finalize_db.commit()
            ingestion_job_state.labels(state="completed").inc()
        finally:
            finalize_db.close()
        return {"status": "completed", "document_id": str(document_id)}

    except Exception as exc:  # noqa: BLE001
        # Bump retry count + mark failed if we're out of attempts.
        fail_db = SessionLocal()
        try:
            fail_repo = JobRepository(fail_db)
            attempts = fail_repo.bump_retry(job_id)
            fail_db.commit()
            if attempts > settings.ingestion_max_retries:
                fail_repo.mark_failed(job_id, error=str(exc))
                fail_db.commit()
                ingestion_job_state.labels(state="failed").inc()
                logger.exception("worker.giving_up", job_id=job_id_str, retries=attempts)
                return {"status": "failed", "error": str(exc)[:500]}
            ingestion_job_state.labels(state="retried").inc()
        finally:
            fail_db.close()
        logger.exception("worker.retry_scheduled", job_id=job_id_str, attempt=attempts)
        raise Retry(defer=_RETRY_BACKOFF_SECONDS * attempts) from exc
    finally:
        work_db.close()


# Linear backoff between attempts: 10s, 20s, 30s, ...
_RETRY_BACKOFF_SECONDS = 10

_worker_settings = get_settings()


class WorkerSettings:
    """Arq worker configuration. Resolved at module load.

    Arq reads ``queue_name``, ``redis_settings``, and ``max_tries`` as
    plain class attributes (it does string concatenation on
    ``queue_name`` to derive the health-check key). Don't turn these into
    classmethods — Arq won't call them, it will try to use the
    classmethod object as a string and crash on startup.
    """

    functions = [ingest_pdf_task]
    on_startup = startup
    on_shutdown = shutdown
    queue_name = _worker_settings.ingestion_queue_name
    # arq retries internally; our job row handles "giving up" semantics.
    max_tries = _worker_settings.ingestion_max_retries + 1
    redis_settings = RedisSettings.from_dsn(_worker_settings.redis_url)
