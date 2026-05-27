"""Helpers for enqueueing ingestion jobs onto the Arq Redis queue.

The web process talks to Redis only when enqueueing. The worker process
(``app.jobs.worker``) is what actually runs the job. Redis is the only
required infrastructure dependency between the two; we keep this layer
thin so swapping queues is a one-file change.
"""

from __future__ import annotations

import uuid

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.telemetry import ingestion_job_state

logger = get_logger(__name__)


async def enqueue_ingestion(job_id: uuid.UUID, file_bytes: bytes) -> None:
    """Enqueue a single ingestion job onto the Arq queue.

    The full file payload travels through Redis. For very large PDFs the
    storage_path on disk is the authoritative copy — the bytes argument is
    only for tests / in-process pools that don't share the upload volume.
    """
    settings = get_settings()
    try:
        from arq import create_pool
        from arq.connections import RedisSettings
    except ImportError:
        # Arq optional — degrade gracefully so unit tests that don't have
        # Redis still work. The web process won't try to enqueue unless
        # the worker is wired up (config check below).
        logger.error("jobs.enqueue.arq_missing")
        raise

    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    try:
        await redis.enqueue_job(
            "ingest_pdf_task",
            str(job_id),
            file_bytes,
            _queue_name=settings.ingestion_queue_name,
        )
        ingestion_job_state.labels(state="queued").inc()
        logger.info("jobs.enqueued", job_id=str(job_id), bytes=len(file_bytes))
    finally:
        await redis.close()
