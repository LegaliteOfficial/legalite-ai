"""Choose between async (Arq) ingestion and the inline fallback.

Audit fix (P0-3): this function is now ``async`` and ``await``s the Arq
enqueue. The previous implementation called ``asyncio.run()`` from a sync
function invoked inside an async FastAPI route, which **raises**
``RuntimeError: asyncio.run() cannot be called from a running event loop``.
The broad ``except`` swallowed that error and every upload silently fell
back to inline ingestion — i.e. the Arq path never worked even when
Redis was up. The fix is straightforward: dispatcher → async; routes
``await`` the call.

When ``force_sync=True`` OR when ``arq`` isn't importable OR the enqueue
itself raises (Redis unreachable), we run the ingestion in-process and
mark the job complete synchronously. The response contract is identical
either way — only the ``status`` field tells the caller whether to poll.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.tenant_context import TenantContext
from app.db.enums import (
    ConfidentialityLevel,
    DocumentOrigin,
    VisibilityScope,
)
from app.db.repositories.job_repository import JobCreateRequest, JobRepository
from app.schemas.jobs import JobEnqueuedResponse
from app.services.ingestion_service import IngestionService

logger = get_logger(__name__)


def _persist_pdf(file_bytes: bytes, original_filename: str) -> tuple[uuid.UUID, Path]:
    """Write the uploaded PDF to the upload dir BEFORE enqueueing the job."""
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    handle_id = uuid.uuid4()
    safe_name = f"{handle_id}_{Path(original_filename).name}"
    storage_path = upload_dir / safe_name
    storage_path.write_bytes(file_bytes)
    return handle_id, storage_path


async def dispatch_upload(
    db: Session,
    *,
    file_bytes: bytes,
    original_filename: str,
    title: str,
    doc_type: str,
    visibility_scope: VisibilityScope,
    document_origin: DocumentOrigin | None,
    confidentiality_level: ConfidentialityLevel,
    tenant: TenantContext | None,
    force_sync: bool = False,
) -> JobEnqueuedResponse:
    """Create a job row, await Arq enqueue (or run sync), return the job handle."""
    _, storage_path = _persist_pdf(file_bytes, original_filename)
    job_repo = JobRepository(db)
    job = job_repo.create(
        JobCreateRequest(
            title=title,
            doc_type=doc_type,
            source_filename=original_filename,
            storage_path=str(storage_path),
            visibility_scope=visibility_scope.value,
            document_origin=document_origin.value if document_origin else None,
            organization_id=tenant.organization_id if tenant else None,
            user_id=tenant.user_id if tenant else None,
        )
    )
    db.commit()
    job_id = job.id

    if force_sync or not _arq_available():
        return _run_inline(
            db,
            job_id=job_id,
            file_bytes=file_bytes,
            original_filename=original_filename,
            title=title,
            doc_type=doc_type,
            visibility_scope=visibility_scope,
            document_origin=document_origin,
            confidentiality_level=confidentiality_level,
            tenant=tenant,
        )

    try:
        from app.jobs.queue import enqueue_ingestion

        await enqueue_ingestion(job_id, file_bytes)
    except Exception:
        logger.exception("dispatch.enqueue_failed", job_id=str(job_id))
        # Redis unreachable, queue misconfigured, etc. — fall back to inline.
        return _run_inline(
            db,
            job_id=job_id,
            file_bytes=file_bytes,
            original_filename=original_filename,
            title=title,
            doc_type=doc_type,
            visibility_scope=visibility_scope,
            document_origin=document_origin,
            confidentiality_level=confidentiality_level,
            tenant=tenant,
        )

    return JobEnqueuedResponse(job_id=job_id, status="queued")


def _arq_available() -> bool:
    """Cheap import probe — does NOT actually connect to Redis."""
    try:
        import arq  # noqa: F401
    except ImportError:
        return False
    return True


def _run_inline(
    db: Session,
    *,
    job_id: uuid.UUID,
    file_bytes: bytes,
    original_filename: str,
    title: str,
    doc_type: str,
    visibility_scope: VisibilityScope,
    document_origin: DocumentOrigin | None,
    confidentiality_level: ConfidentialityLevel,
    tenant: TenantContext | None,
) -> JobEnqueuedResponse:
    """Synchronous ingestion fallback — keeps single-deploy dev simple."""
    repo = JobRepository(db)
    try:
        repo.mark_processing(job_id)
        db.commit()
        document_id, _ = IngestionService(db).ingest_pdf(
            file_bytes=file_bytes,
            original_filename=original_filename,
            title=title,
            doc_type=doc_type,
            visibility_scope=visibility_scope,
            document_origin=document_origin,
            confidentiality_level=confidentiality_level,
            tenant=tenant,
        )
        repo.mark_completed(job_id, document_id=document_id)
        db.commit()
        return JobEnqueuedResponse(job_id=job_id, status="completed", document_id=document_id)
    except Exception as exc:
        db.rollback()
        repo.mark_failed(job_id, error=str(exc))
        db.commit()
        return JobEnqueuedResponse(job_id=job_id, status="failed")
