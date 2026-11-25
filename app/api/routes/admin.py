"""Admin-only routes for the GLOBAL Ghana legal corpus.

``POST /admin/upload-law`` ingests Constitution / Acts / Regulations /
public case law. Phase 5 M4 made this endpoint async — it now returns a
``job_id`` immediately and the actual ingestion runs in the Arq worker.
When Redis / Arq is unavailable the dispatcher transparently falls back
to inline ingestion so dev / single-binary deploys still work.

The legacy ``/upload-law`` endpoint (in documents.py) is unchanged.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies.tenant import require_admin
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.core.tenant_context import TenantContext
from app.db.enums import (
    GLOBAL_ONLY_ORIGINS,
    ConfidentialityLevel,
    DocumentOrigin,
    VisibilityScope,
)
from app.db.session import get_db
from app.schemas.jobs import JobEnqueuedResponse
from app.services.upload_dispatcher import dispatch_upload

router = APIRouter(prefix="/admin", tags=["admin"])

_UPLOAD_LIMIT = get_settings().rate_limit_upload


@router.post(
    "/upload-law",
    response_model=JobEnqueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
@limiter.limit(_UPLOAD_LIMIT)
async def admin_upload_law(
    request: Request,
    file: UploadFile = File(..., description="Ghana legal PDF (Constitution, Act, Regulation, Case)"),
    title: str = Form(..., min_length=2, max_length=512),
    doc_type: str = Form(..., description="Constitution | Act | Regulation | Case Law | ..."),
    document_origin: DocumentOrigin | None = Form(
        default=None,
        description=(
            "Optional explicit classification. Must be a global origin: "
            "constitution | act | regulation | case_law | precedent | judgement."
        ),
    ),
    admin_ctx: TenantContext = Depends(require_admin),
    db: Session = Depends(get_db),
) -> JobEnqueuedResponse:
    """Enqueue a global / authoritative legal document for ingestion."""
    if document_origin is not None and document_origin not in GLOBAL_ONLY_ORIGINS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"document_origin={document_origin.value!r} is tenant-private. "
                "GLOBAL uploads must use a global-only origin."
            ),
        )

    settings = get_settings()
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_mb}MB limit",
        )

    try:
        return await dispatch_upload(
            db,
            file_bytes=content,
            original_filename=file.filename,
            title=title,
            doc_type=doc_type,
            visibility_scope=VisibilityScope.GLOBAL,
            document_origin=document_origin,
            confidentiality_level=ConfidentialityLevel.PUBLIC,
            tenant=admin_ctx,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
