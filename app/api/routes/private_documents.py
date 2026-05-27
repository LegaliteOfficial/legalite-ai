"""Tenant-private document upload.

Phase 5 M4 made this endpoint async: it now returns a ``job_id`` while
the heavy ingestion (parse → chunk → embed → Qdrant) runs in the Arq
worker. The Redis-unavailable fallback keeps single-binary deploys working.

Refuses with 401 if the gateway headers are missing, 422 if the
``document_origin`` is not in the tenant-private allowlist.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies.tenant import get_required_tenant_context
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.core.tenant_context import TenantContext
from app.db.enums import (
    TENANT_PRIVATE_ORIGINS,
    ConfidentialityLevel,
    DocumentOrigin,
    VisibilityScope,
)
from app.db.session import get_db
from app.schemas.jobs import JobEnqueuedResponse
from app.services.upload_dispatcher import dispatch_upload

router = APIRouter(prefix="/documents", tags=["documents"])

_UPLOAD_LIMIT = get_settings().rate_limit_upload


@router.post(
    "/private-upload",
    response_model=JobEnqueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
@limiter.limit(_UPLOAD_LIMIT)
async def private_upload(
    request: Request,
    file: UploadFile = File(..., description="Firm-private PDF (memo, sample case, template, ...)"),
    title: str = Form(..., min_length=2, max_length=512),
    doc_type: str = Form(..., description="Free-form label for the firm's filing system"),
    document_origin: DocumentOrigin = Form(
        ...,
        description=(
            "Must be one of: sample_case | internal_memo | strategy_document | "
            "template | legal_opinion | private_reference."
        ),
    ),
    confidentiality_level: ConfidentialityLevel = Form(
        default=ConfidentialityLevel.INTERNAL,
        description="public | internal | confidential | privileged",
    ),
    tenant: TenantContext = Depends(get_required_tenant_context),
    db: Session = Depends(get_db),
) -> JobEnqueuedResponse:
    if document_origin not in TENANT_PRIVATE_ORIGINS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"document_origin={document_origin.value!r} is not a tenant-private "
                "type. Use one of: " + ", ".join(o.value for o in TENANT_PRIVATE_ORIGINS)
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
            visibility_scope=VisibilityScope.TENANT_PRIVATE,
            document_origin=document_origin,
            confidentiality_level=confidentiality_level,
            tenant=tenant,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
