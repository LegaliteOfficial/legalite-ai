"""Tenant-private document upload.

``POST /documents/private-upload`` is the door for law firms to upload
sample cases, internal memos, strategy documents, and other materials that
must improve answers ONLY for that firm — never globally, never visible to
other tenants.

Required gateway headers (set by legalite-backend):
- ``x-organization-id`` — the firm's UUID
- ``x-user-id``        — who initiated the upload
- ``x-role``           — at least ``viewer``

Refuses with 401 if no org header, 422 if origin/scope are mismatched.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies.services import get_ingestion_service
from app.api.dependencies.tenant import get_required_tenant_context
from app.core.config import get_settings
from app.core.tenant_context import TenantContext
from app.db.enums import (
    TENANT_PRIVATE_ORIGINS,
    ConfidentialityLevel,
    DocumentOrigin,
    VisibilityScope,
)
from app.schemas.documents import UploadLawResponse
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/private-upload",
    response_model=UploadLawResponse,
    status_code=status.HTTP_201_CREATED,
)
async def private_upload(
    file: UploadFile = File(..., description="Firm-private PDF (memo, sample case, template, ...)"),
    title: str = Form(..., min_length=2, max_length=512),
    doc_type: str = Form(..., description="Free-form label for the firm's filing system"),
    document_origin: DocumentOrigin = Form(
        ...,
        description=(
            "What kind of private document this is. Must be one of: "
            "sample_case | internal_memo | strategy_document | template | "
            "legal_opinion | private_reference."
        ),
    ),
    confidentiality_level: ConfidentialityLevel = Form(
        default=ConfidentialityLevel.INTERNAL,
        description="public | internal | confidential | privileged",
    ),
    tenant: TenantContext = Depends(get_required_tenant_context),
    service: IngestionService = Depends(get_ingestion_service),
) -> UploadLawResponse:
    """Ingest a firm-private document. Retrieval is scoped to this org only."""
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
        document_id, chunks_indexed = service.ingest_pdf(
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
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    return UploadLawResponse(
        document_id=document_id,
        title=title,
        doc_type=doc_type,
        page_count=0,
        chunks_indexed=chunks_indexed,
        status="ingested" if chunks_indexed > 0 else "failed",
    )
