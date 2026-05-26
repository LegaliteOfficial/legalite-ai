"""Admin-only routes for the GLOBAL Ghana legal corpus.

``POST /admin/upload-law`` is the new home for ingesting Constitution / Acts
/ Regulations / public case law. It's the same pipeline as the legacy
``/upload-law``, but role-gated: only ``owner`` or ``admin`` headers from
the legalite-backend gateway pass the dependency.

The legacy route stays alive (in ``documents.py``) so the current mobile/web
client keeps working; new operator tooling should target this endpoint.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies.services import get_ingestion_service
from app.api.dependencies.tenant import require_admin
from app.core.config import get_settings
from app.core.tenant_context import TenantContext
from app.db.enums import DocumentOrigin, VisibilityScope
from app.schemas.documents import UploadLawResponse
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post(
    "/upload-law",
    response_model=UploadLawResponse,
    status_code=status.HTTP_201_CREATED,
)
async def admin_upload_law(
    file: UploadFile = File(..., description="Ghana legal PDF (Constitution, Act, Regulation, Case)"),
    title: str = Form(..., min_length=2, max_length=512),
    doc_type: str = Form(..., description="Constitution | Act | Regulation | Case Law | ..."),
    document_origin: DocumentOrigin | None = Form(
        default=None,
        description=(
            "Optional explicit classification. If omitted, the chunker's auto-detected "
            "law_type is used. Must be one of: constitution | act | regulation | "
            "case_law | precedent | judgement."
        ),
    ),
    admin_ctx: TenantContext = Depends(require_admin),
    service: IngestionService = Depends(get_ingestion_service),
) -> UploadLawResponse:
    """Ingest a global / authoritative legal document. Admin-only.

    The uploaded user_id is recorded on the document for audit, but
    ``organization_id`` is left NULL — global documents belong to nobody.
    """
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
            visibility_scope=VisibilityScope.GLOBAL,
            document_origin=document_origin,
            tenant=admin_ctx,  # only uploaded_by_user_id is used; org_id is forced None
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
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
