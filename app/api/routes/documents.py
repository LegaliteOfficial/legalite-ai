"""Legacy document upload — preserved for the current mobile/web client.

Both the original ``POST /upload-law`` and the new tenant-aware uploads share
``IngestionService.ingest_pdf``. The legacy route maps to GLOBAL visibility
unconditionally (matching pre-tenancy behavior); ``POST /admin/upload-law``
lives in ``admin.py``, ``POST /documents/private-upload`` in
``private_documents.py``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies.services import get_ingestion_service
from app.core.config import get_settings
from app.db.enums import VisibilityScope
from app.schemas.documents import UploadLawResponse
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="", tags=["documents"])


@router.post(
    "/upload-law",
    response_model=UploadLawResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_law(
    file: UploadFile = File(..., description="Ghana legal PDF"),
    title: str = Form(..., min_length=2, max_length=512),
    doc_type: str = Form(..., description="Constitution | Act | Regulation | Case Law | ..."),
    service: IngestionService = Depends(get_ingestion_service),
) -> UploadLawResponse:
    """Legacy ingestion endpoint — always GLOBAL.

    Kept for backward compatibility with the current mobile/web client.
    New code should call ``/admin/upload-law`` (GLOBAL, admin-gated) or
    ``/documents/private-upload`` (TENANT_PRIVATE, requires org context).
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
        )
    except Exception as exc:  # noqa: BLE001 — surface a clean 500 to the client
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
