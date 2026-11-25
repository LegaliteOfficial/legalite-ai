"""Firm archive analysis.

``POST /archive/analyze`` is called by the NestJS backend (never by the
browser) for each document a firm uploads to its library. It returns the
document's extracted text plus a structured, verified description; the
backend stores that as the firm's archive entry.

Requires a tenant context (``x-organization-id`` / ``x-user-id``), signed
with ``LEGALITE_GATEWAY_SECRET`` when that is configured, so the endpoint
cannot be used anonymously to spend model tokens.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.dependencies.tenant import get_required_tenant_context
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.rate_limit import limiter
from app.core.tenant_context import TenantContext
from app.schemas.archive import ArchiveAnalysis
from app.services.archive_analyzer import ArchiveAnalyzer, UnsupportedDocumentError

logger = get_logger(__name__)

router = APIRouter(prefix="/archive", tags=["archive"])

_UPLOAD_LIMIT = get_settings().rate_limit_upload


def get_archive_analyzer() -> ArchiveAnalyzer:
    return ArchiveAnalyzer()


@router.post("/analyze", response_model=ArchiveAnalysis)
@limiter.limit(_UPLOAD_LIMIT)
async def analyze_document(
    request: Request,
    file: UploadFile = File(..., description="The uploaded document"),
    file_name: str | None = Form(default=None, max_length=512),
    category: str | None = Form(default=None, max_length=40),
    tenant: TenantContext = Depends(get_required_tenant_context),
    analyzer: ArchiveAnalyzer = Depends(get_archive_analyzer),
) -> ArchiveAnalysis:
    settings = get_settings()
    name = file_name or file.filename or "document"
    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_mb}MB limit",
        )
    try:
        result = await run_in_threadpool(analyzer.analyze, content, name, category=category)
    except UnsupportedDocumentError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    logger.info(
        "archive.analyzed",
        organization_id=str(tenant.organization_id),
        ext=Path(name).suffix.lower(),
        method=result["extraction_method"],
        pages=result["page_count"],
        words=result["word_count"],
        doc_type=result["document_type"],
    )
    return ArchiveAnalysis(**result)
