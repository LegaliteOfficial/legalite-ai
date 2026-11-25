"""Case-law intelligence endpoints.

All routes accept optional tenant headers via
``get_optional_tenant_context``. With no headers, retrieval is scoped to
GLOBAL only (legacy / anonymous behavior). With headers, GLOBAL + the
caller's TENANT_PRIVATE cases are visible.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies.services import get_case_intelligence_service
from app.api.dependencies.tenant import get_optional_tenant_context
from app.core.tenant_context import TenantContext
from app.schemas.cases import (
    CaseCompareRequest,
    CaseCompareResponse,
    CaseDetailOut,
    CaseListResponse,
    CaseSimilarRequest,
    CaseSimilarResponse,
    CaseSummarizeRequest,
    CaseSummarizeResponse,
    CitationGraphResponse,
)
from app.services.case_intelligence_service import CaseIntelligenceService

router = APIRouter(prefix="/cases", tags=["cases"])


@router.get("", response_model=CaseListResponse)
async def list_cases(
    court: str | None = Query(default=None, description="Filter by court name"),
    topic: str | None = Query(default=None, description="Filter by legal topic"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: CaseIntelligenceService = Depends(get_case_intelligence_service),
    tenant: TenantContext = Depends(get_optional_tenant_context),
) -> CaseListResponse:
    return service.list_cases(court=court, topic=topic, limit=limit, offset=offset, tenant=tenant)


@router.get("/{case_id}", response_model=CaseDetailOut)
async def get_case(
    case_id: uuid.UUID,
    service: CaseIntelligenceService = Depends(get_case_intelligence_service),
    tenant: TenantContext = Depends(get_optional_tenant_context),
) -> CaseDetailOut:
    case = service.get_case_detail(case_id, tenant=tenant)
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    return case


@router.post("/similar", response_model=CaseSimilarResponse)
async def find_similar_cases(
    payload: CaseSimilarRequest,
    service: CaseIntelligenceService = Depends(get_case_intelligence_service),
    tenant: TenantContext = Depends(get_optional_tenant_context),
) -> CaseSimilarResponse:
    try:
        return service.find_similar(
            case_id=payload.case_id,
            query=payload.query,
            top_k=payload.top_k,
            tenant=tenant,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post("/summarize", response_model=CaseSummarizeResponse)
async def summarize_case(
    payload: CaseSummarizeRequest,
    service: CaseIntelligenceService = Depends(get_case_intelligence_service),
    tenant: TenantContext = Depends(get_optional_tenant_context),
) -> CaseSummarizeResponse:
    result = service.summarize(payload.case_id, tenant=tenant)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    return result


@router.post("/compare", response_model=CaseCompareResponse)
async def compare_cases(
    payload: CaseCompareRequest,
    service: CaseIntelligenceService = Depends(get_case_intelligence_service),
    tenant: TenantContext = Depends(get_optional_tenant_context),
) -> CaseCompareResponse:
    result = service.compare(payload.case_ids, tenant=tenant)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="At least two cases must exist with the supplied ids",
        )
    return result


@router.get("/{case_id}/citations", response_model=CitationGraphResponse)
async def case_citation_graph(
    case_id: uuid.UUID,
    service: CaseIntelligenceService = Depends(get_case_intelligence_service),
    tenant: TenantContext = Depends(get_optional_tenant_context),
) -> CitationGraphResponse:
    """Outgoing/incoming citation edges + greedy precedent chain.

    Tenant-scoped: only edges whose endpoints are visible to the
    requesting org appear in the response.
    """
    graph = service.citation_graph(case_id, tenant=tenant)
    if graph is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    return graph
