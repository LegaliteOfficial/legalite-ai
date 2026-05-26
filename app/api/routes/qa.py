"""POST /ask — answer a Ghana legal question grounded in retrieved passages.

Backward compatible: requests without `session_id` still work; responses
keep all v1 fields and add v2 fields (`sources_used`, `reasoning_summary`,
`session_id`).

Tenant scope is taken from the gateway headers via
``get_optional_tenant_context``. Requests without those headers (legacy
mobile/web client) fall back to anonymous → GLOBAL-only retrieval, which
preserves pre-tenancy behavior exactly.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies.services import get_qa_service
from app.api.dependencies.tenant import get_optional_tenant_context
from app.core.tenant_context import TenantContext
from app.schemas.qa import AskRequest, AskResponse
from app.services.qa_service import QAService

router = APIRouter(prefix="", tags=["qa"])


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    service: QAService = Depends(get_qa_service),
    tenant: TenantContext = Depends(get_optional_tenant_context),
) -> AskResponse:
    try:
        return service.ask(
            payload.question,
            session_id=payload.session_id,
            tenant=tenant,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Q&A failed: {exc}",
        ) from exc
