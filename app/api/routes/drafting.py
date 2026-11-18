"""POST /draft — AI drafting for the document studio (Server-Sent Events).

Always streams. Event sequence:

    sources_found  — library passages the draft may cite (only when
                     ``use_authorities`` is true). ``{count, sources}``.
    status         — progress line, ``{phase, message}``.
    delta          (write modes, repeated) — HTML fragment text as the
                     model produces it. Provisional: ``completed.text`` is
                     the cleaned, canonical version and replaces it.
    review         (review mode) — ``{summary, issues[]}``.
    completed      — terminal. ``{mode, text, sources}``.
    error          — terminal; ``data.message`` is safe to show as-is.

Input validation (wrong mode inputs, oversize fields) fails before the
stream opens, as a normal 422.
"""

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.dependencies.services import get_drafting_service
from app.api.dependencies.tenant import get_optional_tenant_context
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.core.tenant_context import TenantContext
from app.rag.streaming import to_async
from app.schemas.drafting import DraftRequest
from app.services.drafting_service import DraftingService

router = APIRouter(prefix="", tags=["drafting"])

# Drafting spends model tokens like /ask does, so it shares its budget.
_DRAFT_LIMIT = get_settings().rate_limit_ask


@router.post("/draft")
@limiter.limit(_DRAFT_LIMIT)
async def draft(
    request: Request,  # required by slowapi to resolve the key function
    payload: DraftRequest = Body(...),
    service: DraftingService = Depends(get_drafting_service),
    tenant: TenantContext = Depends(get_optional_tenant_context),
):
    return StreamingResponse(
        to_async(service.stream(payload, tenant=tenant)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
