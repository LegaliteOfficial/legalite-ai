"""POST /ask — answer a Ghana legal question grounded in retrieved passages.

Backward compatible: requests without `session_id` still work; responses
keep all v1 fields and add v2 fields (`sources_used`, `reasoning_summary`,
`session_id`) and v3 fields (`structured_answer`, `query_intent`).

Phase 5 M7 — passing ``?stream=true`` opts into Server-Sent Events:

    POST /ask?stream=true
    Accept: text/event-stream

Streaming sends incremental events around the existing single-call
pipeline (``retrieval_started``, ``sources_found``, ``reasoning``,
``direct_answer``, ``citations``, ``completed`` / ``refused``). The
non-streaming response shape is unchanged.
"""

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.api.dependencies.services import get_qa_service
from app.api.dependencies.tenant import get_optional_tenant_context
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.core.tenant_context import TenantContext
from app.rag.streaming import to_async
from app.schemas.qa import AskRequest, AskResponse
from app.services.qa_service import QAService

router = APIRouter(prefix="", tags=["qa"])

# Limits are resolved at import time. The Settings instance is cached
# (lru_cache), so changing env vars requires a process restart — same
# contract as every other setting in this codebase.
_ASK_LIMIT = get_settings().rate_limit_ask


@router.post("/ask", response_model=AskResponse)
@limiter.limit(_ASK_LIMIT)
async def ask(
    request: Request,  # required by slowapi to resolve the key function
    payload: AskRequest = Body(...),
    stream: bool = Query(
        default=False,
        description=(
            "When true, return Server-Sent Events instead of a single JSON "
            "body. The same pipeline runs; events are emitted as milestones "
            "are reached."
        ),
    ),
    service: QAService = Depends(get_qa_service),
    tenant: TenantContext = Depends(get_optional_tenant_context),
):
    # NOTE: return-type union annotation removed so slowapi + FastAPI introspection
    # plays nicely with `from __future__ import annotations` not being available here.
    if stream:
        events = service.ask_stream(
            payload.question,
            session_id=payload.session_id,
            tenant=tenant,
        )
        return StreamingResponse(
            to_async(events),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable buffering on nginx proxies
            },
        )
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
