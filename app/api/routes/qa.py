"""POST /ask — answer a Ghana legal question grounded in retrieved passages.

Backward compatible: requests without `session_id` still work; responses
keep all v1 fields and add v2 fields (`sources_used`, `reasoning_summary`,
`session_id`) and v3 fields (`structured_answer`, `query_intent`).

Phase 5 M7 (+ token-streaming follow-up) — passing ``?stream=true`` opts
into Server-Sent Events:

    POST /ask?stream=true
    Accept: text/event-stream

Event sequence:

    retrieval_started  — fired immediately, before retrieval runs.
    sources_found       — passage count + scope breakdown, right after
                           retrieval, before the LLM call starts.
    answer_delta         (repeated) — ``direct_answer`` text as OpenAI
                           generates it. Zero or more of these fire while
                           the model is still producing output.
    then either:
      refused            — terminal. The post-generation safety checks
                           (grounding / consistency) rejected the answer
                           AFTER it finished generating. Its ``answer`` /
                           ``structured_answer`` are canonical and may
                           differ from what the ``answer_delta`` events
                           already showed — clients MUST replace whatever
                           they rendered from the deltas with this
                           event's payload, not merge the two.
      or:
      reasoning           — one-shot summary once generation passed the
                           safety checks.
      citations            — structured citation arrays.
      completed  — terminal. Full AskResponse mirror, matches the
                           non-streaming response shape exactly.

    status              — (chat agent) human-readable progress line, e.g.
                           ``{"phase": "searching", "message": "Searching the
                           Legalite library: …", "query": "…"}`` or
                           ``{"phase": "answering", …}``. May repeat — the
                           agent can run several library searches per turn,
                           and ``sources_found`` may fire once per search.
                           Clients that don't know it can ignore it.

    error               — terminal, can appear at any point after
                           ``retrieval_started``. Once the first byte of
                           an SSE response is sent, FastAPI can no longer
                           turn a mid-pipeline exception into a proper
                           500 — the connection would just drop instead.
                           This event exists so a failure still reaches
                           the client with an explanation rather than a
                           silently truncated stream. ``data.message`` is
                           safe to show to the user as-is.

The non-streaming response shape (``AskResponse``) is unchanged.
"""

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.api.dependencies.services import get_qa_service
from app.api.dependencies.tenant import get_optional_tenant_context
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.core.tenant_context import TenantContext
from app.rag.prompt_builder import MemoryTurn
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
    authorization: str | None = Header(
        default=None,
        description=(
            "Optional `Bearer <token>` of the signed-in Legalite user. When "
            "present, the agent can read that user's cases, deadlines, tasks "
            "and documents via the backend (which validates the token) and "
            "propose new records. Never stored or logged."
        ),
    ),
):
    # NOTE: return-type union annotation removed so slowapi + FastAPI introspection
    # plays nicely with `from __future__ import annotations` not being available here.
    if not payload.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question must not be empty",
        )
    user_token = _bearer_token(authorization)
    client_history = (
        [MemoryTurn(role=t.role, content=t.content) for t in payload.history if t.content.strip()]
        if payload.history
        else None
    )
    if stream:
        events = service.ask_stream(
            payload.question,
            session_id=payload.session_id,
            tenant=tenant,
            context=payload.context,
            user_token=user_token,
            client_history=client_history,
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
            context=payload.context,
            user_token=user_token,
            client_history=client_history,
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


def _bearer_token(header: str | None) -> str | None:
    """Extract the token from ``Authorization: Bearer <token>``."""
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token or len(token) > 8192:
        return None
    return token
