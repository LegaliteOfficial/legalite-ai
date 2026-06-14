"""Feedback endpoint — the self-learning loop's user-facing surface.

POST /ask/{message_id}/feedback
    Body:  {"thumbs": "up" | "down", "comment": "optional"}
    Auth:  anonymous OR tenant-signed (matches /ask).
    Idem:  re-posting the same message_id UPDATES the existing row.

This endpoint is intentionally tiny. The heavy lifting (analytics
snapshot, tenant gating, upsert semantics) all lives in
:class:`FeedbackRepository`. The route exists to:

  - Convert the path param to a UUID and 400 on garbage input.
  - Translate ``LookupError`` (wrong tenant / message missing) to 404
    so anonymous callers can't probe the tenant message graph.
  - Translate ``ValueError`` (bad thumbs / non-assistant message) to 422.

Why under /ask rather than its own prefix
-----------------------------------------
Feedback is logically *part of the /ask transaction* — the response
ships back a ``message_id`` and the client uses it within the same
session. Keeping the URL adjacent to /ask makes that contract obvious
to anyone reading the OpenAPI page. The implementation is in a separate
router so qa.py stays scoped to the question-answering flow.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies.tenant import get_optional_tenant_context
from app.core.telemetry import feedback_total
from app.core.tenant_context import TenantContext
from app.db.repositories.feedback_repository import FeedbackRepository
from app.db.session import get_db
from app.schemas.feedback import FeedbackCreate, FeedbackResponse

router = APIRouter(prefix="/ask", tags=["feedback"])


@router.post(
    "/{message_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Attach 👍/👎 feedback to an assistant turn.",
)
def post_feedback(
    message_id: str,
    payload: FeedbackCreate,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_optional_tenant_context),
) -> FeedbackResponse:
    try:
        mid = uuid.UUID(message_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"message_id is not a valid UUID: {message_id!r}",
        ) from exc

    repo = FeedbackRepository(db)
    org_id = tenant.organization_id if tenant else None
    user_id = tenant.user_id if tenant else None

    try:
        row = repo.upsert(
            message_id=mid,
            thumbs=payload.thumbs,
            comment=payload.comment,
            organization_id=org_id,
            user_id=user_id,
        )
    except LookupError as exc:
        # Either the message doesn't exist or it belongs to another
        # tenant. 404 either way — never reveal cross-tenant existence.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="message not found",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    db.commit()

    # Self-learning loop instrumentation: every persisted feedback
    # increments the dashboard counter. Labels match the analytical
    # columns on the row so PromQL queries can pivot the same way SQL
    # would. ``query_intent`` may be None for pre-Phase-A turns whose
    # metadata didn't carry an intent; coerce to ``unspecified`` so the
    # cardinality stays bounded.
    feedback_total.labels(
        thumbs=row.thumbs,
        query_intent=row.query_intent or "unspecified",
        confidence=row.answer_confidence or "unspecified",
        refused="true" if row.refused else "false",
    ).inc()

    return FeedbackResponse(
        id=str(row.id),
        message_id=str(row.message_id),
        session_id=str(row.session_id),
        thumbs=row.thumbs,  # type: ignore[arg-type]
        comment=row.comment,
        answer_confidence=row.answer_confidence,
        query_intent=row.query_intent,
        refused=row.refused,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
