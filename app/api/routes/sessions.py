"""Chat session management.

Tenant-scoped (audit fix P0-1). A session created by Firm A is invisible
to Firm B and to anonymous callers — they get a 404. Anonymous sessions
(created without tenant headers) remain visible to other anonymous
callers; this preserves the legacy /ask + /sessions flow used by the
current mobile/web client.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.dependencies.tenant import get_optional_tenant_context
from app.core.tenant_context import TenantContext
from app.db.repositories.chat_repository import ChatRepository
from app.db.session import get_db

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionResponse(BaseModel):
    session_id: uuid.UUID


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str


class SessionDetail(BaseModel):
    session_id: uuid.UUID
    messages: list[MessageOut]


@router.post("", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_optional_tenant_context),
) -> CreateSessionResponse:
    repo = ChatRepository(db)
    sess = repo.create_session(
        organization_id=tenant.organization_id,
        user_id=tenant.user_id,
    )
    db.commit()
    return CreateSessionResponse(session_id=sess.id)


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_optional_tenant_context),
) -> SessionDetail:
    repo = ChatRepository(db)
    sess = repo.get_session(session_id, organization_id=tenant.organization_id)
    if sess is None:
        # 404 (not 403) so cross-tenant probing can't enumerate session IDs.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found")
    messages = repo.get_recent_messages(session_id, limit=200)
    return SessionDetail(
        session_id=session_id,
        messages=[MessageOut(id=m.id, role=m.role, content=m.content) for m in messages],
    )
