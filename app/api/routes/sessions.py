"""Chat session management.

POST /sessions      — create a fresh chat session
GET  /sessions/{id} — list messages in chronological order
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

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
async def create_session(db: Session = Depends(get_db)) -> CreateSessionResponse:
    repo = ChatRepository(db)
    sess = repo.create_session()
    db.commit()
    return CreateSessionResponse(session_id=sess.id)


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: uuid.UUID, db: Session = Depends(get_db)
) -> SessionDetail:
    repo = ChatRepository(db)
    sess = repo.get_session(session_id)
    if sess is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="session not found"
        )
    messages = repo.get_recent_messages(session_id, limit=200)
    return SessionDetail(
        session_id=session_id,
        messages=[
            MessageOut(id=m.id, role=m.role, content=m.content) for m in messages
        ],
    )
