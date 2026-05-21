"""Repository for chat sessions and messages (conversational memory)."""

from __future__ import annotations

import uuid

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.models.chat import ChatMessage, ChatSession


class ChatRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    # ---- sessions ----------------------------------------------------------

    def create_session(self, title: str | None = None) -> ChatSession:
        session = ChatSession(title=title)
        self._db.add(session)
        self._db.flush()
        return session

    def get_session(self, session_id: uuid.UUID) -> ChatSession | None:
        return self._db.get(ChatSession, session_id)

    # ---- messages ----------------------------------------------------------

    def add_message(
        self,
        session_id: uuid.UUID,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            metadata_json=metadata,
        )
        self._db.add(msg)
        self._db.flush()
        return msg

    def get_recent_messages(
        self, session_id: uuid.UUID, limit: int
    ) -> list[ChatMessage]:
        """Return the most recent `limit` messages in chronological order."""
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(desc(ChatMessage.created_at))
            .limit(limit)
        )
        rows = list(self._db.scalars(stmt).all())
        rows.reverse()
        return rows
