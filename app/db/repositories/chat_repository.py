"""Repository for chat sessions and messages (conversational memory).

Tenant scope (post-audit fix): ``create_session`` stamps the session row
with the caller's ``organization_id`` + ``user_id``; ``get_session`` is the
single tenant-gated read path used by both the /sessions route and the
conversational-memory layer. Callers MUST pass the requesting org_id
through — a missing tenant scope is treated as anonymous (org=NULL) and
will only match sessions also created anonymously.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.db.models.chat import ChatMessage, ChatSession


class ChatRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    # ---- sessions ----------------------------------------------------------

    def create_session(
        self,
        title: str | None = None,
        *,
        organization_id: uuid.UUID | None = None,
        user_id: uuid.UUID | None = None,
    ) -> ChatSession:
        session = ChatSession(
            title=title,
            organization_id=organization_id,
            user_id=user_id,
        )
        self._db.add(session)
        self._db.flush()
        return session

    def get_session(
        self,
        session_id: uuid.UUID,
        *,
        organization_id: uuid.UUID | None = None,
    ) -> ChatSession | None:
        """Return the session iff it belongs to the requesting tenant scope.

        Audit rule (P0-1): a session created with an ``organization_id`` is
        ONLY readable by a request whose ``TenantContext.organization_id``
        matches. A session created anonymously (org=NULL) is only readable
        by anonymous callers. There is no path that returns a tenant
        session to a different tenant.
        """
        session = self._db.get(ChatSession, session_id)
        if session is None:
            return None
        if session.organization_id != organization_id:
            # Cross-tenant probe — pretend the row doesn't exist.
            return None
        return session

    def commit(self) -> None:
        """Make chat writes durable now.

        Needed because /ask?stream=true writes inside the SSE generator,
        which FastAPI runs AFTER the request-scoped ``get_db`` dependency
        has already committed and closed. Without an explicit commit those
        rows are rolled back when the connection returns to the pool — the
        assistant then "forgets" every earlier turn.
        """
        self._db.commit()

    # ---- messages ----------------------------------------------------------

    def add_message(
        self,
        session_id: uuid.UUID,
        role: str,
        content: str,
        metadata: dict | None = None,
        created_at: datetime | None = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            metadata_json=metadata,
        )
        if created_at is not None:
            # Explicit timestamps keep order when several rows are written
            # in one transaction (now() is the transaction start time).
            msg.created_at = created_at
        self._db.add(msg)
        self._db.flush()
        return msg

    def get_recent_messages(self, session_id: uuid.UUID, limit: int) -> list[ChatMessage]:
        """Return the most recent `limit` messages in chronological order.

        Callers MUST gate by ``get_session(session_id, organization_id=…)``
        first — this method does not re-check tenancy because joining
        ``chat_messages`` to ``chat_sessions`` on every read would double
        the query cost for no extra guarantee.
        """
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(desc(ChatMessage.created_at))
            .limit(limit)
        )
        rows = list(self._db.scalars(stmt).all())
        rows.reverse()
        return rows

    def get_user_messages(self, session_id: uuid.UUID, limit: int) -> list[str]:
        """Every user message in the session, oldest first (capped)."""
        stmt = (
            select(ChatMessage.content)
            .where(ChatMessage.session_id == session_id, ChatMessage.role == "user")
            .order_by(ChatMessage.created_at)
            .limit(limit)
        )
        return list(self._db.scalars(stmt).all())

    def count_messages(self, session_id: uuid.UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(ChatMessage)
            .where(ChatMessage.session_id == session_id)
        )
        return int(self._db.scalar(stmt) or 0)

    def get_messages_range(
        self, session_id: uuid.UUID, offset: int, limit: int
    ) -> list[ChatMessage]:
        """Messages ``[offset, offset+limit)`` in chronological order."""
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .offset(offset)
            .limit(limit)
        )
        return list(self._db.scalars(stmt).all())

    def get_session_meta(self, session_id: uuid.UUID) -> dict:  # type: ignore[type-arg]
        session = self._db.get(ChatSession, session_id)
        return dict(session.metadata_json or {}) if session else {}

    def set_session_meta(self, session_id: uuid.UUID, meta: dict) -> None:  # type: ignore[type-arg]
        session = self._db.get(ChatSession, session_id)
        if session is not None:
            session.metadata_json = meta
            self._db.flush()
