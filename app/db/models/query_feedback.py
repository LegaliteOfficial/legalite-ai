"""ORM model for the query_feedback table.

See ``app/db/migrations/versions/0008_query_feedback.py`` for the full
column-by-column rationale. In short: one row per assistant turn that
received a thumbs-up or thumbs-down, with the analytical context
(intent, confidence, refused) denormalised so dashboards don't pay JSON
extraction on every read.
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class QueryFeedback(Base, TimestampMixin):
    __tablename__ = "query_feedback"
    __table_args__ = (
        CheckConstraint("thumbs IN ('up', 'down')", name="ck_query_feedback_thumbs"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # ``message_id`` is unique — re-posting feedback for the same turn
    # UPDATEs the existing row rather than appending duplicates. Cascade
    # delete keeps the table clean when a session is purged.
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )

    thumbs: Mapped[str] = mapped_column(String(8), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Denormalised analytical context — captured at feedback time.
    question_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer_confidence: Mapped[str | None] = mapped_column(String(16), nullable=True)
    query_intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    refused: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
