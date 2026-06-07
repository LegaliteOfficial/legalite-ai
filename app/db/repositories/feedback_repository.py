"""Repository for ``query_feedback`` — the self-improvement loop's data layer.

Tenant scoping (audit-mirror of ChatRepository): every read and write
threads ``organization_id`` so a tenant can never see or overwrite
another tenant's feedback. The upsert is keyed on ``message_id``; the
unique constraint at the schema level makes that race-safe.
"""

from __future__ import annotations

import uuid

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.db.models.chat import ChatMessage, ChatSession
from app.db.models.query_feedback import QueryFeedback


class FeedbackRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    # ---- writes ------------------------------------------------------------

    def upsert(
        self,
        *,
        message_id: uuid.UUID,
        thumbs: str,
        comment: str | None,
        organization_id: uuid.UUID | None,
        user_id: uuid.UUID | None,
    ) -> QueryFeedback:
        """Insert a new feedback row OR update the existing one for this turn.

        Re-thumbing the same assistant turn does NOT create a new row —
        it updates ``thumbs``/``comment``/``updated_at`` on the existing
        row. That keeps analytics clean (one row per turn) and makes
        "I changed my mind" UX trivial.

        Cross-tenant guard: the assistant turn's session must belong to
        the caller's tenant scope. We resolve session→message inside the
        same transaction so a parallel session move can't slip a wrong
        org_id in. Returns ``None`` from the caller's perspective by
        raising — see :meth:`get_message_for_tenant` to pre-check.

        Denormalised analytical columns (``question_text``,
        ``answer_confidence``, ``query_intent``, ``refused``) are
        copied from the assistant turn and its preceding user turn at
        write time; see :meth:`_capture_analytics_snapshot`.
        """
        thumbs = thumbs.strip().lower()
        if thumbs not in ("up", "down"):
            raise ValueError(f"thumbs must be 'up' or 'down', got {thumbs!r}")

        assistant_msg = self.get_message_for_tenant(
            message_id, organization_id=organization_id
        )
        if assistant_msg is None:
            raise LookupError("message not found in caller's tenant scope")
        if assistant_msg.role != "assistant":
            # Feedback only makes sense on assistant turns.
            raise ValueError(
                "feedback can only be attached to assistant messages, "
                f"not role={assistant_msg.role!r}"
            )

        snapshot = self._capture_analytics_snapshot(assistant_msg)

        existing = self._db.execute(
            select(QueryFeedback).where(QueryFeedback.message_id == message_id)
        ).scalar_one_or_none()

        if existing is None:
            row = QueryFeedback(
                message_id=message_id,
                session_id=assistant_msg.session_id,
                organization_id=organization_id,
                user_id=user_id,
                thumbs=thumbs,
                comment=comment,
                question_text=snapshot["question_text"],
                answer_confidence=snapshot["answer_confidence"],
                query_intent=snapshot["query_intent"],
                refused=snapshot["refused"],
            )
            self._db.add(row)
            self._db.flush()
            return row

        existing.thumbs = thumbs
        existing.comment = comment
        # Refresh the analytics snapshot in case the assistant turn's
        # metadata was updated server-side (e.g. delayed grounding
        # check); cheap and keeps the row consistent.
        existing.question_text = snapshot["question_text"]
        existing.answer_confidence = snapshot["answer_confidence"]
        existing.query_intent = snapshot["query_intent"]
        existing.refused = snapshot["refused"]
        self._db.flush()
        return existing

    # ---- reads -------------------------------------------------------------

    def get_message_for_tenant(
        self,
        message_id: uuid.UUID,
        *,
        organization_id: uuid.UUID | None,
    ) -> ChatMessage | None:
        """Return the assistant message iff its session belongs to ``organization_id``.

        Looks like a chat-repository concern, but lives here because
        feedback is the only caller that resolves message→session→tenant
        as a single check. Centralising it in ChatRepository would add a
        public method used by exactly one endpoint.
        """
        stmt = (
            select(ChatMessage)
            .join(ChatSession, ChatSession.id == ChatMessage.session_id)
            .where(ChatMessage.id == message_id)
            .where(ChatSession.organization_id.is_not_distinct_from(organization_id))
        )
        return self._db.execute(stmt).scalar_one_or_none()

    def list_recent_for_org(
        self,
        organization_id: uuid.UUID | None,
        *,
        limit: int = 100,
    ) -> list[QueryFeedback]:
        """Most recent feedback for a tenant (NULL = anonymous bucket)."""
        stmt = (
            select(QueryFeedback)
            .where(
                QueryFeedback.organization_id.is_not_distinct_from(organization_id)
            )
            .order_by(desc(QueryFeedback.created_at))
            .limit(limit)
        )
        return list(self._db.execute(stmt).scalars())

    def summary_by_intent(
        self,
        organization_id: uuid.UUID | None,
    ) -> list[dict]:
        """Aggregate 👍/👎 counts by query intent.

        The Tier-1 gap dashboard's primary view. Returns one row per
        intent with up_count/down_count/refused_count. Empty list if
        no feedback exists for the tenant.
        """
        thumbs_col = QueryFeedback.thumbs
        stmt = (
            select(
                QueryFeedback.query_intent,
                func.count().filter(thumbs_col == "up").label("up_count"),
                func.count().filter(thumbs_col == "down").label("down_count"),
                func.count()
                .filter(QueryFeedback.refused.is_(True))
                .label("refused_count"),
            )
            .where(
                QueryFeedback.organization_id.is_not_distinct_from(organization_id)
            )
            .group_by(QueryFeedback.query_intent)
            .order_by(desc("down_count"))
        )
        rows = self._db.execute(stmt).all()
        return [
            {
                "query_intent": r.query_intent or "unspecified",
                "up_count": int(r.up_count or 0),
                "down_count": int(r.down_count or 0),
                "refused_count": int(r.refused_count or 0),
            }
            for r in rows
        ]

    # ---- internals ---------------------------------------------------------

    def _capture_analytics_snapshot(self, assistant_msg: ChatMessage) -> dict:
        """Extract analytical context from the assistant turn + its prompt.

        We pull ``question_text`` from the most recent user message in the
        same session that precedes this assistant turn. The metadata on
        the assistant message carries confidence / refused / etc. (see
        ``QAService.ask`` where it's written).
        """
        meta = assistant_msg.metadata_json or {}

        # The user turn that triggered this assistant reply is the most
        # recent user-role message in the session with created_at < this
        # assistant message's created_at.
        question_stmt = (
            select(ChatMessage.content)
            .where(ChatMessage.session_id == assistant_msg.session_id)
            .where(ChatMessage.role == "user")
            .where(ChatMessage.created_at <= assistant_msg.created_at)
            .order_by(desc(ChatMessage.created_at))
            .limit(1)
        )
        question_text = self._db.execute(question_stmt).scalar_one_or_none()

        return {
            "question_text": question_text,
            "answer_confidence": meta.get("confidence"),
            "query_intent": meta.get("query_intent"),
            "refused": bool(meta.get("refused", False)),
        }
