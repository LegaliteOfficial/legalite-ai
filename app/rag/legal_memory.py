"""Postgres-backed conversational memory for the legal assistant.

We persist messages so a follow-up like "what if I was pregnant?" can be
interpreted against the previous turn ("can my employer dismiss me?").

Design choices:
- **Postgres, not in-process**: sessions survive restarts and are shared
  across workers.
- **Window + token cap**: we never feed the LLM unbounded history. The
  window keeps the *last N turns* and we truncate by approximate tokens so
  prompts stay deterministic in size.
- **Token estimate is character-based (chars / 4)** — good enough for
  budgeting; we don't ship tiktoken into the hot path for this.
- The retrieval and citation stages are NOT memory-aware: they still ground
  on documents only. Memory only enters at prompt construction.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.tenant_context import TenantContext
from app.db.models.chat import ChatMessage
from app.db.repositories.chat_repository import ChatRepository
from app.rag.prompt_builder import MemoryTurn

_CHARS_PER_TOKEN = 4  # rough estimator

logger = get_logger(__name__)

# (previous summary or None, newly aged-out turns) -> updated summary
Summarizer = Callable[[str | None, list[MemoryTurn]], str]


@dataclass(slots=True)
class ConversationMemory:
    """Everything the agent gets to remember about one conversation.

    - ``recent``: the last turns, verbatim (window + token budget).
    - ``summary``: rolling LLM summary of turns older than ``recent``.
    - ``user_messages``: every user message so far, oldest first (capped),
      so exact recall questions ("what did I ask first?") are answerable
      however long the conversation gets.
    - ``earlier_count``: messages not shown verbatim.
    """

    recent: list[MemoryTurn]
    summary: str | None = None
    user_messages: list[str] = field(default_factory=list)
    earlier_count: int = 0


@dataclass(slots=True)
class MemoryContext:
    session_id: uuid.UUID
    turns: list[MemoryTurn]


class LegalMemory:
    def __init__(self, repo: ChatRepository) -> None:
        self._repo = repo

    def get_or_create_session(
        self,
        session_id: str | None,
        *,
        tenant: TenantContext | None = None,
    ) -> uuid.UUID:
        """Resolve to a session UUID, scoped to the requesting tenant.

        Audit rule (P0-1): a caller can only resume a session that
        belongs to their own ``organization_id``. If they hand us a
        session_id that exists under a different org (or anonymous), we
        treat it as missing and mint a fresh session under the current
        tenant — never leak the other org's session.
        """
        org_id = tenant.organization_id if tenant else None
        user_id = tenant.user_id if tenant else None

        if session_id:
            try:
                sid = uuid.UUID(session_id)
            except ValueError as e:
                raise ValueError(f"invalid session_id: {session_id}") from e
            existing = self._repo.get_session(sid, organization_id=org_id)
            if existing is not None:
                return sid
            # session_id was bogus or belongs to a different tenant — start fresh.
        new_id = self._repo.create_session(organization_id=org_id, user_id=user_id).id
        self._repo.commit()
        return new_id

    def load_recent(self, session_id: uuid.UUID) -> MemoryContext:
        settings = get_settings()
        # window is in turns; one turn ≈ user + assistant pair, so fetch ~2x
        fetch_limit = settings.memory_window_turns * 2
        messages = self._repo.get_recent_messages(session_id, limit=fetch_limit)
        turns = _truncate_to_budget(
            [_to_turn(m) for m in messages],
            max_tokens=settings.memory_max_tokens,
        )
        return MemoryContext(session_id=session_id, turns=turns)

    def load_history(
        self,
        session_id: uuid.UUID,
        *,
        window_turns: int,
        max_tokens: int,
    ) -> list[MemoryTurn]:
        """Wider history window for the chat agent.

        Same storage and budget rule as :meth:`load_recent`, but with the
        agent's own (larger) window so follow-ups can refer back further.
        """
        messages = self._repo.get_recent_messages(session_id, limit=window_turns * 2)
        return _truncate_to_budget([_to_turn(m) for m in messages], max_tokens=max_tokens)

    def load_conversation(
        self,
        session_id: uuid.UUID,
        *,
        window_turns: int,
        max_tokens: int,
        summarizer: Summarizer | None = None,
        summarize_batch: int = 4,
        outline_limit: int = 100,
    ) -> ConversationMemory:
        """Recent turns verbatim + rolling summary + full user-message outline.

        The summary is brought up to date lazily: once at least
        ``summarize_batch`` messages have aged out of the verbatim window
        since the last summary, they are folded into it (one cheap LLM
        call) and the result is stored on the session's ``metadata_json``.
        A failed summary never fails the turn.
        """
        total = self._repo.count_messages(session_id)
        recent = _truncate_to_budget(
            [
                _to_turn(m)
                for m in self._repo.get_recent_messages(session_id, limit=window_turns * 2)
            ],
            max_tokens=max_tokens,
        )
        earlier = max(0, total - len(recent))
        meta = self._repo.get_session_meta(session_id) if earlier else {}
        summary: str | None = meta.get("summary")
        covered = int(meta.get("summary_covers") or 0)
        if summarizer is not None and earlier - covered >= summarize_batch:
            aged = self._repo.get_messages_range(session_id, covered, earlier - covered)
            try:
                summary = summarizer(summary, [_to_turn(m) for m in aged])
                meta.update(summary=summary, summary_covers=earlier)
                self._repo.set_session_meta(session_id, meta)
                self._repo.commit()
            except Exception:  # noqa: BLE001 — memory is best-effort
                logger.exception("memory.summary_failed", session_id=str(session_id))
        users = self._repo.get_user_messages(session_id, limit=outline_limit) if total else []
        return ConversationMemory(
            recent=recent, summary=summary, user_messages=users, earlier_count=earlier
        )

    def import_history(self, session_id: uuid.UUID, turns: list[MemoryTurn]) -> None:
        """Seed an EMPTY server session with the client's local transcript.

        Used when the browser has a conversation the server has no record
        of (e.g. sessions lost before chat writes were committed), so the
        assistant still remembers it and future turns build on it.
        """
        if not turns or self._repo.count_messages(session_id):
            return
        # Back-date well clear of "now" so imported turns sort before any new
        # message even if the app and database clocks are skewed.
        base = datetime.now(UTC) - timedelta(hours=1)
        for i, t in enumerate(turns):
            role = "assistant" if t.role == "assistant" else "user"
            self._repo.add_message(
                session_id,
                role=role,
                content=t.content,
                metadata={"imported_from_client": True},
                created_at=base + timedelta(seconds=i),
            )
        self._repo.commit()

    def append_user(self, session_id: uuid.UUID, content: str) -> None:
        self._repo.add_message(session_id, role="user", content=content)
        self._repo.commit()

    def append_assistant(
        self,
        session_id: uuid.UUID,
        content: str,
        metadata: dict | None = None,
    ) -> uuid.UUID:
        """Append the assistant turn and return its message id.

        Callers expose this id back to the client so users can attach
        feedback via ``POST /ask/{message_id}/feedback``. Returning the
        id is cheap and avoids a follow-up SELECT just to find the row
        we just wrote.
        """
        msg = self._repo.add_message(
            session_id, role="assistant", content=content, metadata=metadata
        )
        self._repo.commit()
        return msg.id


def _to_turn(m: ChatMessage) -> MemoryTurn:
    return MemoryTurn(role=m.role, content=m.content)


def _truncate_to_budget(turns: list[MemoryTurn], max_tokens: int) -> list[MemoryTurn]:
    """Drop earliest turns until the estimated token cost fits the budget."""
    budget = max_tokens * _CHARS_PER_TOKEN
    total = sum(len(t.content) for t in turns)
    while turns and total > budget:
        dropped = turns.pop(0)
        total -= len(dropped.content)
    return turns
