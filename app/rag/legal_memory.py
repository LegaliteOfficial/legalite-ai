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
from dataclasses import dataclass

from app.core.config import get_settings
from app.db.models.chat import ChatMessage
from app.db.repositories.chat_repository import ChatRepository
from app.rag.prompt_builder import MemoryTurn

_CHARS_PER_TOKEN = 4  # rough estimator


@dataclass(slots=True)
class MemoryContext:
    session_id: uuid.UUID
    turns: list[MemoryTurn]


class LegalMemory:
    def __init__(self, repo: ChatRepository) -> None:
        self._repo = repo

    def get_or_create_session(self, session_id: str | None) -> uuid.UUID:
        if session_id:
            try:
                sid = uuid.UUID(session_id)
            except ValueError as e:
                raise ValueError(f"invalid session_id: {session_id}") from e
            if self._repo.get_session(sid) is None:
                # Create with provided id is not supported; create new + return new id
                return self._repo.create_session().id
            return sid
        return self._repo.create_session().id

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

    def append_user(self, session_id: uuid.UUID, content: str) -> None:
        self._repo.add_message(session_id, role="user", content=content)

    def append_assistant(
        self,
        session_id: uuid.UUID,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        self._repo.add_message(
            session_id, role="assistant", content=content, metadata=metadata
        )


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
