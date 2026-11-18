"""Conversation memory — the agent remembers the whole chat.

Covers:
- chat writes are committed by the memory layer itself (the streaming
  /ask path runs after FastAPI has closed the request DB session);
- recent turns verbatim + full user-message outline + rolling summary;
- the summary is updated lazily, in batches, and failures don't break a turn;
- client-supplied history seeds an empty server session, in order, and is
  ignored once the server has the conversation;
- the agent receives the memory message and QAService wires it end to end.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

from app.rag.chat_agent import ChatAgent
from app.rag.conversation_memory import build_memory_message
from app.rag.legal_memory import ConversationMemory, LegalMemory
from app.rag.prompt_builder import MemoryTurn
from tests.unit.test_chat_agent import _drain, _FakeClient, _text_round


@dataclass
class _Msg:
    session_id: uuid.UUID
    role: str
    content: str
    created_at: datetime
    metadata_json: dict[str, Any] | None = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)


@dataclass
class _Sess:
    id: uuid.UUID
    organization_id: uuid.UUID | None
    metadata_json: dict[str, Any] | None = None


class InMemoryChatRepo:
    """Mirrors ChatRepository, tracking commits."""

    def __init__(self) -> None:
        self.sessions: dict[uuid.UUID, _Sess] = {}
        self.msgs: list[_Msg] = []
        self.commits = 0
        self._clock = datetime.now(UTC)  # like the DB's now()

    def create_session(self, title=None, *, organization_id=None, user_id=None):  # type: ignore[no-untyped-def]
        s = _Sess(id=uuid.uuid4(), organization_id=organization_id)
        self.sessions[s.id] = s
        return s

    def get_session(self, session_id, *, organization_id=None):  # type: ignore[no-untyped-def]
        s = self.sessions.get(session_id)
        return s if s and s.organization_id == organization_id else None

    def commit(self) -> None:
        self.commits += 1

    def add_message(self, session_id, role, content, metadata=None, created_at=None):  # type: ignore[no-untyped-def]
        self._clock += timedelta(seconds=1)
        m = _Msg(session_id, role, content, created_at or self._clock, metadata)
        self.msgs.append(m)
        return m

    def _ordered(self, sid: uuid.UUID) -> list[_Msg]:
        return sorted((m for m in self.msgs if m.session_id == sid), key=lambda m: m.created_at)

    def get_recent_messages(self, session_id, limit):  # type: ignore[no-untyped-def]
        return self._ordered(session_id)[-limit:] if limit else []

    def get_user_messages(self, session_id, limit):  # type: ignore[no-untyped-def]
        return [m.content for m in self._ordered(session_id) if m.role == "user"][:limit]

    def count_messages(self, session_id):  # type: ignore[no-untyped-def]
        return len(self._ordered(session_id))

    def get_messages_range(self, session_id, offset, limit):  # type: ignore[no-untyped-def]
        return self._ordered(session_id)[offset : offset + limit]

    def get_session_meta(self, session_id):  # type: ignore[no-untyped-def]
        return dict(self.sessions[session_id].metadata_json or {})

    def set_session_meta(self, session_id, meta):  # type: ignore[no-untyped-def]
        self.sessions[session_id].metadata_json = meta


def _chat(mem: LegalMemory, sid: uuid.UUID, n: int) -> None:
    for i in range(1, n + 1):
        mem.append_user(sid, f"question {i}")
        mem.append_assistant(sid, f"answer {i}")


def test_writes_are_committed_immediately() -> None:
    repo = InMemoryChatRepo()
    mem = LegalMemory(repo)  # type: ignore[arg-type]
    sid = mem.get_or_create_session(None)
    assert repo.commits == 1
    mem.append_user(sid, "hi")
    mem.append_assistant(sid, "hello")
    assert repo.commits == 3
    # Resuming finds the session (it was committed), no new one is created.
    assert mem.get_or_create_session(str(sid)) == sid
    assert len(repo.sessions) == 1


def test_short_chat_has_full_outline_and_no_summary() -> None:
    repo = InMemoryChatRepo()
    mem = LegalMemory(repo)  # type: ignore[arg-type]
    sid = mem.get_or_create_session(None)
    _chat(mem, sid, 3)
    summarizer = MagicMock(return_value="S")
    conv = mem.load_conversation(sid, window_turns=20, max_tokens=12000, summarizer=summarizer)
    assert len(conv.recent) == 6
    assert conv.user_messages == ["question 1", "question 2", "question 3"]
    assert conv.earlier_count == 0
    summarizer.assert_not_called()


def test_long_chat_summarises_aged_turns_in_batches() -> None:
    repo = InMemoryChatRepo()
    mem = LegalMemory(repo)  # type: ignore[arg-type]
    sid = mem.get_or_create_session(None)
    _chat(mem, sid, 30)  # 60 messages
    seen: list[list[str]] = []

    def summarizer(prev: str | None, turns: list[MemoryTurn]) -> str:
        seen.append([t.content for t in turns])
        return f"{prev or ''}+{len(turns)}"

    conv = mem.load_conversation(sid, window_turns=10, max_tokens=12000, summarizer=summarizer)
    # 20 recent verbatim, 40 older folded into the summary in one call.
    assert len(conv.recent) == 20
    assert conv.earlier_count == 40
    assert conv.summary == "+40"
    assert seen[0][0] == "question 1"
    # The outline still has the very first question.
    assert conv.user_messages[0] == "question 1"
    assert len(conv.user_messages) == 30

    # Two more turns: 4 more messages age out -> one incremental update.
    _chat(mem, sid, 2)
    conv = mem.load_conversation(sid, window_turns=10, max_tokens=12000, summarizer=summarizer)
    assert conv.summary == "+40+4"
    assert len(seen[-1]) == 4
    # One more turn: only 2 aged -> below batch, summary unchanged.
    _chat(mem, sid, 1)
    conv = mem.load_conversation(sid, window_turns=10, max_tokens=12000, summarizer=summarizer)
    assert conv.summary == "+40+4"


def test_summary_failure_does_not_break_the_turn() -> None:
    repo = InMemoryChatRepo()
    mem = LegalMemory(repo)  # type: ignore[arg-type]
    sid = mem.get_or_create_session(None)
    _chat(mem, sid, 20)
    boom = MagicMock(side_effect=RuntimeError("llm down"))
    conv = mem.load_conversation(sid, window_turns=5, max_tokens=12000, summarizer=boom)
    assert conv.summary is None
    assert len(conv.recent) == 10
    assert conv.user_messages[0] == "question 1"


def test_import_history_seeds_empty_session_in_order_only_once() -> None:
    repo = InMemoryChatRepo()
    mem = LegalMemory(repo)  # type: ignore[arg-type]
    sid = mem.get_or_create_session(None)
    turns = [
        MemoryTurn("user", "first q"),
        MemoryTurn("assistant", "first a"),
        MemoryTurn("user", "second q"),
    ]
    mem.import_history(sid, turns)
    mem.append_user(sid, "third q")
    assert [m.content for m in repo._ordered(sid)] == ["first q", "first a", "second q", "third q"]
    mem.import_history(sid, [MemoryTurn("user", "should be ignored")])
    assert repo.count_messages(sid) == 4


def test_memory_message_renders_outline_and_summary() -> None:
    assert build_memory_message(ConversationMemory(recent=[])) is None
    msg = build_memory_message(
        ConversationMemory(
            recent=[],
            summary="User acts for Kofi Mensah in a land dispute.",
            user_messages=["What is a caveat?", "x " * 400],
            earlier_count=12,
        )
    )
    assert msg is not None
    assert "1. What is a caveat?" in msg
    assert "Kofi Mensah" in msg
    assert "12 earlier message(s)" in msg
    assert "…" in msg  # long message clipped


def test_agent_receives_memory_message_before_history() -> None:
    client = _FakeClient([_text_round("Your first question was about caveats.")])
    agent = ChatAgent(client=client, model="gpt-test")
    _drain(
        agent,
        "what was my first question?",
        search_fn=MagicMock(),
        conversation_memory="# This conversation so far\n1. What is a caveat?",
    )
    msgs = client.requests[0]["messages"]
    assert msgs[2]["role"] == "system"
    assert "What is a caveat?" in msgs[2]["content"]
    assert msgs[-1] == {"role": "user", "content": "what was my first question?"}


def test_qa_service_uses_client_history_when_server_has_none() -> None:
    from tests.unit.test_chat_agent import _service

    client = _FakeClient([_text_round("You first asked about caveats.")])
    svc, mem, _ = _service(client, [])
    events = list(
        svc.ask_stream(
            "what was my first question?",
            client_history=[
                MemoryTurn("user", "What is a caveat?"),
                MemoryTurn("assistant", "A caveat is…"),
            ],
        )
    )
    assert events[-1].event == "completed"
    req = client.requests[0]["messages"]
    joined = "\n".join(str(m.get("content")) for m in req)
    assert "What is a caveat?" in joined
    # Seeded into server memory, then the new question appended.
    assert [c for _, c, _ in mem.messages][:2] == ["What is a caveat?", "A caveat is…"]
