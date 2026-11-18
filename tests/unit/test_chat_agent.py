"""ChatAgent — tool-calling conversational loop behind Ask Legalite.

Drives a fake *streaming* OpenAI client that scripts each round's output
(text deltas or tool-call deltas) and asserts:

- Non-legal questions are answered directly, no search, mode=general /
  conversational.
- Legal questions trigger ``search_legal_library``; returned passages are
  numbered and the final answer's [n] markers are validated.
- Hallucinated citation numbers are stripped and never surface as sources.
- ``force_search`` pins the first round's tool_choice to the search tool.
- The tool loop is bounded by ``agent_max_tool_rounds``.
- A passage returned by two searches keeps one citation number.
- QAService wires the agent end-to-end and keeps the AskResponse contract.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from app.rag.chat_agent import (
    AgentResult,
    AgentStatus,
    ChatAgent,
    ChatTurn,
    clean_history_text,
    validate_citation_markers,
)
from app.rag.retriever import RetrievedPassage

# ---- fake streaming client --------------------------------------------------


@dataclass
class _Fn:
    name: str | None = None
    arguments: str | None = None


@dataclass
class _ToolCallDelta:
    index: int
    id: str | None = None
    function: _Fn | None = None


@dataclass
class _Delta:
    content: str | None = None
    tool_calls: list[_ToolCallDelta] | None = None


@dataclass
class _Choice:
    delta: _Delta


@dataclass
class _Usage:
    prompt_tokens: int = 100
    completion_tokens: int = 40


@dataclass
class _Chunk:
    choices: list[_Choice] = field(default_factory=list)
    usage: _Usage | None = None


def _text_round(text: str, size: int = 6) -> list[_Chunk]:
    out = [
        _Chunk(choices=[_Choice(_Delta(content=text[i : i + size]))])
        for i in range(0, len(text), size)
    ]
    out.append(_Chunk(usage=_Usage()))
    return out


def _tool_round(*queries: str, name: str = "search_legal_library") -> list[_Chunk]:
    out: list[_Chunk] = []
    for idx, q in enumerate(queries):
        args = json.dumps({"query": q}) if name == "search_legal_library" else "{}"
        half = len(args) // 2
        out.append(
            _Chunk(
                choices=[
                    _Choice(
                        _Delta(
                            tool_calls=[_ToolCallDelta(idx, f"call_{idx}", _Fn(name, args[:half]))]
                        )
                    )
                ]
            )
        )
        out.append(
            _Chunk(
                choices=[
                    _Choice(_Delta(tool_calls=[_ToolCallDelta(idx, None, _Fn(None, args[half:]))]))
                ]
            )
        )
    out.append(_Chunk(usage=_Usage()))
    return out


class _FakeClient:
    """Returns one scripted round per ``create`` call and records requests."""

    def __init__(self, rounds: list[list[_Chunk]]) -> None:
        self._rounds = list(rounds)
        self.requests: list[dict[str, Any]] = []
        self.chat = MagicMock()
        self.chat.completions.create.side_effect = self._create

    def _create(self, **kwargs: Any) -> list[_Chunk]:
        # Snapshot messages: the agent mutates the list between rounds.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self._rounds:
            return _text_round("fallback answer")
        return self._rounds.pop(0)


def _passage(n: int, **overrides: object) -> RetrievedPassage:
    base: dict[str, object] = {
        "score": 0.8,
        "original_score": 0.8,
        "rerank_score": 0.7,
        "content": f"Section {n} text about notice to quit.",
        "law": "Rent Act, 1963 (Act 220)",
        "article": None,
        "section": str(n),
        "chapter": None,
        "clause": None,
        "subsection": None,
        "page_start": n,
        "page_end": n,
        "vector_id": f"vid-{n}",
        "chunk_id": f"chunk-{n}",
        "document_id": "doc-1",
        "law_type": "act",
        "document_origin": "act",
        "visibility_scope": "global",
    }
    base.update(overrides)
    return RetrievedPassage(**base)  # type: ignore[arg-type]


def _drain(agent: ChatAgent, question: str, **kwargs: Any) -> tuple[list[Any], AgentResult]:
    items = list(agent.run_stream(question, **kwargs))
    result = items[-1]
    assert isinstance(result, AgentResult)
    return items, result


# ---- tests ------------------------------------------------------------------


def test_general_question_answered_without_search() -> None:
    client = _FakeClient([_text_round("Photosynthesis converts light into chemical energy.")])
    agent = ChatAgent(client=client, model="gpt-test")
    search = MagicMock(return_value=[])

    items, result = _drain(agent, "How does photosynthesis work?", search_fn=search)

    search.assert_not_called()
    text = "".join(i for i in items if isinstance(i, str))
    assert text == "Photosynthesis converts light into chemical energy."
    assert result.answer == text
    assert result.mode == "general"
    assert result.cited_ids == []
    assert result.rounds == 1
    # Tools are offered; the model simply didn't use them.
    assert client.requests[0]["tools"]
    assert client.requests[0]["tool_choice"] == "auto"


def test_conversational_turn_uses_fast_model_and_mode() -> None:
    client = _FakeClient([_text_round("Hello! How can I help today?")])
    agent = ChatAgent(client=client, model="big-model", fast_model="small-model")

    _, result = _drain(agent, "hi", search_fn=MagicMock(), conversational=True, force_search=True)

    assert result.mode == "conversational"
    assert result.confidence == "high"
    assert client.requests[0]["model"] == "small-model"
    # conversational overrides force_search
    assert client.requests[0]["tool_choice"] == "auto"


def test_legal_question_searches_and_cites() -> None:
    client = _FakeClient(
        [
            _tool_round("notice to quit Rent Act"),
            _text_round("A landlord must give notice [1], in writing [2]."),
        ]
    )
    agent = ChatAgent(client=client, model="gpt-test")
    search = MagicMock(return_value=[_passage(17), _passage(18)])

    items, result = _drain(agent, "What notice must a landlord give?", search_fn=search)

    search.assert_called_once_with("notice to quit Rent Act")
    statuses = [i for i in items if isinstance(i, AgentStatus)]
    assert [s.kind for s in statuses] == ["searching", "sources", "answering"]
    assert statuses[0].query == "notice to quit Rent Act"
    assert statuses[1].count == 2

    assert result.mode == "grounded"
    assert result.cited_ids == [1, 2]
    assert [p.section for p in result.cited_passages] == ["17", "18"]
    assert result.confidence == "high"
    assert result.rounds == 2

    # Round 2 carries the tool result with numbered passages.
    tool_msgs = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "[1] Rent Act" in tool_msgs[0]["content"]
    assert "[2] Rent Act" in tool_msgs[0]["content"]


def test_hallucinated_citations_are_stripped() -> None:
    client = _FakeClient(
        [_tool_round("bail"), _text_round("Bail is a right [1] subject to limits [7][1, 9].")]
    )
    agent = ChatAgent(client=client, model="gpt-test")

    _, result = _drain(
        agent, "Explain bail under the Constitution", search_fn=lambda q: [_passage(1)]
    )

    assert result.cited_ids == [1]
    assert "[7]" not in result.answer and "[9]" not in result.answer
    assert result.answer == "Bail is a right [1] subject to limits [1]."


def test_no_results_yields_general_mode_low_confidence() -> None:
    client = _FakeClient(
        [
            _tool_round("maritime salvage"),
            _text_round("I couldn't find this in the library; generally…"),
        ]
    )
    agent = ChatAgent(client=client, model="gpt-test")

    _, result = _drain(agent, "Ghana maritime salvage law?", search_fn=lambda q: [])

    assert result.mode == "general"
    assert result.confidence == "low"
    tool_msg = [m for m in client.requests[1]["messages"] if m["role"] == "tool"][0]
    assert "No passages" in tool_msg["content"]


def test_force_search_pins_first_round_tool_choice() -> None:
    client = _FakeClient([_tool_round("x"), _text_round("done")])
    agent = ChatAgent(client=client, model="gpt-test")

    _drain(agent, "Article 19 fair trial", search_fn=lambda q: [], force_search=True)

    assert client.requests[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_legal_library"},
    }
    assert client.requests[1]["tool_choice"] == "auto"


def test_tool_loop_is_bounded(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from app.core import config

    settings = config.get_settings()
    monkeypatch.setattr(settings, "agent_max_tool_rounds", 2)
    # Model keeps asking for tools forever.
    client = _FakeClient([_tool_round("a"), _tool_round("b"), _text_round("final")])
    agent = ChatAgent(client=client, model="gpt-test")

    _, result = _drain(agent, "loop", search_fn=lambda q: [])

    assert result.rounds == 3
    assert "tools" not in client.requests[2]  # last round must answer
    assert result.answer == "final"


def test_duplicate_passages_keep_one_number() -> None:
    client = _FakeClient([_tool_round("q1", "q2"), _text_round("See [1] and [2].")])
    agent = ChatAgent(client=client, model="gpt-test")
    calls: list[str] = []

    def search(q: str) -> list[RetrievedPassage]:
        calls.append(q)
        return [_passage(1), _passage(2)] if q == "q1" else [_passage(2), _passage(3)]

    _, result = _drain(agent, "two issues", search_fn=search)

    assert calls == ["q1", "q2"]  # parallel tool calls both executed
    assert len(result.passages) == 3  # passage 2 deduped
    tool_msgs = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert "[2] Rent Act" in tool_msgs[1]["content"]
    assert "[3] Rent Act" in tool_msgs[1]["content"]


def test_history_and_context_reach_the_model() -> None:
    client = _FakeClient([_text_round("Yes, as discussed.")])
    agent = ChatAgent(client=client, model="gpt-test")

    _drain(
        agent,
        "and what about that?",
        search_fn=lambda q: [],
        history=[ChatTurn("user", "Can I be dismissed?"), ChatTurn("assistant", "It depends.")],
        page_context="Employment contract clause 4",
    )

    msgs = client.requests[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert "Employment contract clause 4" in msgs[1]["content"]
    assert [m["role"] for m in msgs[2:]] == ["user", "assistant", "user"]
    assert msgs[-1]["content"] == "and what about that?"


def test_search_failure_does_not_crash_turn() -> None:
    client = _FakeClient([_tool_round("x"), _text_round("Answer anyway.")])
    agent = ChatAgent(client=client, model="gpt-test")

    def boom(q: str) -> list[RetrievedPassage]:
        raise RuntimeError("qdrant down")

    _, result = _drain(agent, "Section 5 Labour Act", search_fn=boom)
    assert result.answer == "Answer anyway."
    assert result.mode == "general"


def test_reasoning_models_omit_temperature() -> None:
    client = _FakeClient([_text_round("ok")])
    _drain(ChatAgent(client=client, model="o4-mini"), "hi", search_fn=lambda q: [])
    assert "temperature" not in client.requests[0]
    assert client.requests[0]["max_completion_tokens"] > 0


# ---- pure helpers -------------------------------------------------------------


def test_validate_citation_markers_forms() -> None:
    text, ids = validate_citation_markers("A [1-3]. B [2, 5]. C [x]. D [link](http://e.com) [4]", 3)
    assert ids == [1, 2, 3]
    assert "[1][2][3]" in text
    assert "[2]" in text and "[5]" not in text
    assert "[link](http://e.com)" in text
    assert "[4]" not in text


def test_clean_history_strips_old_citations_and_legacy_tail() -> None:
    legacy = (
        "Notice is required [1].\n\nCitations:\n- [1] Rent Act | s.17\n\n"
        "Confidence note: single source\n\n"
        "This information is for educational purposes and not legal advice."
    )
    assert clean_history_text(legacy, 1000) == "Notice is required."
    assert clean_history_text("x" * 50 + " tail", 20).endswith("…")


# ---- QAService integration ------------------------------------------------------


class _FakeMemory:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, dict | None]] = []
        self.sid = uuid.uuid4()

    def get_or_create_session(self, session_id, *, tenant=None):  # type: ignore[no-untyped-def]
        return self.sid

    def load_conversation(self, sid, *, window_turns, max_tokens, summarizer=None):  # type: ignore[no-untyped-def]
        from app.rag.legal_memory import ConversationMemory
        from app.rag.prompt_builder import MemoryTurn

        return ConversationMemory(
            recent=[MemoryTurn(role=r, content=c) for r, c, _ in self.messages],
            user_messages=[c for r, c, _ in self.messages if r == "user"],
        )

    def import_history(self, sid, turns):  # type: ignore[no-untyped-def]
        if not self.messages:
            self.messages.extend((t.role, t.content, None) for t in turns)

    def append_user(self, sid, content):  # type: ignore[no-untyped-def]
        self.messages.append(("user", content, None))

    def append_assistant(self, sid, content, metadata=None):  # type: ignore[no-untyped-def]
        self.messages.append(("assistant", content, metadata))
        return uuid.uuid4()


def _service(client: _FakeClient, passages: list[RetrievedPassage]):  # type: ignore[no-untyped-def]
    from app.rag.multi_document_reasoner import CrossDocResult
    from app.rag.reranker import BaseReranker
    from app.services.qa_service import QAService

    svc = QAService(
        MagicMock(),
        embeddings=MagicMock(),
        vector_store=MagicMock(),
        reranker=MagicMock(spec=BaseReranker),
        rewriter=MagicMock(),
        generator=MagicMock(),
        agent=ChatAgent(client=client, model="gpt-test"),
    )
    mem = _FakeMemory()
    svc._memory = mem  # type: ignore[assignment]
    reasoner = MagicMock()
    reasoner.gather.return_value = CrossDocResult(passages=passages, bucket_counts={})
    svc._reasoner = reasoner  # type: ignore[assignment]
    return svc, mem, reasoner


def test_qa_service_agent_stream_contract() -> None:
    client = _FakeClient([_tool_round("notice Rent Act"), _text_round("Notice is required [1].")])
    svc, mem, reasoner = _service(client, [_passage(17)])

    events = list(svc.ask_stream("What does section 17 of the Rent Act say about notice?"))
    names = [e.event for e in events]

    assert names[0] == "retrieval_started"
    assert "status" in names and "sources_found" in names and "answer_delta" in names
    assert names[-2:] == ["citations", "completed"]
    done = events[-1].data
    assert done["answer_mode"] == "grounded"
    assert done["searches"] == ["notice Rent Act"]
    assert done["structured_answer"]["direct_answer"] == "Notice is required [1]."
    assert done["structured_answer"]["applicable_law"][0]["section"] == "17"
    assert done["citations"][0]["section"] == "17"
    assert done["message_id"]
    # Force-search: statute question with legal vocabulary.
    assert client.requests[0]["tool_choice"]["function"]["name"] == "search_legal_library"
    reasoner.gather.assert_called_once()
    # Memory got the user turn and the clean assistant answer.
    assert [m[0] for m in mem.messages] == ["user", "assistant"]
    assert mem.messages[1][2]["answer_mode"] == "grounded"


def test_qa_service_agent_answers_off_topic_question() -> None:
    client = _FakeClient([_text_round("The capital of Australia is Canberra.")])
    svc, _, reasoner = _service(client, [])

    resp = svc.ask("What is the capital of Australia?")

    assert resp.answer == "The capital of Australia is Canberra."
    assert resp.answer_mode == "general"
    assert resp.citations == []
    reasoner.gather.assert_not_called()


def test_qa_service_agent_error_falls_back_for_greeting() -> None:
    client = _FakeClient([])
    client.chat.completions.create.side_effect = RuntimeError("quota")
    svc, _, _ = _service(client, [])

    events = list(svc.ask_stream("hello"))
    assert events[-1].event == "completed"
    assert events[-1].data["answer_mode"] == "conversational"

    events = list(svc.ask_stream("Explain the doctrine of laches in detail please"))
    assert events[-1].event == "error"
