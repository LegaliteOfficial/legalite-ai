"""AnswerGenerator.generate_stream — token-streaming path.

Mirrors the non-streaming coverage in test_answer_generator_structured.py
but drives a mocked *streaming* OpenAI client (chunked deltas instead of
one blocking completion) and asserts:

- Deltas emitted match the ``direct_answer`` text character-for-character.
- The terminal GeneratedAnswer is identical to what :meth:`generate` would
  produce from the same full JSON (same safety checks, same helper).
- The "stream, then retract" contract: when the post-hoc grounding check
  in ``assess_final`` refuses an answer whose direct_answer streamed fine
  token-by-token, the terminal item still reports ``refused=True`` with
  the canonical refusal text — different from what the deltas showed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.rag.answer_generator import AnswerGenerator, GeneratedAnswer
from app.rag.prompt_builder import REFUSAL_TEXT
from app.rag.retriever import RetrievedPassage


def _passage(score: float = 0.8, **overrides: object) -> RetrievedPassage:
    base = dict(
        score=score,
        original_score=score,
        content="Section 17 says a landlord must give notice.",
        law="Rent Act",
        article=None,
        section="17",
        chapter=None,
        clause=None,
        subsection=None,
        page_start=12,
        page_end=12,
        vector_id="vid",
        law_type="act",
        document_origin="act",
        visibility_scope="global",
    )
    base.update(overrides)
    return RetrievedPassage(**base)  # type: ignore[arg-type]


# ---- mock streaming OpenAI client ------------------------------------------


@dataclass
class _MockDelta:
    content: str | None


@dataclass
class _MockStreamChoice:
    delta: _MockDelta


@dataclass
class _MockUsage:
    prompt_tokens: int = 50
    completion_tokens: int = 80


@dataclass
class _MockChunk:
    choices: list[_MockStreamChoice] = field(default_factory=list)
    usage: _MockUsage | None = None


def _chunks_for(payload: str, chunk_size: int = 7) -> list[_MockChunk]:
    """Split ``payload`` into content chunks, ending with a usage-only
    chunk — mirrors ``stream_options={"include_usage": True}`` wire
    behavior (the final chunk carries usage and an empty choices list)."""
    out: list[_MockChunk] = []
    for i in range(0, len(payload), chunk_size):
        piece = payload[i : i + chunk_size]
        out.append(_MockChunk(choices=[_MockStreamChoice(delta=_MockDelta(content=piece))]))
    out.append(_MockChunk(choices=[], usage=_MockUsage()))
    return out


class _MockStreamingCompletions:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> list[_MockChunk]:
        self.last_kwargs = kwargs
        assert kwargs.get("stream") is True
        return _chunks_for(self.payload)


class _MockChat:
    def __init__(self, completions: _MockStreamingCompletions) -> None:
        self.completions = completions


class _MockClient:
    def __init__(self, payload: str) -> None:
        self.chat = _MockChat(_MockStreamingCompletions(payload))


def _make_generator(payload: str) -> AnswerGenerator:
    return AnswerGenerator(client=_MockClient(payload))  # type: ignore[arg-type]


def _run_stream(gen: AnswerGenerator, question: str, passages: list[RetrievedPassage]) -> tuple[str, GeneratedAnswer]:
    deltas: list[str] = []
    result: GeneratedAnswer | None = None
    for item in gen.generate_stream(question, passages):
        if isinstance(item, str):
            deltas.append(item)
        else:
            result = item
    assert result is not None
    return "".join(deltas), result


# ---- happy path --------------------------------------------------------


_GOOD_JSON = json.dumps(
    {
        "direct_answer": "Yes, a landlord must give notice before eviction.",
        "applicable_law": [
            {
                "citation_id": 1,
                "law_name": "Rent Act, 1963 (Act 220)",
                "section": "17",
                "summary": "Notice required before eviction",
            }
        ],
        "relevant_public_cases": [],
        "firm_similar_cases": [],
        "legal_reasoning": "Section 17 of the Rent Act requires notice [1].",
        "confidence_assessment": "Single source; high confidence.",
        "citations": [1],
    }
)


def test_deltas_match_direct_answer_and_final_result_is_not_refused() -> None:
    gen = _make_generator(_GOOD_JSON)
    streamed_text, result = _run_stream(
        gen, "Can a landlord evict without notice?", [_passage()]
    )
    assert streamed_text == "Yes, a landlord must give notice before eviction."
    assert not result.refused
    assert result.structured.direct_answer == streamed_text
    assert result.structured.citations == [1]


def test_streaming_call_uses_json_response_format_and_stream_flag() -> None:
    gen = _make_generator(_GOOD_JSON)
    _run_stream(gen, "q", [_passage()])
    kwargs = gen._client.chat.completions.last_kwargs  # type: ignore[attr-defined]
    assert kwargs is not None
    assert kwargs.get("response_format") == {"type": "json_object"}
    assert kwargs.get("stream") is True


def test_usage_is_captured_from_final_chunk() -> None:
    gen = _make_generator(_GOOD_JSON)
    _, result = _run_stream(gen, "q", [_passage()])
    assert result.tokens_in == 50
    assert result.tokens_out == 80


# ---- retraction: post-hoc grounding refusal after a clean stream -------


def test_ungrounded_answer_streams_fine_but_terminal_result_is_refused() -> None:
    """The 'stream, then retract' contract: direct_answer has no [n]
    markers anywhere in the synthesized legacy answer (no legal_reasoning,
    no citation rows), so assess_final refuses it AFTER the full text
    already streamed cleanly to the caller. The terminal item must report
    refused=True with the canonical refusal text — the caller is
    responsible for replacing whatever it rendered from the deltas.
    """
    payload = json.dumps(
        {
            "direct_answer": "Notice is generally required before eviction.",
            "applicable_law": [],
            "relevant_public_cases": [],
            "firm_similar_cases": [],
            "legal_reasoning": "",
            "confidence_assessment": "",
            "citations": [],
        }
    )
    gen = _make_generator(payload)
    streamed_text, result = _run_stream(gen, "q", [_passage()])

    # The deltas streamed the model's original (ungrounded) text...
    assert streamed_text == "Notice is generally required before eviction."
    # ...but the terminal result overrides it with a refusal.
    assert result.refused
    assert result.structured.direct_answer == REFUSAL_TEXT
    assert REFUSAL_TEXT in result.answer
    assert result.structured.direct_answer != streamed_text


# ---- refusal paths mirror the non-streaming generator -------------------


def test_model_refusal_sentence_yields_empty_structured() -> None:
    payload = json.dumps(
        {
            "direct_answer": REFUSAL_TEXT,
            "applicable_law": [],
            "relevant_public_cases": [],
            "firm_similar_cases": [],
            "legal_reasoning": "No passage addressed this.",
            "confidence_assessment": "Insufficient evidence.",
            "citations": [],
        }
    )
    gen = _make_generator(payload)
    streamed_text, result = _run_stream(gen, "q", [_passage()])
    assert streamed_text == REFUSAL_TEXT
    assert result.refused
    assert result.structured.applicable_law == []


def test_malformed_json_triggers_refusal_without_crashing() -> None:
    gen = _make_generator("this is not json at all")
    streamed_text, result = _run_stream(gen, "q", [_passage()])
    # The key "direct_answer" never appears, so nothing streams.
    assert streamed_text == ""
    assert result.refused
    assert REFUSAL_TEXT in result.answer


def test_no_passages_short_circuits_before_any_llm_call() -> None:
    gen = _make_generator(_GOOD_JSON)  # would succeed if reached
    items = list(gen.generate_stream("q", []))
    assert len(items) == 1
    assert isinstance(items[0], GeneratedAnswer)
    assert items[0].refused
    assert gen._client.chat.completions.last_kwargs is None  # type: ignore[attr-defined]


def test_llm_exception_mid_stream_falls_back_to_refusal() -> None:
    class _ExplodingCompletions:
        def create(self, **_kwargs: Any) -> Any:
            raise RuntimeError("quota exhausted")

    class _ExplodingChat:
        completions = _ExplodingCompletions()

    class _ExplodingClient:
        chat = _ExplodingChat()

    gen = AnswerGenerator(client=_ExplodingClient())  # type: ignore[arg-type]
    items = list(gen.generate_stream("q", [_passage()]))
    assert len(items) == 1
    assert isinstance(items[0], GeneratedAnswer)
    assert items[0].refused
    assert REFUSAL_TEXT in items[0].answer
