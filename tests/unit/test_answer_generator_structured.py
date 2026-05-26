"""AnswerGenerator — structured JSON parsing, legacy synthesis, refusal paths.

We mock the OpenAI client so these tests stay deterministic and offline.
Behavior covered:

- Good JSON → StructuredAnswer + legacy answer synthesized.
- Malformed JSON → refusal (legacy answer = refusal sentence).
- LLM emits the refusal sentence → structured arrays empty.
- Empty passages → no LLM call at all (pre-LLM gate).
- Pre-LLM call ALWAYS uses response_format=json_object (M1 contract).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.rag.answer_generator import (
    AnswerGenerator,
    _empty_refusal,
    _parse_structured,
    _synthesize_legacy_answer,
)
from app.rag.prompt_builder import DISCLAIMER, REFUSAL_TEXT
from app.rag.retriever import RetrievedPassage
from app.schemas.qa import StructuredAnswer, StructuredCitation


def _passage(score: float = 0.8, **overrides: object) -> RetrievedPassage:
    base = dict(
        score=score,
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


# ---- mock OpenAI client ----------------------------------------------------


@dataclass
class _MockUsage:
    prompt_tokens: int = 50
    completion_tokens: int = 80


@dataclass
class _MockMessage:
    content: str


@dataclass
class _MockChoice:
    message: _MockMessage


@dataclass
class _MockCompletion:
    choices: list[_MockChoice]
    usage: _MockUsage


class _MockCompletions:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _MockCompletion:
        self.last_kwargs = kwargs
        return _MockCompletion(
            choices=[_MockChoice(message=_MockMessage(content=self.payload))],
            usage=_MockUsage(),
        )


class _MockChat:
    def __init__(self, completions: _MockCompletions) -> None:
        self.completions = completions


class _MockClient:
    def __init__(self, payload: str) -> None:
        self.chat = _MockChat(_MockCompletions(payload))


def _make_generator(payload: str) -> AnswerGenerator:
    client = _MockClient(payload)
    return AnswerGenerator(client=client)  # type: ignore[arg-type]


# ---- happy path ------------------------------------------------------------


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


def test_structured_answer_returned_on_good_json() -> None:
    gen = _make_generator(_GOOD_JSON)
    result = gen.generate("Can a landlord evict without notice?", [_passage()])

    assert not result.refused
    assert result.structured.direct_answer.startswith("Yes,")
    assert result.structured.citations == [1]
    assert len(result.structured.applicable_law) == 1
    assert result.structured.applicable_law[0].section == "17"
    # Legacy answer synthesized from the structured payload.
    assert "Rent Act" in result.answer
    assert DISCLAIMER in result.answer


def test_llm_call_uses_json_response_format() -> None:
    """M1 contract — every LLM call must request JSON output."""
    gen = _make_generator(_GOOD_JSON)
    gen.generate("q", [_passage()])
    kwargs = gen._client.chat.completions.last_kwargs  # type: ignore[attr-defined]
    assert kwargs is not None
    assert kwargs.get("response_format") == {"type": "json_object"}


# ---- defensive paths -------------------------------------------------------


def test_malformed_json_triggers_refusal() -> None:
    gen = _make_generator("this is not json at all")
    result = gen.generate("anything", [_passage()])
    assert result.refused
    assert REFUSAL_TEXT in result.answer
    assert DISCLAIMER in result.answer
    assert result.structured.direct_answer == REFUSAL_TEXT
    assert result.structured.applicable_law == []
    assert result.structured.relevant_public_cases == []
    assert result.structured.firm_similar_cases == []


def test_json_without_direct_answer_triggers_refusal() -> None:
    gen = _make_generator('{"applicable_law": [], "citations": []}')
    result = gen.generate("anything", [_passage()])
    assert result.refused
    assert REFUSAL_TEXT in result.answer


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
    result = gen.generate("anything", [_passage()])
    assert result.refused
    assert result.structured.direct_answer == REFUSAL_TEXT
    assert result.structured.applicable_law == []
    assert result.confidence == "low"


def test_no_passages_short_circuits_before_llm_call() -> None:
    """Pre-LLM gate must refuse without invoking the model at all."""
    gen = _make_generator(_GOOD_JSON)  # would succeed if reached
    result = gen.generate("anything", [])
    assert result.refused
    # The mock completions object must NOT have been called.
    assert gen._client.chat.completions.last_kwargs is None  # type: ignore[attr-defined]


def test_weak_retrieval_short_circuits_before_llm_call() -> None:
    gen = _make_generator(_GOOD_JSON)
    weak = _passage(score=0.05)
    result = gen.generate("q", [weak])
    assert result.refused
    assert gen._client.chat.completions.last_kwargs is None  # type: ignore[attr-defined]


def test_llm_exception_falls_back_to_refusal() -> None:
    """If the OpenAI SDK raises (quota, timeout, anything), we refuse cleanly."""

    class _ExplodingChat:
        class completions:  # noqa: N801
            @staticmethod
            def create(**_kwargs: Any) -> Any:
                raise RuntimeError("quota exhausted")

    class _ExplodingClient:
        chat = _ExplodingChat()

    gen = AnswerGenerator(client=_ExplodingClient())  # type: ignore[arg-type]
    result = gen.generate("q", [_passage()])
    assert result.refused
    assert REFUSAL_TEXT in result.answer


# ---- _parse_structured: edge cases ----------------------------------------


def test_parse_structured_coerces_int_strings_in_citations() -> None:
    raw = json.dumps(
        {
            "direct_answer": "ok",
            "citations": ["1", "2", "junk", 3],
        }
    )
    out, err = _parse_structured(raw)
    assert err is None
    assert out is not None
    assert out.citations == [1, 2, 3]


def test_parse_structured_drops_rows_without_citation_id() -> None:
    raw = json.dumps(
        {
            "direct_answer": "ok",
            "applicable_law": [
                {"citation_id": 1, "law_name": "Act"},
                {"law_name": "no id"},  # dropped
                {"citation_id": "not-int", "law_name": "bad"},  # dropped
            ],
        }
    )
    out, _ = _parse_structured(raw)
    assert out is not None
    assert len(out.applicable_law) == 1


# ---- _synthesize_legacy_answer --------------------------------------------


def test_legacy_answer_includes_direct_answer_and_disclaimer() -> None:
    s = StructuredAnswer(
        direct_answer="The answer is yes.",
        legal_reasoning="Because of [1] and [2].",
        applicable_law=[
            StructuredCitation(citation_id=1, law_name="Act X", section="3"),
        ],
        confidence_assessment="Solid.",
        citations=[1, 2],
    )
    legacy = _synthesize_legacy_answer(s)
    assert "The answer is yes." in legacy
    assert "Because of [1] and [2]." in legacy
    assert "Act X" in legacy
    assert "s.3" in legacy
    assert "Confidence note: Solid." in legacy
    assert legacy.endswith(DISCLAIMER)


def test_empty_refusal_helper_yields_canonical_empty_payload() -> None:
    empty = _empty_refusal(REFUSAL_TEXT)
    assert empty.direct_answer == REFUSAL_TEXT
    assert empty.applicable_law == []
    assert empty.relevant_public_cases == []
    assert empty.firm_similar_cases == []
    assert empty.citations == []
