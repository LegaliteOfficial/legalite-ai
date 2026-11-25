"""Unit tests for prompt construction."""

from __future__ import annotations

from app.rag.prompt_builder import build_user_prompt
from app.rag.retriever import RetrievedPassage


def _passage(**overrides: object) -> RetrievedPassage:
    base = {
        "score": 0.8,
        "content": "Sample legal text.",
        "law": "Constitution of Ghana",
        "article": "17",
        "section": None,
        "chapter": "5",
        "clause": None,
        "subsection": None,
        "page_start": 12,
        "page_end": 12,
        "vector_id": "abc",
    }
    base.update(overrides)
    return RetrievedPassage(**base)  # type: ignore[arg-type]


def test_user_prompt_includes_numbered_passages() -> None:
    passages = [_passage(), _passage(article="18", page_start=14, page_end=14)]
    prompt = build_user_prompt("What rights does Article 17 protect?", passages)
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "Article 17" in prompt
    assert "Article 18" in prompt


def test_empty_passages_triggers_refusal_instruction() -> None:
    prompt = build_user_prompt("anything", [])
    assert "I could not confidently determine this" in prompt
