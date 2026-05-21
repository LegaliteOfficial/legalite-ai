"""LegalMemory: token-safe truncation of the conversation window."""

from __future__ import annotations

from app.rag.legal_memory import _truncate_to_budget
from app.rag.prompt_builder import MemoryTurn


def test_truncate_drops_oldest_first() -> None:
    turns = [
        MemoryTurn(role="user", content="a" * 100),
        MemoryTurn(role="assistant", content="b" * 100),
        MemoryTurn(role="user", content="c" * 100),
    ]
    # Budget chosen to fit ~2 turns (max_tokens * 4 chars/token)
    out = _truncate_to_budget(turns, max_tokens=50)
    assert out[-1].content.startswith("c")
    assert len(out) <= 2


def test_truncate_keeps_all_when_under_budget() -> None:
    turns = [MemoryTurn(role="user", content="short")] * 3
    out = _truncate_to_budget(turns, max_tokens=2000)
    assert len(out) == 3
