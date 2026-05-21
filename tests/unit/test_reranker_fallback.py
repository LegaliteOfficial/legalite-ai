"""Reranker falls back to no-op when misconfigured."""

from __future__ import annotations

import os

from app.core.config import get_settings
from app.rag.reranker import NoOpReranker, build_reranker
from app.rag.retriever import RetrievedPassage


def _p(vid: str) -> RetrievedPassage:
    return RetrievedPassage(
        score=0.5,
        content="x",
        law=None,
        article=None,
        section=None,
        chapter=None,
        clause=None,
        subsection=None,
        page_start=None,
        page_end=None,
        vector_id=vid,
    )


def test_no_op_returns_top_n() -> None:
    reranker = NoOpReranker()
    out = reranker.rerank("q", [_p("a"), _p("b"), _p("c")], top_n=2)
    assert [p.vector_id for p in out] == ["a", "b"]


def test_build_reranker_falls_back_without_key() -> None:
    # Ensure no Cohere key, then build_reranker should return NoOpReranker.
    os.environ.pop("COHERE_API_KEY", None)
    get_settings.cache_clear()
    reranker = build_reranker()
    assert isinstance(reranker, NoOpReranker)
