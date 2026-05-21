"""Reciprocal Rank Fusion: orders by rank, not raw score."""

from __future__ import annotations

from app.rag.hybrid_retriever import _reciprocal_rank_fusion
from app.rag.retriever import RetrievedPassage


def _p(vid: str, score: float, source: str) -> RetrievedPassage:
    return RetrievedPassage(
        score=score,
        content=f"content for {vid}",
        law=None,
        article=None,
        section=None,
        chapter=None,
        clause=None,
        subsection=None,
        page_start=None,
        page_end=None,
        vector_id=vid,
        source=source,
    )


def test_passage_in_both_rankings_wins() -> None:
    vec = [_p("a", 0.9, "vector"), _p("b", 0.8, "vector"), _p("c", 0.7, "vector")]
    kw = [_p("b", 0.6, "keyword"), _p("d", 0.5, "keyword")]

    fused = _reciprocal_rank_fusion([vec, kw], k=60)
    # b is in both lists → should be first
    assert fused[0].vector_id == "b"
    assert {p.vector_id for p in fused} == {"a", "b", "c", "d"}
    assert all(p.source == "hybrid" for p in fused)


def test_rank_dominates_score() -> None:
    # Even with a tiny score, being rank-1 in keyword should beat being rank-3 in vector
    vec = [_p("x", 0.95, "vector"), _p("y", 0.9, "vector"), _p("z", 0.85, "vector")]
    kw = [_p("z", 0.001, "keyword")]
    fused = _reciprocal_rank_fusion([vec, kw], k=60)
    # z gains by being in both → still rises
    ids = [p.vector_id for p in fused]
    assert ids.index("z") < ids.index("y")
