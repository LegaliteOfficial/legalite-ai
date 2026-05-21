"""CitationFormatter v2: dedup + rank by relevance + sources_used."""

from __future__ import annotations

from app.rag.citation_formatter import format_citations, format_sources
from app.rag.retriever import RetrievedPassage


def _p(law: str, section: str, score: float, rerank: float | None = None) -> RetrievedPassage:
    return RetrievedPassage(
        score=score,
        content=f"{law} {section} content",
        law=law,
        article=None,
        section=section,
        chapter=None,
        clause=None,
        subsection=None,
        page_start=10,
        page_end=10,
        vector_id=f"{law}-{section}",
        rerank_score=rerank,
    )


def test_deduplicates_by_law_section() -> None:
    passages = [
        _p("Labour Act", "57", score=0.5, rerank=0.9),
        _p("Labour Act", "57", score=0.4, rerank=0.7),  # duplicate
        _p("Labour Act", "58", score=0.6, rerank=0.4),
    ]
    citations = format_citations(passages)
    assert len(citations) == 2
    # Higher rerank wins → "57" (rerank 0.9) ranked first
    assert citations[0].section == "57"
    assert citations[0].relevance >= citations[1].relevance


def test_sources_used_includes_excerpt() -> None:
    sources = format_sources([_p("Labour Act", "57", score=0.5)])
    assert sources
    assert sources[0].excerpt
    assert sources[0].law_name == "Labour Act"
