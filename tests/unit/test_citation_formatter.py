"""Unit tests for citation formatting."""

from __future__ import annotations

from app.rag.citation_formatter import format_citations
from app.rag.retriever import RetrievedPassage


def test_format_citations_handles_missing_fields() -> None:
    passages = [
        RetrievedPassage(
            score=0.7,
            content="x",
            law="Act 1052",
            article=None,
            section="12",
            chapter=None,
            clause=None,
            subsection=None,
            page_start=None,
            page_end=None,
            vector_id="v1",
        ),
        RetrievedPassage(
            score=0.5,
            content="y",
            law="Constitution of Ghana",
            article="17",
            section=None,
            chapter="5",
            clause=None,
            subsection=None,
            page_start=20,
            page_end=22,
            vector_id="v2",
        ),
    ]
    citations = format_citations(passages)
    assert citations[0].law == "Act 1052"
    assert citations[0].section == "12"
    assert citations[0].page == ""
    assert citations[1].article == "17"
    assert citations[1].page == "20-22"
