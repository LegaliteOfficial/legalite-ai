"""Unit tests for the legal chunker."""

from __future__ import annotations

import pytest

from app.ingestion.legal_chunker import LegalChunker
from app.ingestion.pdf_parser import ParsedPdf, PdfPage


@pytest.fixture()
def sample_act() -> ParsedPdf:
    text = (
        "CHAPTER 1\n"
        "ARTICLE 1\n"
        "The Republic of Ghana shall be a sovereign state.\n"
        "ARTICLE 2\n"
        "Citizenship shall be defined as follows.\n"
        "(a) by birth\n"
        "(b) by registration\n"
    )
    return ParsedPdf(pages=[PdfPage(page_number=1, text=text)])


def test_chunker_emits_chunks_with_hierarchy(sample_act: ParsedPdf) -> None:
    chunker = LegalChunker()
    chunks = chunker.chunk(sample_act, doc_title="Constitution of Ghana")
    assert chunks, "expected at least one chunk"
    articles = {c.article for c in chunks if c.article}
    assert articles == {"1", "2"}
    # Clauses (a) and (b) should be detected somewhere
    clauses = {c.clause for c in chunks if c.clause}
    assert {"a", "b"}.issubset(clauses)


def test_chunker_preserves_page_numbers(sample_act: ParsedPdf) -> None:
    chunks = LegalChunker().chunk(sample_act)
    for c in chunks:
        assert c.page_start == 1
        assert c.page_end == 1
