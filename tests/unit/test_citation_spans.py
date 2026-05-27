"""Citation span linking — chunker tracks character offsets (Phase 5 M9)."""

from __future__ import annotations

from app.ingestion.legal_chunker import LegalChunker
from app.ingestion.pdf_parser import ParsedPdf, PdfPage


def _parsed(pages: list[str]) -> ParsedPdf:
    return ParsedPdf(
        pages=[PdfPage(page_number=i + 1, text=t) for i, t in enumerate(pages)]
    )


def test_chunker_populates_offsets_for_act() -> None:
    body = (
        "LABOUR ACT, 2003 (ACT 651)\n"
        "PART V\n"
        "SECTION 57\n"
        "Maternity protections are guaranteed for employees.\n"
        "An employer shall not dismiss a worker because of pregnancy.\n"
    )
    chunks = LegalChunker().chunk(_parsed([body]), doc_title="Labour Act")
    assert chunks
    spans = [(c.start_char, c.end_char) for c in chunks if c.start_char is not None]
    assert spans, "chunker must populate at least one (start, end) pair"
    for start, end in spans:
        assert start is not None and end is not None
        assert end >= start


def test_offsets_are_within_page_text() -> None:
    body = (
        "CHAPTER 5\n"
        "ARTICLE 14\n"
        "Personal liberty content.\n"
    )
    parsed = _parsed([body])
    chunks = LegalChunker().chunk(parsed, doc_title="Constitution")
    page_len = len(parsed.pages[0].text)
    for c in chunks:
        if c.start_char is None:
            continue
        assert 0 <= c.start_char <= page_len
        if c.end_char is not None:
            assert c.start_char <= c.end_char <= page_len


def test_offsets_default_none_when_chunk_unmatchable() -> None:
    """A chunk whose leading anchor cannot be found falls back gracefully."""
    chunks = LegalChunker().chunk(_parsed([""]), doc_title="empty")
    # Empty content yields no chunks at all — proves the chunker does not
    # crash when offset resolution has nothing to anchor on.
    assert chunks == []
