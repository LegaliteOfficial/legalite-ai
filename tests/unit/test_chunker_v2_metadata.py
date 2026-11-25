"""Chunker v2: detects law_type/act_number/effective_year + part + topic."""

from __future__ import annotations

from app.ingestion.legal_chunker import LegalChunker
from app.ingestion.pdf_parser import ParsedPdf, PdfPage


def _parsed(text_pages: list[str]) -> ParsedPdf:
    return ParsedPdf(pages=[PdfPage(page_number=i + 1, text=t) for i, t in enumerate(text_pages)])


def test_detects_constitution() -> None:
    parsed = _parsed(
        [
            "CONSTITUTION OF THE REPUBLIC OF GHANA, 1992\n"
            "CHAPTER 5\n"
            "ARTICLE 14\n"
            "Every person shall be entitled to personal liberty.\n",
        ]
    )
    chunks = LegalChunker().chunk(parsed, doc_title="Constitution of Ghana")
    assert chunks
    assert chunks[0].law_type == "constitution"
    assert chunks[0].effective_year == 1992
    assert any(c.article == "14" for c in chunks)


def test_detects_act_number_and_year() -> None:
    parsed = _parsed(
        [
            "LABOUR ACT, 2003 (ACT 651)\n"
            "PART V\n"
            "SECTION 57\n"
            "Maternity protections are guaranteed for employees.\n"
            "An employer shall not dismiss a worker because of pregnancy.\n",
        ]
    )
    chunks = LegalChunker().chunk(parsed, doc_title="Labour Act")
    assert chunks
    assert chunks[0].law_type == "act"
    assert chunks[0].act_number == "651"
    assert chunks[0].effective_year == 2003
    assert any(c.part == "V" for c in chunks)
    assert any(c.section == "57" for c in chunks)


def test_topic_heuristic_employment() -> None:
    parsed = _parsed(
        ["LABOUR ACT, 2003 (ACT 651)\nSECTION 15\nWages must be paid promptly to every employee.\n"]
    )
    chunks = LegalChunker().chunk(parsed, doc_title="Labour Act")
    assert any(c.legal_topic == "employment" for c in chunks)
