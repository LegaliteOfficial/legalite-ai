# ruff: noqa: E501
"""Section labels must be trustworthy: the assistant cites them.

Regression for answers citing "Land Act 2020, s.28, p.146" (a Schedule
item) and "NDK Financial Services v Safkos Farms, s.85" (a numbered
paragraph of a judgment).
"""

from __future__ import annotations

from app.ingestion.legal_chunker import LegalChunker
from app.ingestion.pdf_parser import ParsedPdf, PdfPage


def _pages(*texts: str) -> ParsedPdf:
    return ParsedPdf(pages=[PdfPage(page_number=i + 1, text=t) for i, t in enumerate(texts)])


def _act(total_pages: int = 30) -> ParsedPdf:
    pages = ["" for _ in range(total_pages)]
    pages[0] = (
        "ARRANGEMENT OF SECTIONS\n1. Short title\n2. Interpretation\n3. Registration\nSCHEDULE\n"
    )
    pages[2] = (
        "1. Short title\nThis Act may be cited as the Land Act.\n"
        "2. Interpretation\nIn this Act, unless the context otherwise requires.\n"
    )
    pages[10] = (
        "3. Registration\nAn instrument shall be registered.\n"
        "40. Consequential amendments of other enactments apply.\n"
        "4. Priority\nA registered instrument has priority.\n"
    )
    pages[27] = "SCHEDULE\n1. Form of application\n28. Fee for search\n"
    return _pages(*pages)


def _labels(chunks: list) -> list[tuple[str | None, int | None]]:  # type: ignore[type-arg]
    return [(c.section, c.page_start) for c in chunks]


def test_enactment_sections_follow_order_and_stop_at_schedule() -> None:
    chunks = LegalChunker().chunk(_act(), doc_title="Land Act, 2020", doc_type="Act")
    sections = {c.section for c in chunks if c.section}
    assert sections == {"1", "2", "3", "4"}
    # A far jump ("40.") is text inside section 3, not a new section.
    s3 = next(c for c in chunks if c.section == "3" and c.page_start == 11)
    assert "40. Consequential" in s3.content
    # Nothing in the Schedule carries a section number.
    late = [c for c in chunks if (c.page_start or 0) >= 28]
    assert late and all(c.section is None for c in late)


def test_judgment_paragraphs_are_not_sections() -> None:
    text = (
        "IN THE SUPREME COURT OF GHANA\nJUDGMENT\n"
        "84. The appellant relied on section 26 of the Act.\n"
        "85. We are unable to accept that submission.\n"
        "SECTION 26\nquoted provision text\n"
    )
    chunks = LegalChunker().chunk(_pages(text, "86. Appeal dismissed.\n"), doc_type="Case Law")
    assert chunks
    assert all(c.section is None and c.article is None for c in chunks)
    # Paragraph boundaries still split chunks, so page numbers stay exact.
    assert any(c.page_start == 2 and "86." in c.content for c in chunks)


def test_judgment_detected_from_header_without_doc_type() -> None:
    text = "IN THE COURT OF APPEAL\nJUDGMENT\n12. The trial judge erred.\n"
    chunks = LegalChunker().chunk(_pages(text))
    assert all(c.section is None for c in chunks)


def test_constitution_numbered_provisions_are_articles() -> None:
    text = (
        "THE CONSTITUTION OF THE REPUBLIC OF GHANA\n"
        "18. (1) Every person has the right to own property.\n"
    )
    pages = _pages(
        "THE CONSTITUTION OF THE REPUBLIC OF GHANA\n1. The Sovereignty of Ghana resides in the people.\n",
        "2. Any person who alleges that an enactment is inconsistent.\n",
    )
    chunks = LegalChunker().chunk(pages, doc_type="Constitution")
    assert {c.article for c in chunks if c.article} == {"1", "2"}
    assert all(c.section is None for c in chunks)
    assert text  # header form also recognised without doc_type
    header_only = LegalChunker().chunk(_pages(text.replace("18.", "1.")))
    assert {c.article for c in header_only if c.article} == {"1"}


def test_one_stray_number_does_not_shift_later_sections() -> None:
    """A cross-reference starting a line ("59. ...") must not make the real
    sections after it (53-58) look like they run backwards."""
    pages = ["" for _ in range(40)]
    pages[0] = "ARRANGEMENT OF SECTIONS\n1. Short title\n"
    body = "".join(f"{n}. Heading {n}\nText of section {n}.\n" for n in range(1, 53))
    pages[5] = body
    pages[6] = "59. Despite section 59, this line is text.\n" + "".join(
        f"{n}. Heading {n}\nText of section {n}.\n" for n in range(53, 59)
    )
    pages[7] = "12. referred to above as a list item\n59. Heading 59\nText.\n"
    chunks = LegalChunker().chunk(_pages(*pages), doc_title="Land Act, 2020", doc_type="Act")
    found = [c.section for c in chunks if c.section and (c.page_start or 0) == 7]
    assert found == [str(n) for n in range(52, 59)] or found == [str(n) for n in range(53, 59)]
    stray = next(c for c in chunks if "Despite section 59" in c.content)
    assert stray.section == "52"
    late = [c.section for c in chunks if (c.page_start or 0) == 8 and c.section]
    assert late == ["59"]


def test_repair_matches_chunker_on_stored_chunks() -> None:
    from app.ingestion.section_repair import StoredChunk, relabel

    chunks = LegalChunker().chunk(_act(), doc_title="Land Act, 2020", doc_type="Act")
    stored = [
        StoredChunk(
            id=i,
            vector_id=None,
            page_start=c.page_start,
            start_char=i,
            content=c.content,
            article=c.article,
            section=c.section,
            clause=c.clause,
            subsection=c.subsection,
        )
        for i, c in enumerate(chunks)
    ]
    assert (
        relabel(stored, doc_type="Act", law_type=None, total_pages=30, title="Land Act, 2020") == []
    )
    # Old-style labels on Schedule items are cleared.
    late = next(s for s in stored if (s.page_start or 0) >= 28)
    late.section = "28"
    changes = relabel(stored, doc_type="Act", law_type=None, total_pages=30, title="Land Act, 2020")
    assert [ch.diff() for ch in changes] == [{"section": ("28", None)}]


def test_convention_paragraphs_restart_inside_each_article() -> None:
    pages = ["" for _ in range(12)]
    pages[0] = "AFRICAN UNION CONVENTION\n"
    pages[5] = (
        "Article 30\nAmendment\n1. A State Party may propose amendments.\n2. Proposals shall be submitted.\n"
        "Article 31\nReservations\n1. A State Party may make reservations.\n2. A reservation shall not be incompatible.\n"
        "3. A reservation may be withdrawn.\n"
    )
    chunks = LegalChunker().chunk(
        _pages(*pages), doc_title="Convention on Ending Violence", doc_type="Act"
    )
    got = [(c.article, c.section) for c in chunks if c.section]
    assert got == [("30", "1"), ("30", "2"), ("31", "1"), ("31", "2"), ("31", "3")]


def test_misread_number_in_a_hole_is_left_unlabelled() -> None:
    pages = ["" for _ in range(12)]
    pages[0] = "ARRANGEMENT OF SECTIONS\n1. Short title\n"
    pages[5] = (
        "1. Short title\nText.\n2. Interpretation\nText.\n"
        "9. The benefit of a covenant passes with the land.\n"
        "5. Leases\nText.\n6. Licences\nText.\n"
    )
    chunks = LegalChunker().chunk(_pages(*pages), doc_title="Land Act, 2020", doc_type="Act")
    misread = next(c for c in chunks if "benefit of a covenant" in c.content)
    assert misread.section is None
    assert [c.section for c in chunks if c.section and c.page_start == 6] == ["1", "2", "5", "6"]
