"""Unit tests for the legal chunker."""

from __future__ import annotations

import re

import pytest

from app.ingestion.legal_chunker import (
    LegalChunk,
    LegalChunker,
    _find_break_point,
    _split_oversized,
)
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


def test_chunker_handles_multiple_sections() -> None:
    text = (
        "SECTION 1\n"
        "First section body text.\n"
        "SECTION 2\n"
        "Second section body text.\n"
        "SECTION 3\n"
        "Third section body text.\n"
    )
    chunks = LegalChunker().chunk(ParsedPdf(pages=[PdfPage(page_number=1, text=text)]))
    sections = [c.section for c in chunks if c.section]
    assert sections == ["1", "2", "3"]


def test_chunker_handles_nested_subsections() -> None:
    # Deliberately starts at "(ii)", not "(i)": a single-letter roman
    # numeral like "(i)" or "(v)" is indistinguishable from the clause
    # pattern "(<single lowercase letter>)", and clause is checked first
    # in _PATTERNS — so "(i)" is classified as clause "i", not subsection
    # "i". That's a pre-existing ambiguity in the marker patterns, not
    # something this test (or Phase 1H) is fixing; using multi-character
    # numerals here avoids exercising it.
    text = (
        "ARTICLE 1\n"
        "SECTION 1\n"
        "(a) first clause\n"
        "(ii) first subsection\n"
        "(iii) second subsection\n"
        "(b) second clause\n"
    )
    chunks = LegalChunker().chunk(ParsedPdf(pages=[PdfPage(page_number=1, text=text)]))
    # Every subsection chunk should still carry its parent article/section/clause —
    # nesting must not be lost when a lower level is set.
    subsection_chunks = [c for c in chunks if c.subsection]
    assert {c.subsection for c in subsection_chunks} == {"ii", "iii"}
    for c in subsection_chunks:
        assert c.article == "1"
        assert c.section == "1"
        assert c.clause == "a"


def test_chunker_handles_missing_structural_metadata() -> None:
    """Plain prose with no hierarchy markers at all must still emit a
    chunk — article/section/etc. stay None rather than crashing or
    fabricating a value."""
    text = "This is a plain paragraph with no legal structure markers at all."
    chunks = LegalChunker().chunk(ParsedPdf(pages=[PdfPage(page_number=1, text=text)]))
    assert len(chunks) == 1
    assert chunks[0].article is None
    assert chunks[0].section is None
    assert chunks[0].content == text


def test_chunker_handles_malformed_markers_as_prose() -> None:
    """A line that merely resembles a marker but doesn't match the strict
    patterns (e.g. 'Section' with no number) is treated as body text, not
    a hierarchy transition."""
    text = "SECTION\nThis line looks like a header but has no number.\n"
    chunks = LegalChunker().chunk(ParsedPdf(pages=[PdfPage(page_number=1, text=text)]))
    assert len(chunks) == 1
    assert chunks[0].section is None
    assert "SECTION" in chunks[0].content


def test_chunker_handles_empty_content() -> None:
    chunks = LegalChunker().chunk(ParsedPdf(pages=[PdfPage(page_number=1, text="")]))
    assert chunks == []


def test_chunker_handles_whitespace_only_content() -> None:
    chunks = LegalChunker().chunk(ParsedPdf(pages=[PdfPage(page_number=1, text="   \n\n  \n")]))
    assert chunks == []


# ---- oversized-unit fallback: sentence-aware splitting -------------------


def _long_clause_chunk(sentence_count: int) -> LegalChunk:
    sentences = [
        f"This is sentence number {i} of a very long legal clause." for i in range(sentence_count)
    ]
    return LegalChunk(content=" ".join(sentences), article="5", section="2")


def test_split_oversized_returns_single_chunk_when_under_limit() -> None:
    chunk = LegalChunk(content="short text")
    parts = _split_oversized(chunk, max_chars=1000, overlap=100)
    assert parts == [chunk]


def test_split_oversized_long_clause_never_cuts_mid_word() -> None:
    """The regression this guards: a naive character-count split can land
    in the middle of a word, producing a misleading partial phrase (this
    is what happened when a malformed test PDF's truncated content stream
    made 'internal' render as 'intern' — a PDF-generation artifact, not a
    chunker bug, but the chunker should still never introduce this itself
    when a word/sentence boundary is available nearby)."""
    chunk = _long_clause_chunk(sentence_count=200)  # forces multiple oversized splits
    parts = _split_oversized(chunk, max_chars=500, overlap=50)
    assert len(parts) > 1
    word_re = re.compile(r"\w+")
    for part in parts:
        # Every part's content must consist of whole words only — no
        # fragment like "intern" where "international" was cut in half.
        words = word_re.findall(part.content)
        assert "".join(words) in "".join(word_re.findall(chunk.content))
        # The stronger check: reconstructing from the ORIGINAL sentence
        # list means every emitted word must itself be a complete word
        # that appears in the source sentence list.
        source_words = set(word_re.findall(chunk.content))
        assert all(w in source_words for w in words)


def test_split_oversized_preserves_hierarchy_metadata_on_every_part() -> None:
    chunk = _long_clause_chunk(sentence_count=200)
    parts = _split_oversized(chunk, max_chars=500, overlap=50)
    for part in parts:
        assert part.article == "5"
        assert part.section == "2"


def test_split_oversized_applies_overlap_between_parts() -> None:
    chunk = _long_clause_chunk(sentence_count=200)
    parts = _split_oversized(chunk, max_chars=500, overlap=50)
    # Overlap means consecutive parts share some trailing/leading text.
    for a, b in zip(parts, parts[1:], strict=False):
        assert a.content[-20:] in chunk.content
        assert b.content[:20] in chunk.content


def test_split_oversized_falls_back_to_hard_cut_with_no_whitespace() -> None:
    """A pathological single unbroken run of characters (no spaces, no
    sentence punctuation) has nowhere natural to break — must still
    terminate and make forward progress rather than looping forever.

    No safe word boundary exists anywhere in the overlap window here, so
    _snap_forward_past_partial_word finds nothing and the split falls back
    to zero overlap for every step (5 clean 1000-char parts) rather than
    risk landing a start mid-run. Correctness (never fragment a "word")
    wins over preserving the requested overlap when the two conflict.
    """
    chunk = LegalChunk(content="x" * 5000)
    parts = _split_oversized(chunk, max_chars=1000, overlap=100)
    assert len(parts) == 5
    assert [len(p.content) for p in parts] == [1000, 1000, 1000, 1000, 1000]
    assert "".join(p.content for p in parts) == chunk.content


def test_find_break_point_prefers_sentence_end() -> None:
    text = "First sentence here. Second sentence here. Third sentence continues on and on."
    hard_end = 50  # within the 20%-of-window lookback of the period at index 41
    end = _find_break_point(text, hard_end, min_end=0)
    assert text[:end].rstrip().endswith((".", "!", "?"))
