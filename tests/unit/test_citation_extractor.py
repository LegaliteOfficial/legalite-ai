"""CitationExtractor — pulls inline case-to-case references."""

from __future__ import annotations

from app.ingestion.citation_extractor import CitationExtractor


def test_extracts_titled_citation() -> None:
    text = (
        "The principle was established in Republic v Tommy Thompson Books Ltd "
        "[1996-97] SCGLR 804, where the court held..."
    )
    out = CitationExtractor().extract(text)
    assert len(out) == 1
    assert out[0].case_title is not None and "Tommy Thompson" in out[0].case_title
    assert out[0].reporter == "SCGLR"
    assert "[1996-97]" in out[0].raw_text


def test_extracts_bare_reporter_citation() -> None:
    text = "See [2020] GHASC 5 on the question of jurisdiction."
    out = CitationExtractor().extract(text)
    assert len(out) == 1
    assert out[0].case_title is None
    assert out[0].reporter == "GHASC"


def test_dedupes_repeated_citations() -> None:
    text = "[2020] GHASC 5 ... see again [2020] GHASC 5 below."
    out = CitationExtractor().extract(text)
    assert len(out) == 1


def test_mixed_citations_in_order() -> None:
    text = (
        "Following Asare v Attorney-General [2003-04] SCGLR 823 and as noted in "
        "[2018] GHACA 22, the doctrine applies."
    )
    out = CitationExtractor().extract(text)
    reporters = {c.reporter for c in out}
    assert "SCGLR" in reporters
    assert "GHACA" in reporters
    # The titled match for Asare should preserve the case title.
    titled = [c for c in out if c.reporter == "SCGLR"]
    assert titled and titled[0].case_title is not None
    assert "Asare" in titled[0].case_title


def test_returns_empty_for_no_match() -> None:
    assert CitationExtractor().extract("No citations here.") == []
