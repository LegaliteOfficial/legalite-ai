"""CaseMetadataExtractor — regex-pass tests (LLM enrichment disabled).

These tests use synthetic judgment text — no real PDFs needed. They cover
the structured fields the regex pass is responsible for: case_number,
court, judges, decision_date, parties, outcome, topics.
"""

from __future__ import annotations

from datetime import date

from app.ingestion.case_metadata_extractor import CaseMetadataExtractor
from app.ingestion.pdf_parser import ParsedPdf, PdfPage


def _parsed(pages: list[str]) -> ParsedPdf:
    return ParsedPdf(
        pages=[PdfPage(page_number=i + 1, text=t) for i, t in enumerate(pages)]
    )


def _extractor() -> CaseMetadataExtractor:
    # LLM enrichment off so tests are deterministic / offline.
    return CaseMetadataExtractor(llm_enrichment_enabled=False)


def test_extracts_supreme_court_case_number_and_judges() -> None:
    text = (
        "IN THE SUPREME COURT OF GHANA\n"
        "ACCRA - AD 2020\n"
        "Civil Appeal No. J4/12/2020\n"
        "BETWEEN: REPUBLIC v TOMMY THOMPSON BOOKS LTD\n"
        "CORAM:\n"
        "HON. MR. JUSTICE KOFI ADUMUAH-BOSSMAN, JSC\n"
        "HON. JUSTICE Akamba, JSC\n"
        "DELIVERED ON: 12th March 2020\n"
        "JUDGMENT\n"
        "The appeal is dismissed.\n"
    )
    meta = _extractor().extract(_parsed([text]), declared_title="Republic v Tommy Thompson")
    assert meta.court == "Supreme Court"
    assert meta.case_number is not None and "J4/12/2020" in meta.case_number
    assert meta.decision_date == date(2020, 3, 12)
    assert meta.outcome == "appeal dismissed"
    # Both judges should be picked up (one via long-form, one via post-nominal).
    judge_names = [j.lower() for j in meta.judges]
    assert any("adumuah" in j for j in judge_names)
    assert any("akamba" in j for j in judge_names)
    # Parties detected as appellate roles because "SUPREME COURT" is in head.
    roles = {p.role for p in meta.parties}
    assert roles == {"appellant", "respondent"}


def test_handles_high_court_civil_with_long_date() -> None:
    text = (
        "IN THE HIGH COURT OF JUSTICE\n"
        "ACCRA - LAND DIVISION\n"
        "Suit No. LD/123/2019\n"
        "BETWEEN: KWAME OWUSU (Plaintiff) v ABENA MENSAH (Defendant)\n"
        "JUDGMENT\n"
        "Date: 5th November 2019\n"
        "The plaintiff claims trespass on a parcel of land.\n"
        "Judgment is entered for the plaintiff.\n"
    )
    meta = _extractor().extract(_parsed([text]))
    assert meta.court == "High Court"
    assert meta.case_number is not None and "LD/123/2019" in meta.case_number
    assert meta.decision_date == date(2019, 11, 5)
    assert meta.outcome == "judgment for plaintiff"
    assert "property" in meta.legal_topics
    assert {p.role for p in meta.parties} == {"plaintiff", "defendant"}


def test_falls_back_to_declared_title_when_no_parties() -> None:
    text = "IN RE ESTATE OF JOHN DOE\nHigh Court\nDelivered on 1st January 2024\n"
    meta = _extractor().extract(_parsed([text]), declared_title="Estate of John Doe")
    assert meta.case_title == "Estate of John Doe"
    assert meta.parties == []
    assert meta.court == "High Court"


def test_topic_detection_constitutional_and_employment() -> None:
    text = (
        "IN THE SUPREME COURT OF GHANA\n"
        "BETWEEN: A v B\n"
        "The applicant claims a violation of his fundamental rights under article 17 "
        "of the Constitution. The employer also dismissed him in breach of the labour laws.\n"
    )
    meta = _extractor().extract(_parsed([text]))
    assert "constitutional" in meta.legal_topics
    assert "employment" in meta.legal_topics


def test_no_llm_calls_when_disabled(monkeypatch) -> None:
    """The constructor must respect llm_enrichment_enabled=False even when an API key looks real."""

    extractor = CaseMetadataExtractor(llm_enrichment_enabled=False)
    text = "IN THE HIGH COURT\nBETWEEN: A v B\nDelivered on 1st June 2024\n"

    called = {"count": 0}

    def _fail(*args, **kwargs):  # pragma: no cover — guard against accidental calls
        called["count"] += 1
        raise AssertionError("LLM must not be invoked when enrichment is disabled")

    monkeypatch.setattr(extractor, "_enrich_with_llm", _fail)
    meta = extractor.extract(_parsed([text]))
    assert called["count"] == 0
    # holding/facts_summary stay None — only the regex pass ran.
    assert meta.holding is None
    assert meta.facts_summary is None


def test_reporter_citation_acts_as_case_number_fallback() -> None:
    text = (
        "REPUBLIC v BOAKYE\n"
        "[2020] GHASC 5\n"
        "SUPREME COURT OF GHANA\n"
        "JUDGMENT delivered on 10th February 2020.\n"
    )
    meta = _extractor().extract(_parsed([text]))
    # No "Suit No." present, so the reporter form should be picked up.
    assert meta.case_number is not None and "GHASC" in meta.case_number.upper()


def test_case_number_collapses_pdf_layout_whitespace_and_fits_the_column() -> None:
    """Regression test — production bug.

    A right-aligned page header ("CRIMINAL APPEAL" ... wide gap ... "NO.
    J3/1/2020") extracts with PyMuPDF as a huge run of literal whitespace
    between the label and "No.", because ``_CASE_NUMBER_RE``'s "\\s+"/"\\s*"
    is unbounded. Uncollapsed, this produced a >128-char case_number that
    failed the ``legal_cases.case_number`` VARCHAR(128) insert and took the
    whole ingestion job down with it (see ingestion_service.py's savepoint
    fix for the second half of that incident).
    """
    text = (
        "IN THE SUPREME COURT OF GHANA\n"
        "CRIMINAL APPEAL" + " " * 100 + "\n" + " " * 20 + "NO. J3/1/2020\n"
        "BETWEEN: REPUBLIC v SULEMANA\n"
    )
    meta = _extractor().extract(_parsed([text]))
    assert meta.case_number == "CRIMINAL APPEAL NO. J3/1/2020"
    assert len(meta.case_number) <= 128
    assert "  " not in meta.case_number
    assert "\n" not in meta.case_number
