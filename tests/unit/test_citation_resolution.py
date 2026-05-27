"""CitationResolutionService matching logic + tenant safety (Phase 5 M5)."""

from __future__ import annotations

from app.services.citation_resolution_service import _title_similarity


def test_title_similarity_identical_strings_high() -> None:
    score = _title_similarity("Asare v Attorney-General", "Asare v Attorney-General")
    assert score >= 0.9


def test_title_similarity_case_insensitive_and_punctuation_tolerant() -> None:
    score = _title_similarity(
        "REPUBLIC v TOMMY THOMPSON BOOKS LTD",
        "Republic v. Tommy Thompson Books Ltd",
    )
    assert score >= 0.8


def test_title_similarity_disjoint_titles_low() -> None:
    score = _title_similarity(
        "Republic v Tommy Thompson Books Ltd",
        "Mensah v Bank of Ghana",
    )
    # Some overlap on common tokens ("v") but score must stay well below threshold.
    assert score < 0.3


def test_title_similarity_empty_strings() -> None:
    assert _title_similarity("", "anything") == 0.0
    assert _title_similarity("anything", "") == 0.0
