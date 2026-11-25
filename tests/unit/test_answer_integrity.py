# ruff: noqa: E501
"""Answer integrity: research budget, citation numbering, and flagging
section numbers and cases the model supplied from memory."""

from __future__ import annotations

from app.rag.chat_agent import (
    is_research_question,
    renumber_citations,
    unsupported_references,
    validate_citation_markers,
)
from app.rag.retriever import RetrievedPassage


def _p(content: str, *, law: str = "Land Act 2020", section: str | None = None) -> RetrievedPassage:
    return RetrievedPassage(
        score=1.0,
        content=content,
        law=law,
        article=None,
        section=section,
        chapter=None,
        clause=None,
        subsection=None,
        page_start=1,
        page_end=1,
        vector_id=content[:12],
    )


LAND_PROBLEM = (
    "Kwame sold a plot at East Legon to Ama in 2019 by a written deed, which she did not register. "
    "In 2021 Kwame sold the same plot to Mensah, who searched the register, found nothing, paid and "
    "registered. Ama has been in possession and built a wall. Advise Ama. In your answer: "
    "1. identify the issues; 2. state the relevant provisions of the Land Act, 2020 (Act 1036); "
    "3. discuss the leading Supreme Court decisions on priority between unregistered and registered "
    "interests; 4. explain the remedies available, with authorities."
)


def test_research_detection() -> None:
    assert is_research_question(LAND_PROBLEM)
    assert not is_research_question("What is the notice period for a monthly tenancy?")
    assert not is_research_question("thanks!")


def test_markers_renumbered_in_reading_order_with_passages() -> None:
    passages = [_p(f"passage {i}") for i in range(1, 25)]
    text, cited = validate_citation_markers(
        "Rule A [24]. Rule B [3][24]. Rule C [9].", len(passages)
    )
    out, reordered, ids = renumber_citations(text, passages, cited)
    assert out == "Rule A [1]. Rule B [2][1]. Rule C [3]."
    assert ids == [1, 2, 3]
    assert [p.content for p in reordered[:3]] == ["passage 24", "passage 3", "passage 9"]
    assert len(reordered) == len(passages)


def test_markdown_links_are_not_renumbered() -> None:
    passages = [_p("a"), _p("b")]
    out, _, _ = renumber_citations("See [2] and [1](https://example.com).", passages, [2])
    assert out == "See [1] and [1](https://example.com)."


def test_unsupported_sections_and_cases_are_listed() -> None:
    passages = [
        _p("36. (1) A person shall not deal in land without registration.", section="36"),
        _p(
            "The Supreme Court in Brown v Quarshie held that possession matters.",
            law="Brown v Quarshie",
        ),
    ]
    answer = (
        "Under section 36 [1] registration is required. Section 99 gives priority to the first "
        "registered purchaser. See Brown v Quarshie [2] and Amuzu v Oklikah."
    )
    flagged = unsupported_references(answer, passages, LAND_PROBLEM)
    assert "section 99" in flagged
    assert "Amuzu v Oklikah" in flagged
    assert "section 36" not in flagged
    assert not any("Brown" in f for f in flagged)


def test_references_from_the_question_are_not_flagged() -> None:
    q = "Is Mensah v Boateng still good law on section 12?"
    flagged = unsupported_references(
        "Mensah v Boateng applied section 12.", [_p("unrelated text")], q
    )
    assert flagged == []


def test_references_found_in_the_firms_own_records_are_not_flagged() -> None:
    answer = "Our memo relies on section 10 of NRCD 54 and Boateng v Agyei."
    archived = "Section 10 of the Limitation Act bars recovery. See Boateng v Agyei."
    assert unsupported_references(answer, [_p("unrelated text")], "") != []
    assert unsupported_references(answer, [_p("unrelated text")], "", archived) == []
