"""Tests for the evaluation harness (evaluation/metrics.py, runner.py).

Phase 2: this file existed nowhere before — the evaluator itself had zero
test coverage. Everything here is a pure unit test against fake
AskResponse/Citation/SourceUsed objects — no live Qdrant/Postgres/OpenAI.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.schemas.qa import AskResponse, Citation, SourceUsed
from evaluation.metrics import (
    EvalCase,
    EvalRun,
    citation_accuracy,
    document_recall,
    hallucination_rate,
    law_name_matches,
    passage_support_rate,
    recall_at_k,
    refusal_correctness,
    summarize,
    summarize_by_category,
    summarize_verified,
)


def _citation(law_name: str = "", section: str = "", article: str = "") -> Citation:
    return Citation(law_name=law_name, section=section, article=article)


def _source(law_name: str = "", excerpt: str = "") -> SourceUsed:
    return SourceUsed(law_name=law_name, excerpt=excerpt)


def _response(
    *,
    confidence: str = "high",
    citations: list[Citation] | None = None,
    sources_used: list[SourceUsed] | None = None,
) -> AskResponse:
    return AskResponse(
        answer="An answer.",
        citations=citations or [],
        confidence=confidence,  # type: ignore[arg-type]
        sources_used=sources_used or [],
    )


def _run(case: EvalCase, **response_kwargs: object) -> EvalRun:
    return EvalRun(case=case, response=_response(**response_kwargs))  # type: ignore[arg-type]


# ---- dataset parsing --------------------------------------------------------


def test_eval_case_loads_with_only_required_fields() -> None:
    case = EvalCase(id="c1", question="A question?")
    assert case.verified is False
    assert case.must_refuse is False
    assert case.expected_citations == []
    assert case.notes is None


def test_eval_case_loads_verified_case() -> None:
    case = EvalCase(
        id="c1",
        question="A question?",
        expected_law="Some Act",
        expected_section="12",
        verified=True,
    )
    assert case.verified is True


def test_eval_case_missing_required_field_raises() -> None:
    with pytest.raises(TypeError):
        EvalCase(id="c1")  # type: ignore[call-arg]  # missing `question`


def test_eval_case_rejects_unknown_field() -> None:
    """A malformed row (typo'd key, or a stray annotator field like the
    JSON-only convenience some datasets might add) must fail loudly at
    load time, not silently drop data or crash deep inside a metric."""
    with pytest.raises(TypeError):
        EvalCase(id="c1", question="Q?", this_is_not_a_field="x")  # type: ignore[call-arg]


def test_eval_case_is_a_plain_dataclass_so_dict_unpacking_works() -> None:
    """Sanity check on the loading mechanism runner.load_cases relies on:
    EvalCase(**row) for a plain JSON dict."""
    row = {"id": "c1", "question": "Q?", "verified": True, "category": "statute_lookup"}
    case = EvalCase(**row)
    assert dataclasses.is_dataclass(case)
    assert case.category == "statute_lookup"


# ---- verification separation (Phase 2 core fix) -----------------------------


def test_verified_fraction_calculated_correctly() -> None:
    runs = [
        _run(EvalCase(id="a", question="Q", verified=True)),
        _run(EvalCase(id="b", question="Q", verified=True)),
        _run(EvalCase(id="c", question="Q", verified=False)),
    ]
    report = summarize(runs)
    assert report["n_cases"] == 3
    assert report["n_verified_cases"] == 2
    assert report["n_unverified_cases"] == 1
    assert report["verified_fraction"] == pytest.approx(2 / 3, abs=1e-3)


def test_unverified_cases_excluded_from_verified_baseline() -> None:
    """The core Phase 2 bug fix: an unverified case's guessed expected_law
    must not silently blend into the numbers a reader would treat as
    trustworthy."""
    verified_case = EvalCase(id="v", question="Q", expected_law="Labour Act", verified=True)
    unverified_case = EvalCase(
        id="u", question="Q", expected_law="Totally Wrong Act", verified=False
    )
    # The verified case's citation is correct; the unverified case's guess
    # is deliberately wrong — if it leaked into the verified baseline, the
    # baseline's citation_accuracy would drop below 1.0.
    runs = [
        _run(verified_case, citations=[_citation(law_name="Labour Act")]),
        _run(unverified_case, citations=[_citation(law_name="Labour Act")]),
    ]
    baseline = summarize_verified(runs)
    assert baseline is not None
    assert baseline["n_cases"] == 1
    assert baseline["citation_accuracy@3"] == 1.0

    exploratory = summarize(runs)
    # The exploratory (all-cases) view DOES include the unverified miss —
    # that's expected, it's explicitly labelled as not-trustworthy.
    assert exploratory["citation_accuracy@3"] == pytest.approx(0.5, abs=1e-3)


def test_summarize_verified_returns_none_when_nothing_verified() -> None:
    runs = [_run(EvalCase(id="a", question="Q", verified=False))]
    assert summarize_verified(runs) is None


# ---- metric correctness: citations / documents / passages ------------------


def test_citation_accuracy_hit_on_matching_law_and_section() -> None:
    case = EvalCase(id="a", question="Q", expected_law="Labour Act", expected_section="57")
    run = _run(case, citations=[_citation(law_name="Labour Act, 2003", section="57")])
    assert citation_accuracy([run]) == 1.0


def test_citation_accuracy_miss_on_wrong_section() -> None:
    case = EvalCase(id="a", question="Q", expected_law="Labour Act", expected_section="57")
    run = _run(case, citations=[_citation(law_name="Labour Act, 2003", section="12")])
    assert citation_accuracy([run]) == 0.0


def test_citation_accuracy_none_when_no_eligible_cases() -> None:
    case = EvalCase(id="a", question="Q")  # no expected_law at all
    run = _run(case)
    assert citation_accuracy([run]) is None


# ---- law_name_matches: word-boundary fix (Phase 2.5) ------------------------
#
# Realistic Step 12 scenarios: "Expected document ACT-001, actual citation
# ACT-001" must match; "...actual citation ACT-002" must not. The less
# obvious case plain `in` gets wrong: ACT-001 as a substring of ACT-0011
# (a DIFFERENT act whose code merely starts the same way).


def test_law_name_matches_exact() -> None:
    assert law_name_matches("ACT-001", "ACT-001")


def test_law_name_matches_rejects_different_code() -> None:
    assert not law_name_matches("ACT-001", "ACT-002")


def test_law_name_matches_rejects_numeric_suffix_false_positive() -> None:
    """The bug plain substring matching has: 'act-001' IS a Python
    substring of 'act-0011', but they are different acts."""
    assert "act-001" in "act-0011"  # confirms the naive approach is unsafe
    assert not law_name_matches("ACT-001", "ACT-0011")


def test_law_name_matches_still_allows_the_documented_qualifier_case() -> None:
    """The behavior this whole function exists to preserve: a citation's
    law_name normally carries a year/act-number qualifier beyond the bare
    name a reviewer writes."""
    assert law_name_matches("Labour Act", "Labour Act, 2003 (Act 651)")


def test_law_name_matches_case_insensitive() -> None:
    assert law_name_matches("labour act", "LABOUR ACT, 2003")


def test_citation_accuracy_realistic_document_code_match() -> None:
    case = EvalCase(id="a", question="Q", expected_law="ACT-001")
    assert citation_accuracy([_run(case, citations=[_citation(law_name="ACT-001")])]) == 1.0


def test_citation_accuracy_realistic_document_code_mismatch() -> None:
    case = EvalCase(id="a", question="Q", expected_law="ACT-001")
    assert citation_accuracy([_run(case, citations=[_citation(law_name="ACT-002")])]) == 0.0


def test_citation_accuracy_does_not_false_positive_on_numeric_suffix() -> None:
    case = EvalCase(id="a", question="Q", expected_law="ACT-001")
    assert citation_accuracy([_run(case, citations=[_citation(law_name="ACT-0011")])]) == 0.0


def test_recall_at_k_averages_across_multiple_expected_citations() -> None:
    case = EvalCase(
        id="a",
        question="Q",
        expected_citations=[{"law": "Labour Act"}, {"law": "Constitution"}],
    )
    # Only one of the two expected laws was actually cited.
    run = _run(case, citations=[_citation(law_name="Labour Act")])
    assert recall_at_k([run], k=5) == pytest.approx(0.5, abs=1e-3)


def test_document_recall_hit_when_source_surfaced_without_citation() -> None:
    """document_recall checks sources_used too, not just citations — a
    source can be surfaced as background without being formally cited."""
    case = EvalCase(id="a", question="Q", expected_documents=["Labour Act"])
    run = _run(
        case,
        citations=[],
        sources_used=[_source(law_name="Labour Act, 2003")],
    )
    assert document_recall([run]) == 1.0


def test_document_recall_miss_when_not_retrieved() -> None:
    case = EvalCase(id="a", question="Q", expected_documents=["Labour Act"])
    run = _run(case, citations=[], sources_used=[_source(law_name="Rent Act")])
    assert document_recall([run]) == 0.0


def test_document_recall_none_when_no_case_sets_it() -> None:
    case = EvalCase(id="a", question="Q")
    assert document_recall([_run(case)]) is None


def test_passage_support_rate_substring_match() -> None:
    case = EvalCase(id="a", question="Q", expected_passages=["shall not dismiss"])
    run = _run(
        case,
        sources_used=[_source(excerpt="An employer shall not dismiss an employee for...")],
    )
    assert passage_support_rate([run]) == 1.0


def test_passage_support_rate_miss_when_excerpt_lacks_text() -> None:
    case = EvalCase(id="a", question="Q", expected_passages=["shall not dismiss"])
    run = _run(case, sources_used=[_source(excerpt="Something unrelated.")])
    assert passage_support_rate([run]) == 0.0


def test_passage_support_rate_none_when_no_case_sets_it() -> None:
    assert passage_support_rate([_run(EvalCase(id="a", question="Q"))]) is None


# ---- refusal correctness ----------------------------------------------------


def test_refusal_correctness_hit_when_refusal_expected_and_produced() -> None:
    case = EvalCase(id="a", question="Q", must_refuse=True)
    run = _run(case, confidence="low")
    assert refusal_correctness([run]) == 1.0


def test_refusal_correctness_miss_when_refusal_expected_but_answer_produced() -> None:
    case = EvalCase(id="a", question="Q", must_refuse=True)
    run = _run(case, confidence="high")
    assert refusal_correctness([run]) == 0.0


def test_refusal_correctness_miss_when_refusal_produced_but_not_expected() -> None:
    case = EvalCase(id="a", question="Q", must_refuse=False)
    run = _run(case, confidence="low")
    assert refusal_correctness([run]) == 0.0


def test_refusal_correctness_none_only_when_zero_runs() -> None:
    assert refusal_correctness([]) is None
    # A single non-refusal case is still "applicable" — every case has a
    # must_refuse value, so this must return a real float, never None.
    case = EvalCase(id="a", question="Q")
    assert refusal_correctness([_run(case)]) is not None


# ---- hallucination floor ------------------------------------------------


def test_hallucination_rate_flags_confident_answer_with_no_citations() -> None:
    case = EvalCase(id="a", question="Q")
    run = _run(case, confidence="high", citations=[])
    assert hallucination_rate([run]) == 1.0


def test_hallucination_rate_does_not_flag_grounded_answer() -> None:
    case = EvalCase(id="a", question="Q")
    run = _run(case, confidence="high", citations=[_citation(law_name="Labour Act")])
    assert hallucination_rate([run]) == 0.0


def test_hallucination_rate_does_not_flag_low_confidence_refusal() -> None:
    case = EvalCase(id="a", question="Q", must_refuse=True)
    run = _run(case, confidence="low", citations=[])
    assert hallucination_rate([run]) == 0.0


# ---- category reporting -----------------------------------------------------


def test_summarize_by_category_groups_correctly() -> None:
    runs = [
        _run(EvalCase(id="a", question="Q", category="statute_lookup")),
        _run(EvalCase(id="b", question="Q", category="statute_lookup")),
        _run(EvalCase(id="c", question="Q", category="case_law_lookup")),
        _run(EvalCase(id="d", question="Q")),  # no category
    ]
    by_cat = summarize_by_category(runs)
    assert by_cat["statute_lookup"]["n_cases"] == 2
    assert by_cat["case_law_lookup"]["n_cases"] == 1
    assert by_cat["uncategorized"]["n_cases"] == 1
