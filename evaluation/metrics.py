"""Metrics for the legal RAG evaluation harness.

Each metric takes a list of `EvalRun` records (one per dataset entry) and
returns a single float. The runner aggregates these into a report.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.qa import AskResponse


@dataclass(slots=True)
class EvalCase:
    id: str
    question: str
    expected_law: str | None = None
    expected_section: str | None = None
    expected_article: str | None = None
    expected_law_type: str | None = None
    must_refuse: bool = False


@dataclass(slots=True)
class EvalRun:
    case: EvalCase
    response: AskResponse


# ---- metrics ---------------------------------------------------------------


def recall_at_k(runs: list[EvalRun], k: int = 5) -> float:
    """Fraction of cases where the expected law appears in the top-k citations."""
    eligible = [r for r in runs if r.case.expected_law and not r.case.must_refuse]
    if not eligible:
        return 0.0
    hits = 0
    for r in eligible:
        expected = (r.case.expected_law or "").strip().lower()
        cited = [c.law_name.lower() for c in r.response.citations[:k] if c.law_name]
        if any(expected in c for c in cited):
            hits += 1
    return hits / len(eligible)


def citation_accuracy(runs: list[EvalRun]) -> float:
    """Fraction of cases where (law, section/article) matches a top-3 citation."""
    eligible = [r for r in runs if r.case.expected_law and not r.case.must_refuse]
    if not eligible:
        return 0.0
    hits = 0
    for r in eligible:
        expected_law = (r.case.expected_law or "").strip().lower()
        expected_sec = (r.case.expected_section or "").strip()
        expected_art = (r.case.expected_article or "").strip()
        for c in r.response.citations[:3]:
            law_match = expected_law in (c.law_name or "").lower()
            sec_match = not expected_sec or expected_sec == (c.section or "")
            art_match = not expected_art or expected_art == (c.article or "")
            if law_match and sec_match and art_match:
                hits += 1
                break
    return hits / len(eligible)


def refusal_correctness(runs: list[EvalRun]) -> float:
    """How well refusals align with `must_refuse`.

    For cases marked must_refuse: confidence should be 'low'.
    For cases NOT marked must_refuse: confidence should be 'medium' or 'high'.
    """
    if not runs:
        return 0.0
    hits = 0
    for r in runs:
        is_low = r.response.confidence == "low"
        if r.case.must_refuse and is_low:
            hits += 1
        elif not r.case.must_refuse and not is_low:
            hits += 1
    return hits / len(runs)


def hallucination_rate(runs: list[EvalRun]) -> float:
    """Heuristic: cases where confidence != 'low' but no citations were produced.

    A real implementation would compare against a curated truth set. This is a
    floor estimate: ungrounded answers we *know* are bad.
    """
    if not runs:
        return 0.0
    bad = sum(
        1
        for r in runs
        if r.response.confidence != "low" and not r.response.citations
    )
    return bad / len(runs)


# ---- aggregation -----------------------------------------------------------


def summarize(runs: list[EvalRun]) -> dict[str, float]:
    return {
        "recall@5": round(recall_at_k(runs, k=5), 3),
        "citation_accuracy@3": round(citation_accuracy(runs), 3),
        "refusal_correctness": round(refusal_correctness(runs), 3),
        "hallucination_rate": round(hallucination_rate(runs), 3),
        "n_cases": float(len(runs)),
    }
