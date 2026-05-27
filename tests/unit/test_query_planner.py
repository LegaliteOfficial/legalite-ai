"""QueryPlanner — heuristic classification + safety guarantees (Phase 5 M2)."""

from __future__ import annotations

from app.rag.query_planner import QueryIntent, QueryPlanner


def _plan(question: str, *, has_memory: bool = False):
    return QueryPlanner(llm_fallback_enabled=False).plan(question, has_memory=has_memory)


def test_case_similarity_detected() -> None:
    plan = _plan("find similar cases to this tenancy dispute")
    assert plan.intent == QueryIntent.CASE_SIMILARITY
    assert plan.confidence >= 0.5
    assert plan.source == "heuristic"


def test_case_comparison_detected() -> None:
    plan = _plan("compare these two judgments")
    assert plan.intent == QueryIntent.CASE_COMPARISON


def test_case_summary_detected() -> None:
    plan = _plan("summarize this case")
    assert plan.intent == QueryIntent.CASE_SUMMARY


def test_tenant_contextual_detected() -> None:
    plan = _plan("how did our firm handle the last eviction matter?")
    assert plan.intent == QueryIntent.TENANT_CONTEXTUAL


def test_statute_question_detected() -> None:
    plan = _plan("what does section 17 of the Rent Act say?")
    assert plan.intent == QueryIntent.STATUTE_QUESTION


def test_procedural_question_detected() -> None:
    plan = _plan("how do I file a motion in the high court?")
    assert plan.intent == QueryIntent.PROCEDURAL_QUESTION


def test_follow_up_detected() -> None:
    plan = _plan("what if the tenant is pregnant?", has_memory=True)
    assert plan.intent == QueryIntent.FOLLOW_UP


def test_insufficient_context_for_ambiguous_pronoun_without_memory() -> None:
    plan = _plan("this case")
    assert plan.intent == QueryIntent.INSUFFICIENT_CONTEXT
    assert plan.confidence >= 0.8


def test_ambiguous_pronoun_with_memory_falls_through() -> None:
    """With memory, 'this case' may be valid — don't refuse on the planner level."""
    plan = _plan("this case", has_memory=True)
    assert plan.intent != QueryIntent.INSUFFICIENT_CONTEXT


def test_empty_question_short_circuits() -> None:
    plan = _plan("")
    assert plan.intent == QueryIntent.INSUFFICIENT_CONTEXT


def test_general_fallback_when_no_rule_matches() -> None:
    plan = _plan("explain Ghana legal principles broadly")
    assert plan.intent in (QueryIntent.GENERAL_LEGAL_QA, QueryIntent.STATUTE_QUESTION)


def test_planner_never_calls_llm_when_fallback_disabled(monkeypatch) -> None:
    """Determinism guarantee — heuristic-only mode must not import openai."""
    planner = QueryPlanner(llm_fallback_enabled=False)

    def _fail(*_args, **_kwargs):
        raise AssertionError("LLM fallback must not run when disabled")

    monkeypatch.setattr(planner, "_classify_llm", _fail)
    planner.plan("compare these two judgments")
