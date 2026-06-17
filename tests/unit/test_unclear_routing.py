"""Planner routing for typos and ambiguous short queries.

The motivating bug: a user typed "Am i goo?" (clearly a typo for
"Am I good?") and the system classified it as ``general_legal_qa``,
then refused with the canned "I could not confidently determine"
message. We want short non-legal noise to land in
``CONVERSATIONAL_UNCLEAR`` so the responder can produce a
capability-aware reply instead of a stone-walled refusal.
"""

from __future__ import annotations

from app.rag.conversational_responder import (
    ConversationalResponder,
    _build_unclear_response,
)
from app.rag.query_planner import QueryIntent, QueryPlanner
from app.services.corpus_inventory import CorpusSnapshot


# ── Planner classification ────────────────────────────────────────────────


def _plan(question: str, *, has_memory: bool = False):
    return QueryPlanner(llm_fallback_enabled=False).plan(
        question, has_memory=has_memory
    )


def test_typo_routes_to_unclear() -> None:
    plan = _plan("Am i goo?")
    assert plan.intent == QueryIntent.CONVERSATIONAL_UNCLEAR


def test_fragment_routes_to_unclear() -> None:
    # A bare punctuation fragment used to fall into general_legal_qa
    # and trigger a wasted LLM call. It now short-circuits.
    plan = _plan("???")
    assert plan.intent == QueryIntent.CONVERSATIONAL_UNCLEAR


def test_short_gibberish_routes_to_unclear() -> None:
    plan = _plan("asdfasdf")
    assert plan.intent == QueryIntent.CONVERSATIONAL_UNCLEAR


def test_substantive_legal_question_still_routes_to_legal() -> None:
    # Sanity: the new unclear branch must NOT swallow real legal
    # questions. A real query mentions a legal substance keyword
    # AND is well over 40 characters.
    plan = _plan("What does the Constitution say about citizenship?")
    assert plan.intent != QueryIntent.CONVERSATIONAL_UNCLEAR
    # The exact label is statute_question because "constitution" is
    # in the statute regex; the important thing is "not unclear".
    assert plan.intent == QueryIntent.STATUTE_QUESTION


def test_short_query_with_legal_substance_skips_unclear() -> None:
    # "Constitution?" is short but contains a legal-substance keyword.
    # We don't want to redirect those — the user clearly wants legal
    # content, just terse.
    plan = _plan("Constitution?")
    assert plan.intent != QueryIntent.CONVERSATIONAL_UNCLEAR


def test_specific_greeting_still_wins_over_unclear() -> None:
    # The conversational rules fire before the unclear fallback. "Hi"
    # should still be classified as GREETING, not UNCLEAR.
    plan = _plan("Hi")
    assert plan.intent == QueryIntent.CONVERSATIONAL_GREETING


# ── Unclear responder shape ────────────────────────────────────────────────


def _snapshot(
    counts: dict[str, int],
    samples: dict[str, list[str]] | None = None,
) -> CorpusSnapshot:
    return CorpusSnapshot(
        counts_by_doc_type=counts,
        sample_titles_by_doc_type=samples or {},
        total_docs=sum(counts.values()),
        snapshot_at=0.0,
    )


def test_unclear_response_lists_corpus_categories() -> None:
    snap = _snapshot(
        counts={"Constitution": 1961, "Case Law": 24, "Act": 3},
        samples={
            "Constitution": ["1992 Constitution of Ghana"],
            "Case Law": ["Acquah v. Kwesi (2024 GHACC 215)"],
        },
    )
    body = _build_unclear_response("Am i goo?", snap)
    assert "Constitution" in body
    assert "1,961" in body  # thousands-separated count
    assert "Case Law" in body
    # Seed questions surface in a "could try asking" block.
    assert "could try asking" in body.lower()


def test_unclear_response_uses_concept_questions_not_titles() -> None:
    """Title-specific suggestions ("Summarize X") were dropped because
    vector retrieval can't reliably resolve a case from its title
    alone. We suggest concept questions the corpus can actually
    answer instead.
    """
    snap = _snapshot(
        counts={"Case Law": 163},
        samples={"Case Law": ["Republic v. Danquah (2024 GHACC 285)"]},
    )
    body = _build_unclear_response("???", snap)
    # Should NOT contain "Summarize <title>" — that suggestion shape
    # is explicitly retired.
    assert "Summarize Republic" not in body
    # Should still produce a concept-based suggestion block.
    assert "could try asking" in body.lower()


def test_unclear_response_renders_as_commonmark_list() -> None:
    """The bullets must be true markdown lists (``-`` prefix + blank
    line before the list) so ReactMarkdown can render them properly.
    Single-newline ``•`` bullets collapse to one line in CommonMark.
    """
    snap = _snapshot(counts={"Case Law": 24, "Constitution": 100})
    body = _build_unclear_response("hi?", snap)
    # Bullets use ``- `` prefix, not ``• ``.
    assert "- " in body
    assert "• " not in body
    # Header → blank line → bullets pattern. We check for the pattern
    # by looking for an empty line followed by a bullet.
    assert "\n\n- " in body


def test_unclear_response_when_corpus_is_empty() -> None:
    snap = _snapshot(counts={}, samples={})
    body = _build_unclear_response("Am i goo?", snap)
    # Falls back to the empty-corpus hint — no fake suggestions.
    assert "uploaded" in body.lower()


def test_unclear_response_when_inventory_is_none() -> None:
    body = _build_unclear_response("Am i goo?", None)
    # Same shape as the empty-corpus path — never crashes on missing
    # inventory.
    assert body  # non-empty
    assert "uploaded" in body.lower()


def test_responder_returns_high_confidence_for_unclear() -> None:
    # Conversational replies are deterministic templates, not legal
    # claims — they're "high" confidence by definition.
    responder = ConversationalResponder()
    snap = _snapshot(counts={"Constitution": 1961})
    answer = responder.respond(
        "Am i goo?", QueryIntent.CONVERSATIONAL_UNCLEAR, inventory=snap
    )
    assert answer.confidence == "high"
    assert answer.intent == QueryIntent.CONVERSATIONAL_UNCLEAR
