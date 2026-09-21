"""ConfidenceEngine: refusal on weak retrieval, grounding check, consistency."""

from __future__ import annotations

from app.rag.confidence_engine import ConfidenceEngine
from app.rag.retriever import RetrievedPassage


def _p(score: float, law: str | None = None, rerank: float | None = None) -> RetrievedPassage:
    return RetrievedPassage(
        score=score,
        content="text",
        law=law,
        article=None,
        section=None,
        chapter=None,
        clause=None,
        subsection=None,
        page_start=None,
        page_end=None,
        vector_id="v",
        rerank_score=rerank,
        # Phase A: pre-fusion cosine is preserved on the passage so the
        # engine sees the real retrieval signal, not the RRF rank score.
        original_score=score,
    )


def test_refuses_when_no_passages() -> None:
    engine = ConfidenceEngine()
    a = engine.assess_before_generation([])
    assert a.refused
    assert a.level == "low"


def test_refuses_only_on_catastrophic_noise() -> None:
    """Phase A: LOW retrieval no longer auto-refuses. Only a top cosine
    below the noise floor (~0.10) AND no useful rerank triggers refusal
    pre-generation. Everything else flows to the LLM, which gets the
    chance to synthesise an answer or refuse on its own."""
    engine = ConfidenceEngine()

    # Catastrophic noise — both signals below floor → refuse.
    a = engine.assess_before_generation([_p(0.05, rerank=0.01)])
    assert a.refused
    assert a.level == "low"

    # Weak but above noise floor → LOW label, but NOT refused. The LLM
    # gets to try.
    a = engine.assess_before_generation([_p(0.20)])
    assert not a.refused
    assert a.level == "low"


def test_high_confidence_on_strong_signal() -> None:
    engine = ConfidenceEngine()
    a = engine.assess_before_generation(
        [_p(0.85, law="Labour Act", rerank=0.9), _p(0.8, law="Labour Act")]
    )
    assert a.level == "high"
    assert not a.refused


def test_final_refuses_when_answer_ungrounded() -> None:
    engine = ConfidenceEngine()
    pre = engine.assess_before_generation([_p(0.85, law="Labour Act")])
    final = engine.assess_final(
        [_p(0.85, law="Labour Act")],
        answer="There is no citation marker here at all.",
        pre_assessment=pre,
    )
    assert final.refused
    assert final.level == "low"


def test_final_keeps_high_when_answer_grounded_and_consistent() -> None:
    engine = ConfidenceEngine()
    passages = [
        _p(0.85, law="Labour Act"),
        _p(0.8, law="Labour Act"),
        _p(0.7, law="Labour Act"),
    ]
    pre = engine.assess_before_generation(passages)
    final = engine.assess_final(
        passages,
        answer="Under section 57 [1], dismissal is prohibited [2].",
        pre_assessment=pre,
    )
    assert not final.refused
    assert final.level == "high"


# ---- Phase 1G: prove the thresholds are actually wired, not dead --------
#
# Before this phase, ConfidenceEngine's constructor read
# settings.rag_min_confidence_score into self._min_retrieval and then never
# referenced it again — every gate below was a hardcoded literal. These
# tests fail against that old code (there was no way to change the
# catastrophic-refusal behavior via configuration at all) and pass now.


def test_catastrophic_threshold_is_actually_configurable() -> None:
    # A cosine of 0.20 is NOT catastrophic under the production default
    # (0.10) — the earlier test above already covers that. Raise the
    # threshold past 0.20 and the same passage must now refuse.
    default_engine = ConfidenceEngine()
    assert not default_engine.assess_before_generation([_p(0.20)]).refused

    strict_engine = ConfidenceEngine(catastrophic_cosine_threshold=0.25)
    strict = strict_engine.assess_before_generation([_p(0.20)])
    assert strict.refused
    assert strict.level == "low"


def test_confidence_high_threshold_is_actually_configurable() -> None:
    # 0.6 is "high" under the production default (>= 0.5) but "medium"
    # once the high bar is raised past it.
    default_engine = ConfidenceEngine()
    assert default_engine.assess_before_generation([_p(0.6)]).level == "high"

    stricter_engine = ConfidenceEngine(confidence_high_threshold=0.7)
    assert stricter_engine.assess_before_generation([_p(0.6)]).level == "medium"


def test_citation_consistency_threshold_is_actually_configurable() -> None:
    # Two out of three passages share a law → consistency ~0.67, which
    # clears the production default (0.6) for a "high" final label given a
    # grounded, high-score answer.
    passages = [
        _p(0.9, law="Labour Act"),
        _p(0.85, law="Labour Act"),
        _p(0.8, law="Constitution"),
    ]
    default_engine = ConfidenceEngine()
    pre = default_engine.assess_before_generation(passages)
    final = default_engine.assess_final(passages, answer="See [1].", pre_assessment=pre)
    assert final.level == "high"

    stricter_engine = ConfidenceEngine(citation_consistency_high_threshold=0.9)
    pre2 = stricter_engine.assess_before_generation(passages)
    final2 = stricter_engine.assess_final(passages, answer="See [1].", pre_assessment=pre2)
    assert final2.level == "medium"
