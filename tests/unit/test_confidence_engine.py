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
    )


def test_refuses_when_no_passages() -> None:
    engine = ConfidenceEngine()
    a = engine.assess_before_generation([])
    assert a.refused
    assert a.level == "low"


def test_refuses_when_retrieval_below_threshold() -> None:
    engine = ConfidenceEngine(min_retrieval_score=0.5)
    a = engine.assess_before_generation([_p(0.2)])
    assert a.refused
    assert a.level == "low"


def test_high_confidence_on_strong_signal() -> None:
    engine = ConfidenceEngine(min_retrieval_score=0.3)
    a = engine.assess_before_generation(
        [_p(0.85, law="Labour Act", rerank=0.9), _p(0.8, law="Labour Act")]
    )
    assert a.level == "high"
    assert not a.refused


def test_final_refuses_when_answer_ungrounded() -> None:
    engine = ConfidenceEngine(min_retrieval_score=0.3)
    pre = engine.assess_before_generation([_p(0.85, law="Labour Act")])
    final = engine.assess_final(
        [_p(0.85, law="Labour Act")],
        answer="There is no citation marker here at all.",
        pre_assessment=pre,
    )
    assert final.refused
    assert final.level == "low"


def test_final_keeps_high_when_answer_grounded_and_consistent() -> None:
    engine = ConfidenceEngine(min_retrieval_score=0.3)
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
