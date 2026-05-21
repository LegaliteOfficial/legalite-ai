"""Confidence scoring for legal answers.

We do NOT trust the LLM's self-reported confidence. Instead we compute a
deterministic, multi-signal score from objective inputs:

  1. Retrieval strength   — top retrieval/RRF score (proxy: are we close in
     vector or keyword space?)
  2. Rerank strength      — top rerank score when available (cross-encoder
     judgement that the passage actually answers the question)
  3. Citation consistency — do the top passages agree on the same law? When
     several distinct laws appear with similar relevance, the question may
     be genuinely conflicted, so we down-rank.
  4. Answer grounding     — does the generated answer reference at least one
     citation marker ([1], [2], …)?

A "low" confidence triggers the refusal contract: the answer is replaced
with the exact refusal sentence so the system never speculates on weak
evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.core.config import get_settings
from app.rag.retriever import RetrievedPassage

Confidence = Literal["high", "medium", "low"]

_CITATION_MARKER_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


@dataclass(slots=True)
class ConfidenceAssessment:
    level: Confidence
    score: float  # 0..1
    refused: bool
    reason: str  # short human-readable summary


class ConfidenceEngine:
    def __init__(
        self,
        *,
        min_retrieval_score: float | None = None,
    ) -> None:
        settings = get_settings()
        self._min_retrieval = (
            min_retrieval_score
            if min_retrieval_score is not None
            else settings.rag_min_confidence_score
        )

    def assess_before_generation(
        self, passages: list[RetrievedPassage]
    ) -> ConfidenceAssessment:
        """Cheap, pre-LLM check. Used to short-circuit weak retrieval."""
        if not passages:
            return ConfidenceAssessment(
                level="low",
                score=0.0,
                refused=True,
                reason="no passages retrieved",
            )

        top_retrieval = passages[0].score
        top_rerank = next(
            (p.rerank_score for p in passages if p.rerank_score is not None), None
        )

        # If we have rerank scores, trust them more. Otherwise rely on retrieval.
        primary = top_rerank if top_rerank is not None else top_retrieval

        if top_rerank is None and top_retrieval < self._min_retrieval:
            return ConfidenceAssessment(
                level="low",
                score=float(primary),
                refused=True,
                reason=f"top retrieval score {top_retrieval:.2f} below threshold {self._min_retrieval}",
            )
        if top_rerank is not None and top_rerank < 0.15:
            return ConfidenceAssessment(
                level="low",
                score=float(top_rerank),
                refused=True,
                reason=f"top rerank score {top_rerank:.2f} indicates no relevant passage",
            )

        level: Confidence
        if primary >= 0.6:
            level = "high"
        elif primary >= 0.3:
            level = "medium"
        else:
            level = "low"
        return ConfidenceAssessment(
            level=level,
            score=float(primary),
            refused=level == "low",
            reason=f"primary={primary:.2f} (rerank={top_rerank}, retrieval={top_retrieval:.2f})",
        )

    def assess_final(
        self,
        passages: list[RetrievedPassage],
        answer: str,
        pre_assessment: ConfidenceAssessment,
    ) -> ConfidenceAssessment:
        """Post-LLM refinement using answer grounding + citation consistency."""
        if pre_assessment.refused:
            return pre_assessment

        grounded = bool(_CITATION_MARKER_RE.search(answer))
        consistency = _citation_consistency(passages)

        # Down-rank when answer is ungrounded.
        bonus = 0.0
        if grounded:
            bonus += 0.05
        bonus += (consistency - 0.5) * 0.2  # +/- 0.1
        final_score = max(0.0, min(1.0, pre_assessment.score + bonus))

        if not grounded:
            level: Confidence = "low"
            return ConfidenceAssessment(
                level="low",
                score=final_score,
                refused=True,
                reason="answer contains no citation markers; refusing to surface ungrounded content",
            )
        if final_score >= 0.6 and consistency >= 0.6:
            level = "high"
        elif final_score >= 0.3:
            level = "medium"
        else:
            level = "low"

        return ConfidenceAssessment(
            level=level,
            score=final_score,
            refused=level == "low",
            reason=(
                f"final={final_score:.2f} consistency={consistency:.2f} "
                f"grounded={grounded}"
            ),
        )


def _citation_consistency(passages: list[RetrievedPassage]) -> float:
    """Fraction of the top passages that come from the same law.

    1.0 = perfect agreement; 0.0 = every passage cites a different law.
    """
    laws = [p.law for p in passages[:5] if p.law]
    if not laws:
        return 0.5  # neutral when no law metadata available
    top_law = max(set(laws), key=laws.count)
    return laws.count(top_law) / len(laws)
