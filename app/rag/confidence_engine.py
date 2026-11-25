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
        catastrophic_cosine_threshold: float | None = None,
        catastrophic_rerank_threshold: float | None = None,
        confidence_high_threshold: float | None = None,
        confidence_medium_threshold: float | None = None,
        citation_consistency_high_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self._catastrophic_cosine = (
            catastrophic_cosine_threshold
            if catastrophic_cosine_threshold is not None
            else settings.rag_catastrophic_cosine_threshold
        )
        self._catastrophic_rerank = (
            catastrophic_rerank_threshold
            if catastrophic_rerank_threshold is not None
            else settings.rag_catastrophic_rerank_threshold
        )
        self._confidence_high = (
            confidence_high_threshold
            if confidence_high_threshold is not None
            else settings.rag_confidence_high_threshold
        )
        self._confidence_medium = (
            confidence_medium_threshold
            if confidence_medium_threshold is not None
            else settings.rag_confidence_medium_threshold
        )
        self._consistency_high = (
            citation_consistency_high_threshold
            if citation_consistency_high_threshold is not None
            else settings.rag_citation_consistency_high_threshold
        )

    def assess_before_generation(self, passages: list[RetrievedPassage]) -> ConfidenceAssessment:
        """Pre-LLM gating.

        Phase A change: the pre-LLM bar is now very low. We only refuse
        here when retrieval is *catastrophically* empty (zero passages or
        a top cosine indistinguishable from noise). For everything else
        we hand the passages to the LLM and let it synthesise or refuse
        on its own — the post-LLM grounding check (``assess_final``)
        catches ungrounded outputs.

        Why this matters: with hard-cutoff refusal at retrieval time, a
        conversationally-phrased question (e.g. "tell me about the
        constitution") that lands at cosine 0.32 was rejected even
        though dozens of strongly-relevant chunks existed. The LLM,
        given those same chunks, would have produced a solid grounded
        answer. We were leaving quality on the table.
        """
        if not passages:
            return ConfidenceAssessment(
                level="low",
                score=0.0,
                refused=True,
                reason="no passages retrieved",
            )

        # ``passages[0].score`` is the RRF-fused rank score (max ≈ 2/(k+1)
        # ≈ 0.033 for our k=60), which is NOT comparable to the cosine
        # threshold below. Prefer the preserved pre-fusion cosine
        # (``original_score``) when available — that's the signal we tuned
        # thresholds against.
        top_cosine = next(
            (p.original_score for p in passages if p.original_score is not None),
            None,
        )
        top_retrieval = top_cosine if top_cosine is not None else passages[0].score
        top_rerank = next((p.rerank_score for p in passages if p.rerank_score is not None), None)

        # If we have rerank scores, trust them more. Otherwise rely on retrieval.
        primary = top_rerank if top_rerank is not None else top_retrieval

        # Catastrophic-noise refusal — below this we're below noise even
        # on modern text-embedding-3-small.
        if (
            top_cosine is not None
            and top_cosine < self._catastrophic_cosine
            and (top_rerank is None or top_rerank < self._catastrophic_rerank)
        ):
            return ConfidenceAssessment(
                level="low",
                score=float(primary),
                refused=True,
                reason=(
                    f"retrieval is essentially noise "
                    f"(top cosine {top_cosine:.2f}, top rerank "
                    f"{top_rerank if top_rerank is not None else 'n/a'})"
                ),
            )

        # Three-tier confidence on the cosine scale: HIGH = clearly on
        # topic, MEDIUM = related but partial, LOW = weak but the LLM
        # still gets to try. LOW is NOT a refusal anymore — refusal only
        # happens on the catastrophic-noise branch above or in
        # assess_final when the LLM produced ungrounded text.
        level: Confidence
        if primary >= self._confidence_high:
            level = "high"
        elif primary >= self._confidence_medium:
            level = "medium"
        else:
            level = "low"
        return ConfidenceAssessment(
            level=level,
            score=float(primary),
            refused=False,
            reason=f"primary={primary:.2f} (rerank={top_rerank}, retrieval={top_retrieval:.2f})",
        )

    def assess_final(
        self,
        passages: list[RetrievedPassage],
        answer: str,
        pre_assessment: ConfidenceAssessment,
    ) -> ConfidenceAssessment:
        """Post-LLM refinement using answer grounding + citation consistency.

        Phase A change: LOW-confidence retrieval is no longer an
        automatic refusal. If the LLM produced a grounded answer (at
        least one [n] marker) we surface it with a LOW confidence label
        so the client can show a caveat — that's strictly more useful
        than the previous behavior, which returned the canned refusal
        sentence and dropped the LLM's reasoning entirely.

        Refusal still happens in two cases:

        1. Pre-assessment already refused (catastrophic-noise branch).
        2. The LLM output is ungrounded — no [n] markers at all.
           Ungrounded content is the system's biggest hallucination
           risk; we hold the line there.
        """
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
            return ConfidenceAssessment(
                level="low",
                score=final_score,
                refused=True,
                reason="answer contains no citation markers; refusing to surface ungrounded content",
            )

        level: Confidence
        if final_score >= self._confidence_high and consistency >= self._consistency_high:
            level = "high"
        elif final_score >= self._confidence_medium:
            level = "medium"
        else:
            level = "low"

        # LOW is NOT a refusal anymore. A grounded LOW-confidence answer
        # is more useful than the canned refusal sentence — the client
        # already sees ``confidence: "low"`` and can present a caveat.
        return ConfidenceAssessment(
            level=level,
            score=final_score,
            refused=False,
            reason=(f"final={final_score:.2f} consistency={consistency:.2f} grounded={grounded}"),
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
