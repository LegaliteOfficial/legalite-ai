"""Compose the final answer from retrieved passages using the OpenAI LLM.

Now driven by the ConfidenceEngine: we may refuse before the LLM is invoked,
and we re-assess after generation based on answer grounding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.confidence_engine import (
    Confidence,
    ConfidenceAssessment,
    ConfidenceEngine,
)
from app.rag.prompt_builder import (
    DISCLAIMER,
    REFUSAL_TEXT,
    SYSTEM_PROMPT,
    MemoryTurn,
    build_user_prompt,
)
from app.rag.retriever import RetrievedPassage

logger = get_logger(__name__)


@dataclass(slots=True)
class GeneratedAnswer:
    answer: str
    confidence: Confidence
    confidence_score: float
    used_passages: list[RetrievedPassage]
    refused: bool
    reasoning_summary: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0


class AnswerGenerator:
    def __init__(
        self,
        *,
        confidence_engine: ConfidenceEngine | None = None,
    ) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_llm_model
        self._confidence = confidence_engine or ConfidenceEngine()

    def generate(
        self,
        question: str,
        passages: list[RetrievedPassage],
        *,
        memory: list[MemoryTurn] | None = None,
    ) -> GeneratedAnswer:
        pre = self._confidence.assess_before_generation(passages)
        if pre.refused:
            return _refusal(passages, pre, reason=pre.reason)

        user_prompt = build_user_prompt(question, passages, memory=memory)
        t0 = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception:
            logger.exception("answer.llm_failed", model=self._model)
            return _refusal(passages, pre, reason="LLM call failed")
        latency_ms = int((time.perf_counter() - t0) * 1000)

        content = (completion.choices[0].message.content or "").strip()
        usage = completion.usage
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0

        if REFUSAL_TEXT in content:
            return GeneratedAnswer(
                answer=_ensure_disclaimer(content),
                confidence="low",
                confidence_score=pre.score,
                used_passages=[],
                refused=True,
                reasoning_summary="LLM declined: insufficient evidence in provided passages",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
            )

        final = self._confidence.assess_final(passages, content, pre)
        if final.refused:
            return _refusal(passages, final, reason=final.reason)

        return GeneratedAnswer(
            answer=_ensure_disclaimer(content),
            confidence=final.level,
            confidence_score=final.score,
            used_passages=passages,
            refused=False,
            reasoning_summary=final.reason,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )


def _refusal(
    passages: list[RetrievedPassage],
    assessment: ConfidenceAssessment,
    *,
    reason: str,
) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer=f"{REFUSAL_TEXT}\n\n{DISCLAIMER}",
        confidence="low",
        confidence_score=assessment.score,
        used_passages=[],
        refused=True,
        reasoning_summary=reason,
    )


def _ensure_disclaimer(content: str) -> str:
    return content if DISCLAIMER in content else f"{content.rstrip()}\n\n{DISCLAIMER}"
