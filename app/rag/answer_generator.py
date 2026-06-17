"""Compose the legal answer using a single structured-output LLM call.

Pipeline (Phase 5 — Milestone 1):

1. ``ConfidenceEngine.assess_before_generation`` gates the LLM call. If
   retrieval is weak, we refuse without calling the model.
2. The retrieved passages are annotated with authority tiers and the
   prompt renders them grouped by tier so the model can apply the
   hierarchy of authority.
3. ONE chat-completion call with ``response_format={"type":"json_object"}``
   returns a JSON object matching ``StructuredAnswer``.
4. We parse the JSON defensively. Any failure ⇒ refusal; we never let
   malformed output surface to the client.
5. ``ConfidenceEngine.assess_final`` re-checks grounding (inline [n]
   citations in legal_reasoning) and consistency.
6. We synthesize the legacy ``answer`` string from the structured payload
   so the existing /ask contract continues to work.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from openai import OpenAI

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.telemetry import llm_latency, observe
from app.rag.authority import annotate as annotate_authority
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
from app.schemas.qa import StructuredAnswer, StructuredCitation

logger = get_logger(__name__)


@dataclass(slots=True)
class GeneratedAnswer:
    answer: str
    confidence: Confidence
    confidence_score: float
    used_passages: list[RetrievedPassage]
    refused: bool
    reasoning_summary: str
    structured: StructuredAnswer = field(
        default_factory=lambda: StructuredAnswer(direct_answer="")
    )
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0


class AnswerGenerator:
    def __init__(
        self,
        *,
        confidence_engine: ConfidenceEngine | None = None,
        client: OpenAI | None = None,
    ) -> None:
        settings = get_settings()
        self._client = client or OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_llm_model
        self._confidence = confidence_engine or ConfidenceEngine()

    def generate(
        self,
        question: str,
        passages: list[RetrievedPassage],
        *,
        memory: list[MemoryTurn] | None = None,
    ) -> GeneratedAnswer:
        # Stamp authority metadata on each passage before the prompt sees them.
        annotate_authority(passages)

        pre = self._confidence.assess_before_generation(passages)
        if pre.refused:
            return _refusal(passages, pre, reason=pre.reason)

        user_prompt = build_user_prompt(question, passages, memory=memory)
        t0 = time.perf_counter()
        try:
            with observe(llm_latency):
                completion = self._client.chat.completions.create(
                    model=self._model,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
        except Exception:
            logger.exception("answer.llm_failed", model=self._model)
            return _refusal(passages, pre, reason="LLM call failed")
        latency_ms = int((time.perf_counter() - t0) * 1000)

        raw_content = (completion.choices[0].message.content or "").strip()
        usage = completion.usage
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0

        structured, parse_error = _parse_structured(raw_content)
        if structured is None:
            logger.warning(
                "answer.parse_failed",
                model=self._model,
                reason=parse_error,
                preview=raw_content[:200],
            )
            return _refusal(
                passages,
                pre,
                reason=f"LLM output not valid structured JSON: {parse_error}",
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

        # Phase B — enrich each structured citation with the source
        # document_id / chunk_id / page_number from its underlying
        # passage. The LLM emits ``citation_id`` as the 1-indexed
        # passage position from the prompt; we round-trip that back to
        # the actual passage object so the frontend can open the
        # preview drawer directly. Out-of-range ids (model
        # hallucination) are left with their nullable fields untouched.
        _enrich_structured_citations(structured, passages)

        # Model explicitly refused → honor it and keep arrays empty.
        if REFUSAL_TEXT in structured.direct_answer:
            return GeneratedAnswer(
                answer=_synthesize_legacy_answer(_empty_refusal(structured.direct_answer)),
                confidence="low",
                confidence_score=pre.score,
                used_passages=[],
                refused=True,
                reasoning_summary="LLM declined: insufficient evidence in provided passages",
                structured=_empty_refusal(structured.direct_answer),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
            )

        # Build the legacy answer string before the post-LLM grounding check
        # so the grounding scan sees the same [n] markers the model produced.
        legacy_answer = _synthesize_legacy_answer(structured)

        final = self._confidence.assess_final(passages, legacy_answer, pre)
        if final.refused:
            return _refusal(
                passages,
                final,
                reason=final.reason,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

        return GeneratedAnswer(
            answer=legacy_answer,
            confidence=final.level,
            confidence_score=final.score,
            used_passages=passages,
            refused=False,
            reasoning_summary=final.reason,
            structured=structured,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_structured(raw: str) -> tuple[StructuredAnswer | None, str | None]:
    """Defensive JSON → StructuredAnswer. Never raises."""
    if not raw:
        return None, "empty response"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"json decode: {e.msg}"
    if not isinstance(data, dict):
        return None, "top-level JSON is not an object"

    # Coerce lists / strings rather than reject — the model may emit minor
    # shape drift (e.g. a string when a list is expected). We never invent
    # content; we only normalize types.
    def _str(key: str) -> str:
        v = data.get(key)
        return v if isinstance(v, str) else ""

    def _ints(key: str) -> list[int]:
        v = data.get(key) or []
        if not isinstance(v, list):
            return []
        out: list[int] = []
        for item in v:
            if isinstance(item, int):
                out.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                out.append(int(item.strip()))
        return out

    def _cites(key: str) -> list[StructuredCitation]:
        v = data.get(key) or []
        if not isinstance(v, list):
            return []
        out: list[StructuredCitation] = []
        for item in v:
            if not isinstance(item, dict):
                continue
            try:
                out.append(
                    StructuredCitation(
                        citation_id=int(item.get("citation_id", 0)) or 0,
                        summary=str(item.get("summary") or "").strip(),
                        law_name=str(item.get("law_name") or "").strip(),
                        section=str(item.get("section") or "").strip(),
                        article=str(item.get("article") or "").strip(),
                        case_title=str(item.get("case_title") or "").strip(),
                        court=str(item.get("court") or "").strip(),
                    )
                )
            except (TypeError, ValueError):
                continue
        # Drop any citation row that didn't carry a usable passage id.
        return [c for c in out if c.citation_id > 0]

    direct = _str("direct_answer").strip()
    if not direct:
        return None, "direct_answer missing"

    return (
        StructuredAnswer(
            direct_answer=direct,
            applicable_law=_cites("applicable_law"),
            relevant_public_cases=_cites("relevant_public_cases"),
            firm_similar_cases=_cites("firm_similar_cases"),
            legal_reasoning=_str("legal_reasoning"),
            confidence_assessment=_str("confidence_assessment"),
            citations=_ints("citations"),
        ),
        None,
    )


def _synthesize_legacy_answer(s: StructuredAnswer) -> str:
    """Render the structured answer back into the v1/v2 ``answer`` string.

    The legacy contract was: a free-text answer ending with the disclaimer.
    We rebuild that shape so existing /ask clients see no breaking change.
    """
    parts: list[str] = [s.direct_answer.strip()]
    if s.legal_reasoning.strip():
        parts.append(s.legal_reasoning.strip())

    citation_lines: list[str] = []
    for row in s.applicable_law:
        bits: list[str] = []
        if row.law_name:
            bits.append(row.law_name)
        if row.section:
            bits.append(f"s.{row.section}")
        if row.article:
            bits.append(f"art.{row.article}")
        label = " | ".join(bits) or row.summary or ""
        if label:
            citation_lines.append(f"- [{row.citation_id}] {label}")
    for row in s.relevant_public_cases:
        label = row.case_title or row.summary or ""
        if label:
            citation_lines.append(f"- [{row.citation_id}] {label}")
    for row in s.firm_similar_cases:
        label = row.case_title or row.summary or ""
        if label:
            citation_lines.append(f"- [{row.citation_id}] (firm) {label}")

    if citation_lines:
        parts.append("Citations:\n" + "\n".join(citation_lines))
    if s.confidence_assessment.strip():
        parts.append(f"Confidence note: {s.confidence_assessment.strip()}")
    parts.append(DISCLAIMER)
    return "\n\n".join(parts).strip()


def _enrich_structured_citations(
    structured: StructuredAnswer,
    passages: list[RetrievedPassage],
) -> None:
    """Backfill ``document_id`` / ``chunk_id`` / ``page_number`` onto
    each StructuredCitation by looking up the underlying passage.

    The LLM's ``citation_id`` is the 1-indexed position in the prompt's
    passages list — see ``build_user_prompt``. We round-trip that index
    back to the passage object so the front-end can open the source
    preview directly from a structured row without doing its own index
    arithmetic. Out-of-range ids (model hallucination) are left
    untouched: the nullable fields stay None and the UI degrades to
    "click does nothing" rather than rendering a broken link.

    Mutates ``structured`` in place. Safe to call multiple times — it
    only writes when the fields are unset.
    """
    for group in (
        structured.applicable_law,
        structured.relevant_public_cases,
        structured.firm_similar_cases,
    ):
        for cite in group:
            idx = cite.citation_id - 1
            if idx < 0 or idx >= len(passages):
                continue
            p = passages[idx]
            if cite.document_id is None:
                cite.document_id = p.document_id
            if cite.chunk_id is None:
                cite.chunk_id = p.chunk_id
            if cite.page_number is None:
                cite.page_number = p.page_start


def _empty_refusal(refusal_text: str) -> StructuredAnswer:
    """Build the canonical empty structured answer used on refusal."""
    return StructuredAnswer(
        direct_answer=refusal_text or REFUSAL_TEXT,
        applicable_law=[],
        relevant_public_cases=[],
        firm_similar_cases=[],
        legal_reasoning=(
            "No passages in the supplied Ghana legal materials supported a confident answer."
        ),
        confidence_assessment="Insufficient evidence — system refused to speculate.",
        citations=[],
    )


def _refusal(
    passages: list[RetrievedPassage],
    assessment: ConfidenceAssessment,
    *,
    reason: str,
    latency_ms: int = 0,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> GeneratedAnswer:
    structured = _empty_refusal(REFUSAL_TEXT)
    return GeneratedAnswer(
        answer=f"{REFUSAL_TEXT}\n\n{DISCLAIMER}",
        confidence="low",
        confidence_score=assessment.score,
        used_passages=[],
        refused=True,
        reasoning_summary=reason,
        structured=structured,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )
