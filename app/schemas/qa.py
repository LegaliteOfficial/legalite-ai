"""Pydantic schemas for the Q&A endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]


class CitationSpan(BaseModel):
    """Character offsets used by the UI to highlight the source PDF span.

    ``start_char`` / ``end_char`` are zero-indexed character offsets within
    the page text of the citation's ``page_number``. Both NULL when the
    chunk was ingested before migration 0005.
    """

    start_char: int | None = None
    end_char: int | None = None


class Citation(BaseModel):
    # Legacy fields — kept for backward compatibility with v1 clients
    law: str = ""
    article: str = ""
    section: str = ""
    page: str = ""
    # v2 fields — populated when available, default empty for compat
    law_name: str = ""
    law_type: str = ""
    act_number: str = ""
    chapter: str = ""
    part: str = ""
    clause: str = ""
    subsection: str = ""
    page_number: int | None = None
    relevance: float = 0.0
    # v3 (Phase 5 M1): source provenance — global vs tenant-private.
    source_scope: Literal["global", "tenant_private"] = "global"
    document_origin: str | None = None
    # Phase 5 M9 — exact span for UI highlighting.
    citation_span: CitationSpan | None = None


class SourceUsed(BaseModel):
    """A retrieved passage referenced in the answer (more detail than Citation)."""

    law_name: str = ""
    law_type: str = ""
    section: str = ""
    article: str = ""
    page: str = ""
    excerpt: str = ""
    score: float = 0.0
    source_scope: Literal["global", "tenant_private"] = "global"
    document_origin: str | None = None


# ---------------------------------------------------------------------------
# Structured answer (Phase 5 — Milestone 1)
# ---------------------------------------------------------------------------


class StructuredCitation(BaseModel):
    """One row in any of the structured-answer law / case lists.

    ``citation_id`` is the passage number (1-indexed) the LLM was given in
    the prompt; the citation formatter maps it back to a full Citation
    record using the same passages.
    """

    citation_id: int
    summary: str = ""
    law_name: str = ""
    section: str = ""
    article: str = ""
    case_title: str = ""
    court: str = ""


class StructuredAnswer(BaseModel):
    """7-section legal answer surface.

    Always present in /ask responses (Phase 5). On refusal, ``direct_answer``
    holds the refusal sentence and every list is empty.
    """

    direct_answer: str
    applicable_law: list[StructuredCitation] = Field(default_factory=list)
    relevant_public_cases: list[StructuredCitation] = Field(default_factory=list)
    firm_similar_cases: list[StructuredCitation] = Field(default_factory=list)
    legal_reasoning: str = ""
    confidence_assessment: str = ""
    citations: list[int] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: str | None = Field(default=None, description="Optional chat session UUID for memory")


class AskResponse(BaseModel):
    # v1 fields (unchanged)
    answer: str
    citations: list[Citation]
    confidence: Confidence
    disclaimer: str = (
        "This information is for educational purposes and not legal advice."
    )
    # v2 additions
    sources_used: list[SourceUsed] = Field(default_factory=list)
    reasoning_summary: str = ""
    session_id: str | None = None
    # v3 (Phase 5 M1): structured answer surface. Always present.
    structured_answer: StructuredAnswer | None = None
    # v3.1 (Phase 5 M2): which intent the query planner inferred. Surfaced for
    # observability only — does not change retrieval semantics for clients.
    query_intent: str | None = None
    query_intent_confidence: float | None = None
    # Phase B (self-learning loop): the assistant turn's persistent
    # ``chat_messages.id``. Clients use it to attach feedback via
    # ``POST /ask/{message_id}/feedback``. Optional so legacy /ask
    # callers that don't care about feedback can keep ignoring it.
    message_id: str | None = None
