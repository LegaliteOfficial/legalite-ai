"""Pydantic schemas for the Q&A endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]


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


class SourceUsed(BaseModel):
    """A retrieved passage referenced in the answer (more detail than Citation)."""

    law_name: str = ""
    law_type: str = ""
    section: str = ""
    article: str = ""
    page: str = ""
    excerpt: str = ""
    score: float = 0.0


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
    # v2 additions (optional; absent in older clients' usage but always populated)
    sources_used: list[SourceUsed] = Field(default_factory=list)
    reasoning_summary: str = ""
    session_id: str | None = None
