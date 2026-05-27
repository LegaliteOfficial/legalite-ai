"""Pydantic schemas for the case-intelligence endpoints."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

PartyRole = Literal[
    "plaintiff", "defendant", "appellant", "respondent", "applicant", "intervener", "unknown"
]


class PartyOut(BaseModel):
    role: PartyRole
    party_name: str


class CaseSummaryOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    case_title: str
    case_number: str | None = None
    court: str | None = None
    judges: list[str] = Field(default_factory=list)
    decision_date: date | None = None
    outcome: str | None = None
    legal_topics: list[str] = Field(default_factory=list)
    jurisdiction: str = "GH"
    created_at: datetime


class CaseDetailOut(CaseSummaryOut):
    holding: str | None = None
    facts_summary: str | None = None
    parties: list[PartyOut] = Field(default_factory=list)


# ---- /cases/similar ----------------------------------------------------------


class CaseSimilarRequest(BaseModel):
    """Either ``case_id`` (find similar to existing) or ``query`` (find by fact pattern)."""

    case_id: uuid.UUID | None = Field(
        default=None,
        description="If set, find cases similar to this case.",
    )
    query: str | None = Field(
        default=None,
        min_length=10,
        max_length=4000,
        description="Free-text fact pattern to match against case law.",
    )
    top_k: int | None = Field(default=None, ge=1, le=25)


class CaseSimilarHitOut(BaseModel):
    case: CaseSummaryOut
    score: float
    semantic_score: float
    topic_overlap: int
    shared_topics: list[str] = Field(default_factory=list)
    matched_chunks: int


class CaseSimilarResponse(BaseModel):
    query_topics: list[str] = Field(default_factory=list)
    hits: list[CaseSimilarHitOut]


# ---- /cases/summarize --------------------------------------------------------


class CaseSummarizeRequest(BaseModel):
    case_id: uuid.UUID


class CaseSummarizeResponse(BaseModel):
    case: CaseDetailOut
    disclaimer: str = (
        "This information is for educational purposes and not legal advice."
    )


# ---- /cases/compare ---------------------------------------------------------


class CaseCompareRequest(BaseModel):
    case_ids: list[uuid.UUID] = Field(..., min_length=2, max_length=4)


class CaseComparisonField(BaseModel):
    field: str
    values: dict[str, str | None]  # case_id (stringified) -> field value


class CaseCompareResponse(BaseModel):
    cases: list[CaseDetailOut]
    differences: list[CaseComparisonField]
    overlap: list[str]  # topics shared by ALL cases compared
    disclaimer: str = (
        "This information is for educational purposes and not legal advice."
    )


# ---- GET /cases listing -----------------------------------------------------


class CaseListResponse(BaseModel):
    cases: list[CaseSummaryOut]
    limit: int
    offset: int


# ---- GET /cases/{id}/citations ---------------------------------------------


class CitationGraphNode(BaseModel):
    case_id: uuid.UUID
    case_title: str
    case_number: str | None = None
    court: str | None = None
    confidence: float


class CitationGraphResponse(BaseModel):
    cites: list[CitationGraphNode] = Field(default_factory=list)
    cited_by: list[CitationGraphNode] = Field(default_factory=list)
    precedent_chain: list[CitationGraphNode] = Field(default_factory=list)
