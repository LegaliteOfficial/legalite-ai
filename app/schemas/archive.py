"""Response schema for ``POST /archive/analyze``."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ArchiveParty(BaseModel):
    name: str
    role: str = ""


class ArchiveDate(BaseModel):
    date: str
    description: str = ""


class ArchiveAmount(BaseModel):
    amount: str
    description: str = ""


class ArchiveAnalysis(BaseModel):
    title: str
    document_type: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    parties: list[ArchiveParty] = Field(default_factory=list)
    court: str | None = None
    suit_number: str | None = None
    document_date: str | None = None
    key_dates: list[ArchiveDate] = Field(default_factory=list)
    amounts: list[ArchiveAmount] = Field(default_factory=list)
    legal_references: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    is_legal_document: bool = True
    legibility: Literal["good", "partial", "poor"] = "good"
    completeness: Literal["complete", "partial", "unclear"] = "unclear"
    verification_notes: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    # The document's full extracted text (the archived content).
    content: str
    page_count: int | None = None
    word_count: int = 0
    extraction_method: Literal["text", "ocr", "vision", "mixed"] = "text"
    warnings: list[str] = Field(default_factory=list)
    model: str
