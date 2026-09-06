"""Pydantic schemas for document upload / listing."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class UploadLawResponse(BaseModel):
    document_id: uuid.UUID
    title: str
    doc_type: str
    page_count: int
    chunks_indexed: int
    status: str


class DocumentSummary(BaseModel):
    id: uuid.UUID
    title: str
    doc_type: str
    page_count: int
    status: str
    created_at: datetime


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    version: str = "0.1.0"
    qdrant: str = Field(default="ok", description="'ok' or 'unavailable'")
