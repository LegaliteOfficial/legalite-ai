"""ORM model: a single uploaded Ghana legal document (PDF)."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.db.enums import ConfidentialityLevel, VisibilityScope

if TYPE_CHECKING:
    from app.db.models.legal_chunk import LegalChunk


class LegalDocument(Base, TimestampMixin):
    __tablename__ = "legal_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    # e.g. "Constitution", "Act", "Regulation", "Case Law"
    doc_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Normalized law category (constitution | act | regulation | case_law | bill | ...)
    law_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    act_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    effective_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Original filename uploaded
    source_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    # On-disk path under UPLOAD_DIR — the working copy the worker reads.
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    # Cloud-object-storage key (Supabase Storage). NULL when storage was
    # not configured at ingest time or when the upload step failed. The
    # /documents/{id} endpoint converts this into a short-lived signed
    # URL — clients never see the raw key.
    remote_storage_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # ingested | failed | processing
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="processing")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Set when doc_type was detected from content rather than supplied by
    # the uploader (migration 0010). ``doc_type_source`` is llm | heuristic.
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    doc_type_source: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Tenancy + classification (migration 0004)
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    visibility_scope: Mapped[str] = mapped_column(
        String(32), nullable=False, default=VisibilityScope.GLOBAL.value, index=True
    )
    document_origin: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    confidentiality_level: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ConfidentialityLevel.PUBLIC.value
    )
    uploaded_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )

    chunks: Mapped[list[LegalChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
