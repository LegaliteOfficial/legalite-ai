"""ORM model: one structurally-aware chunk of a legal document.

The chunk text and embedding live in Qdrant; Postgres holds the metadata,
hierarchy, and back-reference so we can audit, re-index, or delete by source.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.db.enums import VisibilityScope

if TYPE_CHECKING:
    from app.db.models.legal_document import LegalDocument


class LegalChunk(Base, TimestampMixin):
    __tablename__ = "legal_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("legal_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Legal hierarchy (any may be null)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    chapter: Mapped[str | None] = mapped_column(String(128), nullable=True)
    part: Mapped[str | None] = mapped_column(String(128), nullable=True)
    article: Mapped[str | None] = mapped_column(String(128), nullable=True)
    section: Mapped[str | None] = mapped_column(String(128), nullable=True)
    clause: Mapped[str | None] = mapped_column(String(128), nullable=True)
    subsection: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Extended legal metadata (mirrored from parent document where applicable)
    law_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    act_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    effective_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    legal_topic: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Qdrant point id (UUID string) — lets us map a vector hit back to this row
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # Tenancy mirror (migration 0004). Duplicated from the parent document so
    # retrieval filters never need a join on the hot path.
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    visibility_scope: Mapped[str] = mapped_column(
        String(32), nullable=False, default=VisibilityScope.GLOBAL.value, index=True
    )
    document_origin: Mapped[str | None] = mapped_column(String(64), nullable=True)

    document: Mapped[LegalDocument] = relationship(back_populates="chunks")
