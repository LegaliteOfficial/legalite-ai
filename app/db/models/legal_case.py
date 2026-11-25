"""ORM model: a Ghana case-law judgment with extracted structured metadata.

A LegalCase row is created for every ingested document whose detected
``law_type == 'case_law'``. It complements the existing LegalDocument /
LegalChunk pair with court-specific fields used by precedent retrieval,
case similarity, and case-summary endpoints.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import TYPE_CHECKING, Any

from sqlalchemy import Date, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.db.enums import VisibilityScope

if TYPE_CHECKING:
    from app.db.models.case_citation import CaseCitation
    from app.db.models.case_party import CaseParty
    from app.db.models.legal_document import LegalDocument


class LegalCase(Base, TimestampMixin):
    __tablename__ = "legal_cases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("legal_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True,
    )

    case_number: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    case_title: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    court: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    judges: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    decision_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    outcome: Mapped[str | None] = mapped_column(String(256), nullable=True)
    holding: Mapped[str | None] = mapped_column(Text, nullable=True)
    facts_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    legal_topics: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    jurisdiction: Mapped[str] = mapped_column(String(8), nullable=False, default="GH")

    # Free-form bag for fields we discover later without a migration.
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Tenancy mirror (migration 0004). Mirrored from the underlying
    # LegalDocument so the similarity engine filters on a single table.
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    visibility_scope: Mapped[str] = mapped_column(
        String(32), nullable=False, default=VisibilityScope.GLOBAL.value, index=True
    )

    document: Mapped[LegalDocument] = relationship()
    parties: Mapped[list[CaseParty]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
    )
    citing: Mapped[list[CaseCitation]] = relationship(
        back_populates="citing_case",
        cascade="all, delete-orphan",
        foreign_keys="CaseCitation.citing_case_id",
    )
