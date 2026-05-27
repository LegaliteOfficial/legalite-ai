"""ORM model: an inline case-to-case citation extracted from a judgment.

If the cited case is itself ingested in our corpus, ``cited_case_id`` is
resolved (lazily, by a separate matcher pass). Otherwise we keep the raw
citation text and best-effort title so future ingestions can backfill the
edge in the citation graph.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.legal_case import LegalCase


class CaseCitation(Base, TimestampMixin):
    __tablename__ = "case_citations"
    __table_args__ = (
        UniqueConstraint(
            "citing_case_id", "cited_raw_text", name="uq_case_citations_edge"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    citing_case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("legal_cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    cited_case_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("legal_cases.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    cited_raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    cited_case_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    cited_reporter: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Phase 5 M5 — populated by CitationResolutionService.
    resolution_confidence: Mapped[Decimal | None] = mapped_column(
        Numeric(precision=4, scale=3), nullable=True
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    citing_case: Mapped[LegalCase] = relationship(
        back_populates="citing",
        foreign_keys=[citing_case_id],
    )
    cited_case: Mapped[LegalCase | None] = relationship(
        foreign_keys=[cited_case_id],
    )
