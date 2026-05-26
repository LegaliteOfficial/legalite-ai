"""ORM model: a party named in a legal case (plaintiff, defendant, etc.)."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.legal_case import LegalCase


class CaseParty(Base, TimestampMixin):
    __tablename__ = "case_parties"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("legal_cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # plaintiff | defendant | appellant | respondent | applicant | intervener | unknown
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    party_name: Mapped[str] = mapped_column(String(512), nullable=False)

    case: Mapped[LegalCase] = relationship(back_populates="parties")
