"""Repository for legal cases, parties, and the citation graph.

Mirrors `DocumentRepository` in style: services pass plain dataclasses in,
the repo handles ORM concerns. Keeps the case-intelligence services free of
SQLAlchemy imports and trivially testable with a real Postgres in CI.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, selectinload

from app.db.enums import VisibilityScope
from app.db.models.case_citation import CaseCitation
from app.db.models.case_party import CaseParty
from app.db.models.legal_case import LegalCase


@dataclass(slots=True)
class CasePartyRecord:
    role: str
    party_name: str


@dataclass(slots=True)
class CaseCitationRecord:
    cited_raw_text: str
    cited_case_title: str | None = None
    cited_reporter: str | None = None
    cited_case_id: uuid.UUID | None = None


@dataclass(slots=True)
class CaseRecord:
    """Plain payload for creating a LegalCase. Mirrors model fields one-to-one."""

    document_id: uuid.UUID
    case_title: str
    case_number: str | None = None
    court: str | None = None
    judges: list[str] | None = None
    decision_date: date | None = None
    outcome: str | None = None
    holding: str | None = None
    facts_summary: str | None = None
    legal_topics: list[str] | None = None
    jurisdiction: str = "GH"
    organization_id: uuid.UUID | None = None
    visibility_scope: str = VisibilityScope.GLOBAL.value
    parties: list[CasePartyRecord] = field(default_factory=list)
    citations: list[CaseCitationRecord] = field(default_factory=list)


class CaseRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    # ---- writes ------------------------------------------------------------

    def create_case(self, record: CaseRecord) -> LegalCase:
        case = LegalCase(
            document_id=record.document_id,
            case_title=record.case_title,
            case_number=record.case_number,
            court=record.court,
            judges=record.judges,
            decision_date=record.decision_date,
            outcome=record.outcome,
            holding=record.holding,
            facts_summary=record.facts_summary,
            legal_topics=record.legal_topics,
            jurisdiction=record.jurisdiction,
            organization_id=record.organization_id,
            visibility_scope=record.visibility_scope,
        )
        self._db.add(case)
        self._db.flush()

        if record.parties:
            self._db.add_all(
                [
                    CaseParty(case_id=case.id, role=p.role, party_name=p.party_name)
                    for p in record.parties
                ]
            )
        if record.citations:
            self._db.add_all(
                [
                    CaseCitation(
                        citing_case_id=case.id,
                        cited_case_id=c.cited_case_id,
                        cited_raw_text=c.cited_raw_text,
                        cited_case_title=c.cited_case_title,
                        cited_reporter=c.cited_reporter,
                    )
                    for c in record.citations
                ]
            )
        self._db.flush()
        return case

    def attach_citations(
        self, case_id: uuid.UUID, citations: Iterable[CaseCitationRecord]
    ) -> None:
        """Append citations to an existing case, ignoring duplicates (unique edge)."""
        existing = {
            row[0]
            for row in self._db.execute(
                select(CaseCitation.cited_raw_text).where(
                    CaseCitation.citing_case_id == case_id
                )
            ).all()
        }
        rows = [
            CaseCitation(
                citing_case_id=case_id,
                cited_case_id=c.cited_case_id,
                cited_raw_text=c.cited_raw_text,
                cited_case_title=c.cited_case_title,
                cited_reporter=c.cited_reporter,
            )
            for c in citations
            if c.cited_raw_text not in existing
        ]
        if rows:
            self._db.add_all(rows)
            self._db.flush()

    # ---- reads -------------------------------------------------------------

    def get_case(
        self,
        case_id: uuid.UUID,
        *,
        organization_id: uuid.UUID | None = None,
    ) -> LegalCase | None:
        stmt = (
            select(LegalCase)
            .where(LegalCase.id == case_id)
            .options(
                selectinload(LegalCase.parties),
                selectinload(LegalCase.citing),
            )
        )
        stmt = _apply_tenant_filter(stmt, organization_id)
        return self._db.scalar(stmt)

    def get_case_by_document(
        self,
        document_id: uuid.UUID,
        *,
        organization_id: uuid.UUID | None = None,
    ) -> LegalCase | None:
        stmt = select(LegalCase).where(LegalCase.document_id == document_id)
        stmt = _apply_tenant_filter(stmt, organization_id)
        return self._db.scalar(stmt)

    def list_cases(
        self,
        *,
        court: str | None = None,
        topic: str | None = None,
        limit: int = 50,
        offset: int = 0,
        organization_id: uuid.UUID | None = None,
    ) -> list[LegalCase]:
        stmt = select(LegalCase).order_by(
            LegalCase.decision_date.desc().nulls_last(),
            LegalCase.created_at.desc(),
        )
        if court:
            stmt = stmt.where(LegalCase.court == court)
        if topic:
            # JSONB containment: legal_topics ? 'topic'
            stmt = stmt.where(LegalCase.legal_topics.op("?")(topic))
        stmt = _apply_tenant_filter(stmt, organization_id)
        stmt = stmt.limit(limit).offset(offset)
        return list(self._db.scalars(stmt).all())

    def get_cases(
        self,
        case_ids: Sequence[uuid.UUID],
        *,
        organization_id: uuid.UUID | None = None,
    ) -> list[LegalCase]:
        if not case_ids:
            return []
        stmt = (
            select(LegalCase)
            .where(LegalCase.id.in_(list(case_ids)))
            .options(selectinload(LegalCase.parties))
        )
        stmt = _apply_tenant_filter(stmt, organization_id)
        return list(self._db.scalars(stmt).all())

    def search_cases_by_text(
        self,
        query: str,
        limit: int = 20,
        *,
        organization_id: uuid.UUID | None = None,
    ) -> list[tuple[LegalCase, float]]:
        """Lexical search over case_title + holding + facts_summary using `ILIKE`.

        Kept intentionally simple: complementary to vector retrieval, not a
        replacement. The strong recall path is the existing chunk-level FTS in
        ``DocumentRepository.keyword_search``; this is for top-of-page browsing.
        """
        if not query.strip():
            return []
        pattern = f"%{query.strip()}%"
        stmt = select(LegalCase).where(
            or_(
                LegalCase.case_title.ilike(pattern),
                LegalCase.holding.ilike(pattern),
                LegalCase.facts_summary.ilike(pattern),
            )
        )
        stmt = _apply_tenant_filter(stmt, organization_id).limit(limit)
        cases = list(self._db.scalars(stmt).all())
        # Pseudo-score: title hits > holding > facts.
        scored: list[tuple[LegalCase, float]] = []
        q = query.lower()
        for c in cases:
            s = 0.0
            if q in (c.case_title or "").lower():
                s += 1.0
            if q in (c.holding or "").lower():
                s += 0.5
            if q in (c.facts_summary or "").lower():
                s += 0.25
            scored.append((c, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ---------------------------------------------------------------------------
# Tenant filter — shared between every read on this repo.
# ---------------------------------------------------------------------------


def _apply_tenant_filter(stmt, organization_id):  # type: ignore[no-untyped-def]
    """Apply the visibility scope clause that mirrors the Qdrant filter.

    - ``organization_id is None``: only GLOBAL cases.
    - ``organization_id`` set: GLOBAL cases OR cases belonging to that org
      (with visibility_scope = tenant_private).
    """
    if organization_id is None:
        return stmt.where(LegalCase.visibility_scope == VisibilityScope.GLOBAL.value)
    return stmt.where(
        or_(
            LegalCase.visibility_scope == VisibilityScope.GLOBAL.value,
            (LegalCase.visibility_scope == VisibilityScope.TENANT_PRIVATE.value)
            & (LegalCase.organization_id == organization_id),
        )
    )

