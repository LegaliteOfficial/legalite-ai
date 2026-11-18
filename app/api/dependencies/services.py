"""FastAPI dependency providers for service objects.

Routes depend on these — never on concrete classes — so we can substitute
stubs in tests via `app.dependency_overrides`.
"""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.case_intelligence_service import CaseIntelligenceService
from app.services.drafting_service import DraftingService, HybridAuthorityRetriever
from app.services.ingestion_service import IngestionService
from app.services.qa_service import QAService


def get_ingestion_service(db: Session = Depends(get_db)) -> IngestionService:
    return IngestionService(db)


def get_qa_service(db: Session = Depends(get_db)) -> QAService:
    return QAService(db)


def get_case_intelligence_service(
    db: Session = Depends(get_db),
) -> CaseIntelligenceService:
    return CaseIntelligenceService(db)


def get_drafting_service(db: Session = Depends(get_db)) -> DraftingService:
    # The retriever is built lazily: requests with use_authorities=false,
    # and every unit test, never touch the vector store.
    return DraftingService(retriever_factory=lambda: HybridAuthorityRetriever(db))
