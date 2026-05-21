"""FastAPI dependency providers for service objects.

Routes depend on these — never on concrete classes — so we can substitute
stubs in tests via `app.dependency_overrides`.
"""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.ingestion_service import IngestionService
from app.services.qa_service import QAService


def get_ingestion_service(db: Session = Depends(get_db)) -> IngestionService:
    return IngestionService(db)


def get_qa_service(db: Session = Depends(get_db)) -> QAService:
    return QAService(db)
