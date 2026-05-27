"""Eager imports of all ORM models.

Importing this package (e.g. `import app.db.models`) registers every model on
`Base.metadata`, which is what Alembic autogenerate needs.
"""

from app.db.models.case_citation import CaseCitation
from app.db.models.case_party import CaseParty
from app.db.models.chat import ChatMessage, ChatSession
from app.db.models.ingestion_job import IngestionJob
from app.db.models.legal_case import LegalCase
from app.db.models.legal_chunk import LegalChunk
from app.db.models.legal_document import LegalDocument

__all__ = [
    "CaseCitation",
    "CaseParty",
    "ChatMessage",
    "ChatSession",
    "IngestionJob",
    "LegalCase",
    "LegalChunk",
    "LegalDocument",
]
