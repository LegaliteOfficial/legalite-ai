"""Auto-classified document type + summary.

Revision ID: 0010_doc_classification
Revises: 0009_remote_storage
Create Date: 2026-10-16

``POST /upload-law/bulk`` no longer requires a ``doc_type``: when it is
omitted, the worker classifies each PDF from its content (see
``app/ingestion/document_classifier.py``). These columns keep what the
classifier produced so the upload UI can show it back:

- ``summary`` — one-to-two sentence description (LLM path only).
- ``doc_type_source`` — ``llm`` | ``heuristic``; NULL when the uploader
  chose the type, which is every row that predates this migration.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_doc_classification"
down_revision: str | Sequence[str] | None = "0009_remote_storage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("legal_documents", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column(
        "legal_documents",
        sa.Column("doc_type_source", sa.String(length=16), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("legal_documents", "doc_type_source")
    op.drop_column("legal_documents", "summary")
