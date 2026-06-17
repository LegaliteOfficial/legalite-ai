"""Cloud-object-storage key for the original PDF.

Revision ID: 0009_remote_storage
Revises: 0008_query_feedback
Create Date: 2026-05-28

The Q&A pipeline already extracts and chunks each PDF into Postgres +
Qdrant, which is enough to ANSWER questions. But the in-app citation
view needs the ORIGINAL artifact — the judge's signature, court stamp,
page layout — so lawyers can verify what they're reading. Storing only
extracted text is fine for retrieval and lossy for review.

We address that by uploading the original PDF to Supabase Storage at
ingestion time and persisting the resulting object key here. The
``GET /documents/{id}`` endpoint hands the client a short-lived signed
URL derived from this key — never the key itself directly. Backward
compat: the column is NULLABLE because (a) historical rows pre-date
the storage upload, and (b) Supabase credentials might be unset in dev,
in which case the column stays NULL and the endpoint returns text-only.

Why a single column instead of a separate ``document_files`` table:
- Exactly one original PDF per document, today and as far as we can
  see. No 1:N modelling needed.
- Querying "give me the file for this doc" is the only access pattern,
  and it's already a single-row lookup.
- If we ever support multiple originals (e.g. translated copies) we
  spin a separate table THEN — premature normalization is just empty
  joins.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_remote_storage"
down_revision: str | Sequence[str] | None = "0008_query_feedback"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "legal_documents",
        sa.Column("remote_storage_key", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("legal_documents", "remote_storage_key")
