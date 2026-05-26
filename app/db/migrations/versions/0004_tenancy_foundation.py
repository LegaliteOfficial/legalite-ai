"""Tenancy foundation: visibility scope + organization_id + document_origin.

Revision ID: 0004_tenancy_foundation
Revises: 0003_case_intelligence
Create Date: 2026-05-26

Adds the columns required to enforce hard knowledge-isolation boundaries
between the GLOBAL Ghana legal corpus and TENANT_PRIVATE firm material.

Design notes
------------
- ``organization_id`` is stored as a plain ``uuid`` column with **no FK**.
  Users / organizations live in the separate ``legalite-backend`` service
  (NestJS) which co-tenants the same Postgres database. Coupling our
  migrations to the backend's tables would create an ordering hazard while
  the backend schema is still in flux.

- Enums are stored as ``VARCHAR`` rather than native Postgres ENUM types so
  adding a new ``document_origin`` value (e.g. ``litigation_brief``) never
  requires another DDL migration. The application layer
  (``app/db/enums.py``) is the source of truth.

- Existing rows are backfilled to ``visibility_scope = 'global'`` and
  ``organization_id = NULL`` so legacy data continues to surface in /ask
  results without any code change at the client.

- Mirrored columns on ``legal_chunks`` and ``legal_cases`` keep retrieval
  filters single-table (no join on hot path). ``chat_sessions`` gets
  ``user_id`` too because the backend's session-listing UI will scope by
  user, not just org.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_tenancy_foundation"
down_revision: str | Sequence[str] | None = "0003_case_intelligence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Tables that get the (organization_id, visibility_scope) pair.
_SCOPED_TABLES: tuple[str, ...] = (
    "legal_documents",
    "legal_chunks",
    "legal_cases",
    "chat_sessions",
)


def upgrade() -> None:
    # ---- legal_documents ---------------------------------------------------
    op.add_column(
        "legal_documents",
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "legal_documents",
        sa.Column(
            "visibility_scope",
            sa.String(length=32),
            nullable=False,
            server_default="global",
        ),
    )
    op.add_column(
        "legal_documents",
        sa.Column("document_origin", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "legal_documents",
        sa.Column(
            "confidentiality_level",
            sa.String(length=32),
            nullable=False,
            server_default="public",
        ),
    )
    op.add_column(
        "legal_documents",
        sa.Column("uploaded_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index("ix_legal_documents_organization_id", "legal_documents", ["organization_id"])
    op.create_index(
        "ix_legal_documents_visibility_scope", "legal_documents", ["visibility_scope"]
    )
    op.create_index("ix_legal_documents_document_origin", "legal_documents", ["document_origin"])
    # Compound index — the most common retrieval filter pattern.
    op.create_index(
        "ix_legal_documents_scope_org",
        "legal_documents",
        ["visibility_scope", "organization_id"],
    )

    # ---- legal_chunks (mirrored for filter-on-the-hot-path) ---------------
    op.add_column(
        "legal_chunks",
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "legal_chunks",
        sa.Column(
            "visibility_scope",
            sa.String(length=32),
            nullable=False,
            server_default="global",
        ),
    )
    op.add_column(
        "legal_chunks",
        sa.Column("document_origin", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_legal_chunks_organization_id", "legal_chunks", ["organization_id"]
    )
    op.create_index(
        "ix_legal_chunks_visibility_scope", "legal_chunks", ["visibility_scope"]
    )
    op.create_index(
        "ix_legal_chunks_scope_org",
        "legal_chunks",
        ["visibility_scope", "organization_id"],
    )

    # ---- legal_cases ------------------------------------------------------
    op.add_column(
        "legal_cases",
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "legal_cases",
        sa.Column(
            "visibility_scope",
            sa.String(length=32),
            nullable=False,
            server_default="global",
        ),
    )
    op.create_index(
        "ix_legal_cases_organization_id", "legal_cases", ["organization_id"]
    )
    op.create_index(
        "ix_legal_cases_visibility_scope", "legal_cases", ["visibility_scope"]
    )

    # ---- chat_sessions ----------------------------------------------------
    op.add_column(
        "chat_sessions",
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "chat_sessions",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "chat_sessions",
        sa.Column(
            "visibility_scope",
            sa.String(length=32),
            nullable=False,
            server_default="global",
        ),
    )
    op.create_index(
        "ix_chat_sessions_organization_id", "chat_sessions", ["organization_id"]
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])

    # ---- backfill ---------------------------------------------------------
    # Every existing row predates tenancy — they all belong to the global corpus.
    # The server_default already gave them 'global'; this is belt-and-braces
    # for environments where the default did not fire (e.g. partial migrations).
    for table in _SCOPED_TABLES:
        op.execute(
            f"UPDATE {table} SET visibility_scope = 'global' "
            f"WHERE visibility_scope IS NULL"
        )

    # Drop the server defaults now that the backfill is done. We want new
    # inserts to be explicit about visibility — defaults can mask bugs where
    # a private upload accidentally falls through to GLOBAL.
    for table in _SCOPED_TABLES:
        op.alter_column(table, "visibility_scope", server_default=None)
    op.alter_column("legal_documents", "confidentiality_level", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_chat_sessions_user_id", table_name="chat_sessions")
    op.drop_index("ix_chat_sessions_organization_id", table_name="chat_sessions")
    op.drop_column("chat_sessions", "visibility_scope")
    op.drop_column("chat_sessions", "user_id")
    op.drop_column("chat_sessions", "organization_id")

    op.drop_index("ix_legal_cases_visibility_scope", table_name="legal_cases")
    op.drop_index("ix_legal_cases_organization_id", table_name="legal_cases")
    op.drop_column("legal_cases", "visibility_scope")
    op.drop_column("legal_cases", "organization_id")

    op.drop_index("ix_legal_chunks_scope_org", table_name="legal_chunks")
    op.drop_index("ix_legal_chunks_visibility_scope", table_name="legal_chunks")
    op.drop_index("ix_legal_chunks_organization_id", table_name="legal_chunks")
    op.drop_column("legal_chunks", "document_origin")
    op.drop_column("legal_chunks", "visibility_scope")
    op.drop_column("legal_chunks", "organization_id")

    op.drop_index("ix_legal_documents_scope_org", table_name="legal_documents")
    op.drop_index("ix_legal_documents_document_origin", table_name="legal_documents")
    op.drop_index("ix_legal_documents_visibility_scope", table_name="legal_documents")
    op.drop_index("ix_legal_documents_organization_id", table_name="legal_documents")
    op.drop_column("legal_documents", "uploaded_by_user_id")
    op.drop_column("legal_documents", "confidentiality_level")
    op.drop_column("legal_documents", "document_origin")
    op.drop_column("legal_documents", "visibility_scope")
    op.drop_column("legal_documents", "organization_id")
