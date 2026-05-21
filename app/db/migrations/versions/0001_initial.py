"""Initial schema: legal_documents and legal_chunks.

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-19
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "legal_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("doc_type", sa.String(length=64), nullable=False),
        sa.Column("source_filename", sa.String(length=512), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="processing"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_legal_documents_title", "legal_documents", ["title"])
    op.create_index("ix_legal_documents_doc_type", "legal_documents", ["doc_type"])

    op.create_table(
        "legal_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("legal_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=512), nullable=True),
        sa.Column("chapter", sa.String(length=128), nullable=True),
        sa.Column("article", sa.String(length=128), nullable=True),
        sa.Column("section", sa.String(length=128), nullable=True),
        sa.Column("clause", sa.String(length=128), nullable=True),
        sa.Column("subsection", sa.String(length=128), nullable=True),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("vector_id", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_legal_chunks_document_id", "legal_chunks", ["document_id"])
    op.create_index("ix_legal_chunks_vector_id", "legal_chunks", ["vector_id"])


def downgrade() -> None:
    op.drop_index("ix_legal_chunks_vector_id", table_name="legal_chunks")
    op.drop_index("ix_legal_chunks_document_id", table_name="legal_chunks")
    op.drop_table("legal_chunks")
    op.drop_index("ix_legal_documents_doc_type", table_name="legal_documents")
    op.drop_index("ix_legal_documents_title", table_name="legal_documents")
    op.drop_table("legal_documents")
