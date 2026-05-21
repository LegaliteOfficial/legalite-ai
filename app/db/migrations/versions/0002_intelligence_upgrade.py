"""Intelligence upgrade: richer chunk metadata, Postgres FTS, chat memory.

Revision ID: 0002_intelligence_upgrade
Revises: 0001_initial
Create Date: 2026-05-19
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_intelligence_upgrade"
down_revision: str | Sequence[str] | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ---- legal_chunks: richer metadata -----------------------------------
    op.add_column("legal_chunks", sa.Column("law_type", sa.String(length=64), nullable=True))
    op.add_column("legal_chunks", sa.Column("act_number", sa.String(length=64), nullable=True))
    op.add_column("legal_chunks", sa.Column("part", sa.String(length=128), nullable=True))
    op.add_column("legal_chunks", sa.Column("effective_year", sa.Integer(), nullable=True))
    op.add_column("legal_chunks", sa.Column("legal_topic", sa.String(length=128), nullable=True))
    op.create_index("ix_legal_chunks_law_type", "legal_chunks", ["law_type"])
    op.create_index("ix_legal_chunks_legal_topic", "legal_chunks", ["legal_topic"])

    # ---- legal_chunks: Postgres FTS --------------------------------------
    # Generated column built from `content`; populated automatically on insert/update.
    op.execute(
        "ALTER TABLE legal_chunks "
        "ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED"
    )
    op.execute(
        "CREATE INDEX ix_legal_chunks_content_tsv "
        "ON legal_chunks USING GIN (content_tsv)"
    )

    # ---- legal_documents: mirror new metadata for filtering ---------------
    op.add_column("legal_documents", sa.Column("law_type", sa.String(length=64), nullable=True))
    op.add_column("legal_documents", sa.Column("act_number", sa.String(length=64), nullable=True))
    op.add_column("legal_documents", sa.Column("effective_year", sa.Integer(), nullable=True))
    op.create_index("ix_legal_documents_law_type", "legal_documents", ["law_type"])

    # ---- chat memory ------------------------------------------------------
    op.create_table(
        "chat_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(length=256), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
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

    op.create_table(
        "chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(length=16), nullable=False),  # user | assistant | system
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])
    op.create_index(
        "ix_chat_messages_session_created",
        "chat_messages",
        ["session_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_chat_messages_session_created", table_name="chat_messages")
    op.drop_index("ix_chat_messages_session_id", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")

    op.drop_index("ix_legal_documents_law_type", table_name="legal_documents")
    op.drop_column("legal_documents", "effective_year")
    op.drop_column("legal_documents", "act_number")
    op.drop_column("legal_documents", "law_type")

    op.execute("DROP INDEX IF EXISTS ix_legal_chunks_content_tsv")
    op.execute("ALTER TABLE legal_chunks DROP COLUMN IF EXISTS content_tsv")

    op.drop_index("ix_legal_chunks_legal_topic", table_name="legal_chunks")
    op.drop_index("ix_legal_chunks_law_type", table_name="legal_chunks")
    op.drop_column("legal_chunks", "legal_topic")
    op.drop_column("legal_chunks", "effective_year")
    op.drop_column("legal_chunks", "part")
    op.drop_column("legal_chunks", "act_number")
    op.drop_column("legal_chunks", "law_type")
