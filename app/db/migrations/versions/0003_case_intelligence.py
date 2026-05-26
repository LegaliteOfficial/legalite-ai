"""Case-law intelligence: structured case metadata, parties, citation graph.

Revision ID: 0003_case_intelligence
Revises: 0002_intelligence_upgrade
Create Date: 2026-05-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_case_intelligence"
down_revision: str | Sequence[str] | None = "0002_intelligence_upgrade"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ---- legal_cases ------------------------------------------------------
    op.create_table(
        "legal_cases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("legal_documents.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("case_number", sa.String(length=128), nullable=True),
        sa.Column("case_title", sa.String(length=512), nullable=False),
        sa.Column("court", sa.String(length=128), nullable=True),
        sa.Column("judges", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("decision_date", sa.Date(), nullable=True),
        sa.Column("outcome", sa.String(length=256), nullable=True),
        sa.Column("holding", sa.Text(), nullable=True),
        sa.Column("facts_summary", sa.Text(), nullable=True),
        sa.Column("legal_topics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("jurisdiction", sa.String(length=8), nullable=False, server_default="GH"),
        sa.Column("extra", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
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
    op.create_index("ix_legal_cases_document_id", "legal_cases", ["document_id"])
    op.create_index("ix_legal_cases_case_number", "legal_cases", ["case_number"])
    op.create_index("ix_legal_cases_case_title", "legal_cases", ["case_title"])
    op.create_index("ix_legal_cases_court", "legal_cases", ["court"])
    op.create_index("ix_legal_cases_decision_date", "legal_cases", ["decision_date"])
    # GIN on legal_topics for jsonb containment lookups (?, @>) used by similarity.
    op.execute(
        "CREATE INDEX ix_legal_cases_legal_topics "
        "ON legal_cases USING GIN (legal_topics)"
    )

    # ---- case_parties -----------------------------------------------------
    op.create_table(
        "case_parties",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "case_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("legal_cases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("party_name", sa.String(length=512), nullable=False),
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
    op.create_index("ix_case_parties_case_id", "case_parties", ["case_id"])
    op.create_index("ix_case_parties_party_name", "case_parties", ["party_name"])

    # ---- case_citations ---------------------------------------------------
    op.create_table(
        "case_citations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "citing_case_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("legal_cases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "cited_case_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("legal_cases.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("cited_raw_text", sa.Text(), nullable=False),
        sa.Column("cited_case_title", sa.String(length=512), nullable=True),
        sa.Column("cited_reporter", sa.String(length=64), nullable=True),
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
        sa.UniqueConstraint(
            "citing_case_id", "cited_raw_text", name="uq_case_citations_edge"
        ),
    )
    op.create_index(
        "ix_case_citations_citing_case_id", "case_citations", ["citing_case_id"]
    )
    op.create_index(
        "ix_case_citations_cited_case_id", "case_citations", ["cited_case_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_case_citations_cited_case_id", table_name="case_citations")
    op.drop_index("ix_case_citations_citing_case_id", table_name="case_citations")
    op.drop_table("case_citations")

    op.drop_index("ix_case_parties_party_name", table_name="case_parties")
    op.drop_index("ix_case_parties_case_id", table_name="case_parties")
    op.drop_table("case_parties")

    op.execute("DROP INDEX IF EXISTS ix_legal_cases_legal_topics")
    op.drop_index("ix_legal_cases_decision_date", table_name="legal_cases")
    op.drop_index("ix_legal_cases_court", table_name="legal_cases")
    op.drop_index("ix_legal_cases_case_title", table_name="legal_cases")
    op.drop_index("ix_legal_cases_case_number", table_name="legal_cases")
    op.drop_index("ix_legal_cases_document_id", table_name="legal_cases")
    op.drop_table("legal_cases")
