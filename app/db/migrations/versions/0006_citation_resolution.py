"""Citation graph resolution metadata (Phase 5 — M5).

Revision ID: 0006_citation_resolution
Revises: 0005_chunk_offsets
Create Date: 2026-05-27

Extends ``case_citations`` with the fuzzy-match confidence and a
``resolved_at`` timestamp so we can audit which edges in the precedent
graph were resolved automatically vs left dangling.

``cited_case_id`` was already nullable on creation — this migration only
adds the audit columns and one index that the resolver query needs.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_citation_resolution"
down_revision: str | Sequence[str] | None = "0005_chunk_offsets"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "case_citations",
        sa.Column("resolution_confidence", sa.Numeric(precision=4, scale=3), nullable=True),
    )
    op.add_column(
        "case_citations",
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_case_citations_resolved", "case_citations", ["cited_case_id", "resolved_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_case_citations_resolved", table_name="case_citations")
    op.drop_column("case_citations", "resolved_at")
    op.drop_column("case_citations", "resolution_confidence")
