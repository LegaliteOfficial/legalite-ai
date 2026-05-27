"""Chunk character offsets for UI source-highlighting (Phase 5 — M9).

Revision ID: 0005_chunk_offsets
Revises: 0004_tenancy_foundation
Create Date: 2026-05-27

Adds ``start_char`` / ``end_char`` to ``legal_chunks``. Both are nullable
so already-ingested chunks remain queryable; re-uploading a document
populates the offsets via the updated chunker.

The offsets are measured against the page text of ``page_start`` (the page
the chunk begins on). For multi-page chunks ``end_char`` refers to the
character offset within ``page_end``.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_chunk_offsets"
down_revision: str | Sequence[str] | None = "0004_tenancy_foundation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("legal_chunks", sa.Column("start_char", sa.Integer(), nullable=True))
    op.add_column("legal_chunks", sa.Column("end_char", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("legal_chunks", "end_char")
    op.drop_column("legal_chunks", "start_char")
