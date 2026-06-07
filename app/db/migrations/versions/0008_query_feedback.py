"""Query feedback for the self-improvement loop (Tier 1).

Revision ID: 0008_query_feedback
Revises: 0007_ingestion_jobs
Create Date: 2026-05-28

The ``query_feedback`` table is the foundation of the self-learning loop.
Each row is a 👍/👎 (and optional free-text note) that a user attached to a
specific assistant turn — identified by the ``chat_messages.id`` it
references. Downstream, this data drives:

  - Reranker fine-tuning (positive/negative pairs).
  - Gap dashboards (queries that consistently 👎).
  - Quality observability (👍/👎 rate per intent / confidence band).

Schema choices that future-you will appreciate
----------------------------------------------
- ``message_id`` is a foreign key to ``chat_messages`` so deleting an
  assistant turn cascades its feedback (no orphan rows). It is also
  UNIQUE — a user can update their thumbs by re-posting, but we keep
  one row per assistant turn to make analytics straightforward
  (no "which feedback row is canonical" disambiguation downstream).
- ``question_text`` is DENORMALISED — copied from the user turn at
  feedback-time. ``chat_messages`` can rotate out via session-eviction
  jobs but the analytics signal stays. The original ``message_id``
  remains the source of truth when both rows still exist.
- ``answer_confidence`` / ``query_intent`` / ``refused`` are also
  denormalised — the assistant turn's ``metadata_json`` carries them,
  but pivoting analytics by intent×confidence is the most frequent
  query shape, and a top-level column is far faster than JSON path
  extraction on every read.
- ``organization_id`` / ``user_id`` are nullable for parity with the
  rest of the schema (anonymous /ask is supported).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_query_feedback"
down_revision: str | Sequence[str] | None = "0007_ingestion_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "query_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "message_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_messages.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "organization_id", postgresql.UUID(as_uuid=True), nullable=True
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        # "up" | "down" — enforced by app layer + check constraint below so
        # we can index it without storing the longer string.
        sa.Column("thumbs", sa.String(length=8), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        # Denormalised analytical columns (see module docstring).
        sa.Column("question_text", sa.Text(), nullable=True),
        sa.Column("answer_confidence", sa.String(length=16), nullable=True),
        sa.Column("query_intent", sa.String(length=64), nullable=True),
        sa.Column("refused", sa.Boolean(), nullable=False, server_default=sa.false()),
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
        sa.CheckConstraint(
            "thumbs IN ('up', 'down')", name="ck_query_feedback_thumbs",
        ),
    )

    # Analytics indexes. The (organization_id, created_at DESC) shape
    # supports the per-tenant gap dashboard; (thumbs, query_intent) is for
    # "what intents are most thumbed-down" rollups.
    op.create_index(
        "ix_query_feedback_org_created",
        "query_feedback",
        ["organization_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_query_feedback_thumbs_intent",
        "query_feedback",
        ["thumbs", "query_intent"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_query_feedback_thumbs_intent", table_name="query_feedback"
    )
    op.drop_index("ix_query_feedback_org_created", table_name="query_feedback")
    op.drop_table("query_feedback")
