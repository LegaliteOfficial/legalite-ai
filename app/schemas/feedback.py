"""Pydantic schemas for the feedback endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class FeedbackCreate(BaseModel):
    """Body of POST /ask/{message_id}/feedback."""

    thumbs: Literal["up", "down"] = Field(
        ...,
        description=(
            "Whether the user found the assistant's answer helpful. "
            "Re-posting overwrites the existing feedback for this turn."
        ),
    )
    comment: str | None = Field(
        default=None,
        max_length=2000,
        description=(
            "Optional free-text note. Useful for capturing WHY the user "
            "thumbed-down — surfaces in the admin dashboard alongside "
            "the question text."
        ),
    )


class FeedbackResponse(BaseModel):
    """Echo of the persisted feedback row."""

    id: str
    message_id: str
    session_id: str
    thumbs: Literal["up", "down"]
    comment: str | None = None
    answer_confidence: str | None = None
    query_intent: str | None = None
    refused: bool = False
    created_at: datetime
    updated_at: datetime
