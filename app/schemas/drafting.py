"""Request / event shapes for ``POST /draft`` — AI drafting in the studio.

The drafting studio sends the document it has open (as plain text), an
optional selection, and what the user wants done. Four modes:

- ``generate``  write a new document or section from an instruction.
- ``rewrite``   rewrite the selection (improve, formalise, shorten, ...).
- ``continue``  write the next part of the document from where it stops.
- ``review``    check the document and return a list of issues.

Write modes stream an HTML fragment; review returns structured issues.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

DraftMode = Literal["generate", "rewrite", "continue", "review"]

# Preset rewrite actions the studio exposes as one-click buttons. Free-form
# instructions go in ``instruction``; a preset just supplies a default one.
RewriteAction = Literal[
    "improve",
    "formalise",
    "shorten",
    "expand",
    "plain_english",
    "add_authority",
]

# Caps keep a single request bounded. The studio trims before sending; the
# server enforces it so a direct caller cannot run up the token bill.
MAX_INSTRUCTION_CHARS = 4_000
MAX_DOCUMENT_CHARS = 60_000
MAX_SELECTION_CHARS = 20_000
MAX_MATTER_CHARS = 6_000


class DraftRequest(BaseModel):
    mode: DraftMode
    instruction: str = Field(default="", max_length=MAX_INSTRUCTION_CHARS)
    action: RewriteAction | None = None
    document_title: str | None = Field(default=None, max_length=300)
    document: str | None = Field(default=None, max_length=MAX_DOCUMENT_CHARS)
    selection: str | None = Field(default=None, max_length=MAX_SELECTION_CHARS)
    matter: str | None = Field(
        default=None,
        max_length=MAX_MATTER_CHARS,
        description=(
            "Facts about the linked case: title, court, suit number, parties, "
            "client. Treated as data, never as instructions."
        ),
    )
    use_authorities: bool = Field(
        default=True,
        description="Search the legal library and let the draft cite what it finds.",
    )

    @model_validator(mode="after")
    def _check_mode_inputs(self) -> DraftRequest:
        has_instruction = bool(self.instruction.strip())
        has_selection = bool((self.selection or "").strip())
        has_document = bool((self.document or "").strip())
        if self.mode == "generate" and not has_instruction:
            raise ValueError("generate needs an instruction")
        if self.mode == "rewrite":
            if not has_selection:
                raise ValueError("rewrite needs a selection")
            if not has_instruction and self.action is None:
                raise ValueError("rewrite needs an instruction or an action")
        if self.mode in ("continue", "review") and not has_document:
            raise ValueError(f"{self.mode} needs the document text")
        return self


class DraftSource(BaseModel):
    """A library passage the draft was allowed to cite."""

    ref: int
    label: str
    law: str = ""
    section: str = ""
    excerpt: str = ""


ReviewSeverity = Literal["high", "medium", "low"]


class ReviewIssue(BaseModel):
    severity: ReviewSeverity = "medium"
    category: str = ""
    title: str
    detail: str = ""
    # Verbatim text from the document the issue is about, so the studio can
    # find and highlight it. Empty when the issue is about something missing.
    quote: str = ""
    suggestion: str = ""


class ReviewResult(BaseModel):
    summary: str = ""
    issues: list[ReviewIssue] = Field(default_factory=list)
