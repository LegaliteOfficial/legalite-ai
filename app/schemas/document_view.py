"""Schemas for the ``GET /documents/{id}`` source-view endpoint.

The shape is tuned for the front-end "click a citation, see the full
source" interaction:

- ``full_text`` lets the UI render the document inline (no PDF viewer
  needed for a fast hover-preview).
- ``chunks[].start_char`` / ``end_char`` map cleanly onto the
  citation ids the model returns, so the UI can highlight exactly
  the passage that was quoted.
- ``pdf_url`` is the escape hatch when the user wants the original —
  judge's signature, court stamp, paragraph numbering. Short-lived
  signed URL; NEVER the raw storage key.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentChunkView(BaseModel):
    """One indexed chunk with the offsets needed for inline highlighting."""

    chunk_id: str = Field(..., description="LegalChunk.id — also the citation anchor")
    content: str
    page_start: int | None = None
    page_end: int | None = None
    # Character offsets within the chunk's page text. Populated post
    # migration 0005; nullable for older rows.
    start_char: int | None = None
    end_char: int | None = None
    # Legal hierarchy — populated when the chunker detected structural
    # context (article, section, etc.). The UI can show these as a
    # breadcrumb above the highlighted passage.
    chapter: str | None = None
    part: str | None = None
    article: str | None = None
    section: str | None = None
    clause: str | None = None
    subsection: str | None = None


class DocumentView(BaseModel):
    """Full ``GET /documents/{id}`` payload.

    ``pdf_url`` is None when (a) Supabase Storage was not configured at
    ingest time, (b) the upload failed and ``remote_storage_key`` is
    still NULL, or (c) signing the URL just failed. The UI should
    degrade to text-only rendering in any of those cases.
    """

    id: str
    title: str
    doc_type: str
    law_type: str | None = None
    visibility_scope: str
    document_origin: str | None = None
    page_count: int = 0
    status: str
    full_text: str = Field(
        ...,
        description=(
            "Concatenated chunk content in document order. Each chunk "
            "is separated by a double newline so callers can also "
            "split back into chunks if they need to."
        ),
    )
    chunks: list[DocumentChunkView]
    pdf_url: str | None = Field(
        default=None,
        description=(
            "Short-lived signed URL for the original PDF. Absent when "
            "storage is unconfigured, upload failed, or signing failed."
        ),
    )
    pdf_url_expires_in_seconds: int | None = Field(
        default=None,
        description="TTL of the signed URL, if one was minted.",
    )
    created_at: datetime
    updated_at: datetime
