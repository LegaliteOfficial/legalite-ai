"""GET /documents/{id} — full document text + signed PDF URL.

The Q&A pipeline gives the user citations like ``[1]`` pointing at
specific chunks. This endpoint hydrates those citations into a
viewable source: the entire document text (so the UI can render it
inline with the cited passage highlighted) plus a signed URL to the
original PDF (so a lawyer can verify against the actual filing —
judge's signature, court stamp, paragraph numbering).

Tenant gate
-----------
- Anonymous request → only GLOBAL documents are visible.
- Tenant-signed request → GLOBAL ∨ TENANT_PRIVATE rows belonging to the
  caller's organization.

Cross-tenant probing always returns 404 (never 403) so an unauthorised
caller cannot enumerate private document IDs from the response code.

Signed-URL TTL is driven by ``visibility_scope``:

- GLOBAL → 24h by default (configurable). Public-record law, safe to
  cache, friendly to CDN intermediaries.
- TENANT_PRIVATE → 5 min by default. If the URL leaks, the blast
  radius is bounded.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.dependencies.tenant import get_optional_tenant_context
from app.core.config import get_settings
from app.core.tenant_context import TenantContext
from app.db.enums import VisibilityScope
from app.db.models.legal_chunk import LegalChunk
from app.db.models.legal_document import LegalDocument
from app.db.session import get_db
from app.schemas.document_view import DocumentChunkView, DocumentView
from app.services.storage_service import StorageService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get(
    "/{document_id}",
    response_model=DocumentView,
    summary="Full text + signed PDF URL for a single document",
)
def get_document_view(
    document_id: str,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_optional_tenant_context),
) -> DocumentView:
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"document_id is not a valid UUID: {document_id!r}",
        ) from exc

    doc = db.get(LegalDocument, doc_uuid)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")

    # Tenant gate. We treat "not visible to this caller" identically to
    # "doesn't exist" — never reveal existence across tenants.
    caller_org_id = tenant.organization_id if tenant else None
    if not _is_visible_to(doc, caller_org_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")

    # Chunks ordered by their natural document position. ``page_start``
    # is the primary key; ``start_char`` disambiguates within a page;
    # ``created_at`` is the tiebreaker for legacy rows where the offset
    # columns were never populated.
    chunk_rows: list[LegalChunk] = list(
        db.execute(
            select(LegalChunk)
            .where(LegalChunk.document_id == doc_uuid)
            .order_by(
                LegalChunk.page_start.asc().nulls_last(),
                LegalChunk.start_char.asc().nulls_last(),
                LegalChunk.created_at.asc(),
            )
        ).scalars()
    )

    chunk_views = [
        DocumentChunkView(
            chunk_id=str(c.id),
            content=c.content,
            page_start=c.page_start,
            page_end=c.page_end,
            start_char=c.start_char,
            end_char=c.end_char,
            chapter=c.chapter,
            part=c.part,
            article=c.article,
            section=c.section,
            clause=c.clause,
            subsection=c.subsection,
        )
        for c in chunk_rows
    ]
    # Double-newline separator preserves chunk boundaries when callers
    # split full_text back — useful for highlighters that key off
    # paragraph offsets rather than chunk ids.
    full_text = "\n\n".join(c.content for c in chunk_rows)

    pdf_url, ttl = _mint_signed_url(doc)

    return DocumentView(
        id=str(doc.id),
        title=doc.title,
        doc_type=doc.doc_type,
        law_type=doc.law_type,
        visibility_scope=doc.visibility_scope,
        document_origin=doc.document_origin,
        page_count=doc.page_count,
        status=doc.status,
        full_text=full_text,
        chunks=chunk_views,
        pdf_url=pdf_url,
        pdf_url_expires_in_seconds=ttl,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


# ---- helpers ---------------------------------------------------------------


def _is_visible_to(doc: LegalDocument, caller_org_id: uuid.UUID | None) -> bool:
    """Mirror of ``build_tenant_filter`` for a single-row check.

    GLOBAL is visible to everyone. TENANT_PRIVATE is visible only when
    the caller's org_id matches the row's. Anonymous callers
    (``caller_org_id is None``) only see GLOBAL.
    """
    if doc.visibility_scope == VisibilityScope.GLOBAL.value:
        return True
    if doc.visibility_scope == VisibilityScope.TENANT_PRIVATE.value:
        return caller_org_id is not None and doc.organization_id == caller_org_id
    # Future scopes default-deny.
    return False


def _mint_signed_url(doc: LegalDocument) -> tuple[str | None, int | None]:
    """Build a signed URL for the original PDF, if storage was configured.

    Returns ``(url, ttl_seconds)`` or ``(None, None)``. The TTL varies
    by ``visibility_scope`` so private docs aren't long-lived.
    """
    if not doc.remote_storage_key:
        return None, None
    storage = StorageService()
    if not storage.enabled:
        return None, None

    settings = get_settings()
    ttl = (
        settings.storage_signed_url_ttl_private_seconds
        if doc.visibility_scope == VisibilityScope.TENANT_PRIVATE.value
        else settings.storage_signed_url_ttl_global_seconds
    )
    url = storage.signed_url(doc.remote_storage_key, expires_in_seconds=ttl)
    if url is None:
        return None, None
    return url, ttl
