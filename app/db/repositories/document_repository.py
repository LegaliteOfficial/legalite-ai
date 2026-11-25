"""Repository for legal documents and their chunks.

Repositories encapsulate persistence so services don't depend on SQLAlchemy
directly. This keeps the RAG layer testable and the swap-the-DB story cheap.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.telemetry import record_postgres_failure
from app.db.enums import ConfidentialityLevel, VisibilityScope
from app.db.models.legal_chunk import LegalChunk
from app.db.models.legal_document import LegalDocument


@dataclass(slots=True)
class ChunkRecord:
    """Plain payload for inserting a chunk — keeps callers free of ORM imports."""

    content: str
    title: str | None = None
    chapter: str | None = None
    part: str | None = None
    article: str | None = None
    section: str | None = None
    clause: str | None = None
    subsection: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    vector_id: str | None = None
    law_type: str | None = None
    act_number: str | None = None
    effective_year: int | None = None
    legal_topic: str | None = None
    # Tenancy mirror — must match the parent document's scope.
    organization_id: uuid.UUID | None = None
    visibility_scope: str = VisibilityScope.GLOBAL.value
    document_origin: str | None = None
    # Phase 5 M9 — character offsets within source page text.
    start_char: int | None = None
    end_char: int | None = None
    # Phase 2.5 bug fix: generated up front (not left to the ORM's insert-
    # time default) so the SAME id can be threaded into the Qdrant
    # VectorPayload.chunk_id before this row is ever inserted — previously
    # VectorPayload.chunk_id was a throwaway uuid4() with no relation to
    # the row's real eventual id, so every citation's chunk_id was
    # unresolvable against legal_chunks. See ingestion_service.py.
    id: uuid.UUID = field(default_factory=uuid.uuid4)


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    # ---- documents ---------------------------------------------------------

    def create_document(
        self,
        *,
        title: str,
        doc_type: str,
        source_filename: str,
        storage_path: str,
        page_count: int = 0,
        law_type: str | None = None,
        act_number: str | None = None,
        effective_year: int | None = None,
        organization_id: uuid.UUID | None = None,
        visibility_scope: str = VisibilityScope.GLOBAL.value,
        document_origin: str | None = None,
        confidentiality_level: str = ConfidentialityLevel.PUBLIC.value,
        uploaded_by_user_id: uuid.UUID | None = None,
    ) -> LegalDocument:
        doc = LegalDocument(
            title=title,
            doc_type=doc_type,
            source_filename=source_filename,
            storage_path=storage_path,
            page_count=page_count,
            status="processing",
            law_type=law_type,
            act_number=act_number,
            effective_year=effective_year,
            organization_id=organization_id,
            visibility_scope=visibility_scope,
            document_origin=document_origin,
            confidentiality_level=confidentiality_level,
            uploaded_by_user_id=uploaded_by_user_id,
        )
        self._db.add(doc)
        self._db.flush()
        return doc

    def find_global_by_titles(
        self, titles: Iterable[str], doc_type: str | None
    ) -> dict[str, LegalDocument]:
        """Batch lookup for bulk-upload idempotency.

        Returns a dict mapping ``title`` → existing GLOBAL document
        whose ``status == 'ingested'``. Only completed documents
        count — half-ingested rows shouldn't block a retry.

        One SQL query regardless of batch size, which matters because
        ``POST /upload-law/bulk`` is the only caller and the alternative
        is N queries per request.

        ``doc_type=None`` matches on title alone — used for auto-classified
        uploads, whose type isn't known until the worker has read the PDF.
        """
        title_list = [t for t in titles if t]
        if not title_list:
            return {}
        query = (
            select(LegalDocument)
            .where(LegalDocument.title.in_(title_list))
            .where(LegalDocument.visibility_scope == VisibilityScope.GLOBAL.value)
            .where(LegalDocument.status == "ingested")
        )
        if doc_type is not None:
            query = query.where(LegalDocument.doc_type == doc_type)
        rows = self._db.execute(query).scalars()
        # If the same title exists twice (it shouldn't but might in
        # legacy data), the later row wins. Both rows are equally
        # valid skip targets so this is fine.
        return {doc.title: doc for doc in rows}

    def set_remote_storage_key(self, document_id: uuid.UUID, remote_storage_key: str) -> None:
        """Persist the Supabase Storage key for the original PDF.

        Called after a successful object-store upload during ingestion.
        Failures upstream leave the column NULL — the API surface then
        gracefully degrades to text-only viewing.
        """
        doc = self._db.get(LegalDocument, document_id)
        if doc is None:
            return
        doc.remote_storage_key = remote_storage_key
        self._db.flush()

    def mark_document_status(
        self, document_id: uuid.UUID, status: str, notes: str | None = None
    ) -> None:
        doc = self._db.get(LegalDocument, document_id)
        if doc is None:
            return
        doc.status = status
        if notes is not None:
            doc.notes = notes

    def get_document(self, document_id: uuid.UUID) -> LegalDocument | None:
        return self._db.get(LegalDocument, document_id)

    def list_documents(self) -> list[LegalDocument]:
        stmt = select(LegalDocument).order_by(LegalDocument.created_at.desc())
        return list(self._db.scalars(stmt).all())

    # ---- chunks ------------------------------------------------------------

    def add_chunks(self, document_id: uuid.UUID, chunks: Iterable[ChunkRecord]) -> list[LegalChunk]:
        rows = [
            LegalChunk(
                id=c.id,
                document_id=document_id,
                content=c.content,
                title=c.title,
                chapter=c.chapter,
                part=c.part,
                article=c.article,
                section=c.section,
                clause=c.clause,
                subsection=c.subsection,
                page_start=c.page_start,
                page_end=c.page_end,
                vector_id=c.vector_id,
                law_type=c.law_type,
                act_number=c.act_number,
                effective_year=c.effective_year,
                legal_topic=c.legal_topic,
                organization_id=c.organization_id,
                visibility_scope=c.visibility_scope,
                document_origin=c.document_origin,
                start_char=c.start_char,
                end_char=c.end_char,
            )
            for c in chunks
        ]
        self._db.add_all(rows)
        self._db.flush()
        return rows

    def get_chunks_by_vector_ids(self, vector_ids: list[str]) -> list[LegalChunk]:
        if not vector_ids:
            return []
        stmt = select(LegalChunk).where(LegalChunk.vector_id.in_(vector_ids))
        return list(self._db.scalars(stmt).all())

    # ---- keyword search (Postgres FTS) ------------------------------------

    def keyword_search(
        self,
        query: str,
        top_k: int,
        law_type: str | None = None,
        *,
        organization_id: uuid.UUID | None = None,
    ) -> list[tuple[LegalChunk, float]]:
        """Full-text keyword search using the generated `content_tsv` column.

        Returns rows ordered by ts_rank_cd descending, paired with their score.

        Tenant scope (must mirror the Qdrant filter exactly so vector and
        keyword paths return the same eligible set):

        - ``organization_id is None``: GLOBAL chunks only.
        - ``organization_id`` set: GLOBAL chunks OR chunks belonging to that
          org (with ``visibility_scope = tenant_private``).

        Implemented with parameterized SQL so an arbitrary tenant_id can
        never break out of the filter via injection.
        """
        if not query.strip():
            return []

        params: dict[str, object] = {"q": query, "limit": top_k}
        clauses: list[str] = ["content_tsv @@ q"]

        if law_type:
            clauses.append("law_type = :law_type")
            params["law_type"] = law_type

        # Tenant boundary — enforced even when the chunks/payloads in this
        # database predate the tenancy migration (the backfill set every
        # legacy row to visibility_scope='global').
        if organization_id is None:
            clauses.append("visibility_scope = 'global'")
        else:
            clauses.append(
                "(visibility_scope = 'global' "
                "OR (visibility_scope = 'tenant_private' "
                "AND organization_id = :organization_id))"
            )
            params["organization_id"] = str(organization_id)

        sql = (
            "SELECT id, ts_rank_cd(content_tsv, q) AS score "
            "FROM legal_chunks, plainto_tsquery('english', :q) q "
            "WHERE " + " AND ".join(clauses) + " "
            "ORDER BY score DESC LIMIT :limit"
        )

        try:
            rows = self._db.execute(text(sql), params).all()
        except SQLAlchemyError:
            record_postgres_failure("keyword_search")
            raise
        if not rows:
            return []

        id_to_score = {row[0]: float(row[1]) for row in rows}
        try:
            chunks = self._db.scalars(
                select(LegalChunk).where(LegalChunk.id.in_(list(id_to_score.keys())))
            ).all()
        except SQLAlchemyError:
            record_postgres_failure("keyword_search")
            raise
        # Preserve score order
        chunk_by_id = {c.id: c for c in chunks}
        return [(chunk_by_id[cid], id_to_score[cid]) for cid in id_to_score if cid in chunk_by_id]
