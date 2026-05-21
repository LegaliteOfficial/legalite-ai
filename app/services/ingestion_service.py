"""End-to-end ingestion: PDF → chunks → embeddings → Qdrant + Postgres.

Kept in a service so the API route stays thin and the pipeline is easy to
invoke from scripts (bulk import, re-index, etc.).
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.repositories.document_repository import ChunkRecord, DocumentRepository
from app.ingestion.embedding_service import EmbeddingService
from app.ingestion.legal_chunker import LegalChunker
from app.ingestion.pdf_parser import PdfParser
from app.vector_store.qdrant_client import QdrantVectorStore, VectorPayload

logger = get_logger(__name__)


class IngestionService:
    def __init__(
        self,
        db: Session,
        *,
        parser: PdfParser | None = None,
        chunker: LegalChunker | None = None,
        embeddings: EmbeddingService | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self._db = db
        self._repo = DocumentRepository(db)
        self._parser = parser or PdfParser()
        self._chunker = chunker or LegalChunker()
        self._embeddings = embeddings or EmbeddingService()
        self._vector_store = vector_store  # injected lazily to avoid import-time qdrant connect

    def ingest_pdf(
        self,
        *,
        file_bytes: bytes,
        original_filename: str,
        title: str,
        doc_type: str,
    ) -> tuple[uuid.UUID, int]:
        """Persist the PDF, parse + chunk + embed, write to Qdrant and Postgres.

        Returns (document_id, chunks_indexed).
        """
        settings = get_settings()
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        doc_id = uuid.uuid4()
        safe_name = f"{doc_id}_{Path(original_filename).name}"
        storage_path = upload_dir / safe_name
        storage_path.write_bytes(file_bytes)

        document = self._repo.create_document(
            title=title,
            doc_type=doc_type,
            source_filename=original_filename,
            storage_path=str(storage_path),
        )
        # Force the ORM-generated id to match what we computed so storage_path matches.
        # (We do this by reading back the actual id; storage filename keeps the temp uuid.)
        document_id = document.id

        try:
            parsed = self._parser.parse(storage_path)
            chunks = self._chunker.chunk(parsed, doc_title=title)
            if not chunks:
                self._repo.mark_document_status(
                    document_id, "failed", notes="No chunks produced from PDF"
                )
                return document_id, 0

            texts = [c.content for c in chunks]
            embeddings = self._embeddings.embed_documents(texts)

            from app.vector_store.qdrant_client import get_vector_store

            store = self._vector_store or get_vector_store()
            payloads = [
                VectorPayload(
                    document_id=str(document_id),
                    chunk_id=str(uuid.uuid4()),
                    law=title,
                    doc_type=doc_type,
                    title=c.title,
                    chapter=c.chapter,
                    part=c.part,
                    article=c.article,
                    section=c.section,
                    clause=c.clause,
                    subsection=c.subsection,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    content=c.content,
                    law_type=c.law_type,
                    act_number=c.act_number,
                    effective_year=c.effective_year,
                    legal_topic=c.legal_topic,
                )
                for c in chunks
            ]
            point_ids = store.upsert(embeddings, payloads)

            records = [
                ChunkRecord(
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
                    vector_id=pid,
                    law_type=c.law_type,
                    act_number=c.act_number,
                    effective_year=c.effective_year,
                    legal_topic=c.legal_topic,
                )
                for c, pid in zip(chunks, point_ids, strict=True)
            ]
            self._repo.add_chunks(document_id, records)
            document.page_count = parsed.page_count
            self._repo.mark_document_status(document_id, "ingested")
            self._db.commit()
            logger.info(
                "ingestion.completed",
                document_id=str(document_id),
                chunks=len(records),
                pages=parsed.page_count,
            )
            return document_id, len(records)
        except Exception as e:
            self._db.rollback()
            # Reopen a session-friendly state: mark failure in its own transaction.
            self._repo.mark_document_status(document_id, "failed", notes=str(e)[:500])
            self._db.commit()
            logger.exception("ingestion.failed", document_id=str(document_id))
            raise
