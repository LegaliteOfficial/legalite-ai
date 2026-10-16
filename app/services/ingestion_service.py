"""End-to-end ingestion: PDF → chunks → embeddings → Qdrant + Postgres.

Kept in a service so the API route stays thin and the pipeline is easy to
invoke from scripts (bulk import, re-index, etc.).

When the chunker detects a case-law document, we additionally run the
``CaseMetadataExtractor`` + ``CitationExtractor`` and persist a structured
``LegalCase`` row plus its parties and citation edges. Statutes /
constitution / regulations are unaffected.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.telemetry import record_upload
from app.core.tenant_context import TenantContext
from app.db.enums import (
    GLOBAL_ONLY_ORIGINS,
    TENANT_PRIVATE_ORIGINS,
    ConfidentialityLevel,
    DocumentOrigin,
    VisibilityScope,
)
from app.db.repositories.case_repository import (
    CaseCitationRecord,
    CasePartyRecord,
    CaseRecord,
    CaseRepository,
)
from app.db.repositories.document_repository import ChunkRecord, DocumentRepository
from app.ingestion.case_metadata_extractor import CaseMetadataExtractor
from app.ingestion.citation_extractor import CitationExtractor
from app.ingestion.document_classifier import AUTO_DOC_TYPE, DocumentClassifier
from app.ingestion.embedding_service import EmbeddingService
from app.ingestion.legal_chunker import LegalChunker
from app.ingestion.pdf_parser import ParsedPdf, PdfParser
from app.services.citation_resolution_service import CitationResolutionService
from app.services.storage_service import StorageService
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
        case_extractor: CaseMetadataExtractor | None = None,
        citation_extractor: CitationExtractor | None = None,
        classifier: DocumentClassifier | None = None,
        storage: StorageService | None = None,
    ) -> None:
        self._db = db
        self._repo = DocumentRepository(db)
        self._case_repo = CaseRepository(db)
        self._parser = parser or PdfParser()
        self._chunker = chunker or LegalChunker()
        self._embeddings = embeddings or EmbeddingService()
        self._vector_store = vector_store  # injected lazily to avoid import-time qdrant connect
        self._case_extractor = case_extractor or CaseMetadataExtractor()
        self._citation_extractor = citation_extractor or CitationExtractor()
        self._classifier = classifier
        # Optional: when Supabase Storage env is unset, this constructs
        # in disabled mode and every upload silently no-ops. See
        # StorageService.upload for the graceful-degradation contract.
        self._storage = storage if storage is not None else StorageService()

    def ingest_pdf(
        self,
        *,
        file_bytes: bytes,
        original_filename: str,
        title: str,
        doc_type: str,
        visibility_scope: VisibilityScope = VisibilityScope.GLOBAL,
        document_origin: DocumentOrigin | None = None,
        confidentiality_level: ConfidentialityLevel = ConfidentialityLevel.PUBLIC,
        tenant: TenantContext | None = None,
    ) -> tuple[uuid.UUID, int]:
        """Persist the PDF, parse + chunk + embed, write to Qdrant and Postgres.

        ``visibility_scope`` and ``document_origin`` decide whether this is a
        GLOBAL upload (Constitution, Acts, ...) or a TENANT_PRIVATE upload
        (sample case, internal memo, ...). The two paths share this code so
        the chunking / embedding pipeline can never drift between them.

        ``doc_type == AUTO_DOC_TYPE`` means the uploader didn't pick one:
        it is classified from the parsed text before chunking, so the
        detected type is what lands on the document, chunks and vectors.

        Returns (document_id, chunks_indexed).
        """
        _validate_origin_scope(visibility_scope, document_origin)
        org_id = _resolve_org_id(visibility_scope, tenant)
        uploaded_by = tenant.user_id if tenant else None

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
            organization_id=org_id,
            visibility_scope=visibility_scope.value,
            document_origin=document_origin.value if document_origin else None,
            confidentiality_level=confidentiality_level.value,
            uploaded_by_user_id=uploaded_by,
        )
        document_id = document.id

        try:
            parsed = self._parser.parse(storage_path)
            if doc_type == AUTO_DOC_TYPE:
                classification = (self._classifier or DocumentClassifier()).classify(
                    parsed, title
                )
                doc_type = classification.doc_type
                document.doc_type = classification.doc_type
                document.summary = classification.summary
                document.doc_type_source = classification.source
                logger.info(
                    "ingestion.classified",
                    document_id=str(document_id),
                    doc_type=doc_type,
                    source=classification.source,
                )
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
            origin_str = document_origin.value if document_origin else None
            org_str = str(org_id) if org_id else None
            # Bug fix (Phase 2.5): generate each chunk's id up front and use
            # the SAME value for both the Qdrant payload and the eventual
            # Postgres row (see ChunkRecord.id) — previously VectorPayload.chunk_id
            # was a throwaway uuid4() unrelated to the row Postgres would
            # later create, so every citation's chunk_id was unresolvable
            # against legal_chunks (caught by
            # tests/integration/test_synthetic_corpus_e2e.py).
            chunk_ids = [uuid.uuid4() for _ in chunks]
            payloads = [
                VectorPayload(
                    document_id=str(document_id),
                    chunk_id=str(chunk_ids[i]),
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
                    organization_id=org_str,
                    visibility_scope=visibility_scope.value,
                    document_origin=origin_str,
                    confidentiality_level=confidentiality_level.value,
                    start_char=c.start_char,
                    end_char=c.end_char,
                )
                for i, c in enumerate(chunks)
            ]
            point_ids = store.upsert(embeddings, payloads)

            records = [
                ChunkRecord(
                    id=chunk_ids[i],
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
                    organization_id=org_id,
                    visibility_scope=visibility_scope.value,
                    document_origin=origin_str,
                    start_char=c.start_char,
                    end_char=c.end_char,
                )
                for i, (c, pid) in enumerate(zip(chunks, point_ids, strict=True))
            ]
            self._repo.add_chunks(document_id, records)
            document.page_count = parsed.page_count
            # Bug fix (Phase 1I): create_document() runs before parsing/
            # chunking, so law_type/act_number/effective_year weren't known
            # yet and were never passed to it — legal_documents.law_type
            # etc. were always NULL even though the identical values were
            # correctly computed and stored per-chunk on legal_chunks.
            # chunks[0] is safe here: the "not chunks" branch above already
            # returned early, so this line only runs with >=1 chunk, and
            # doc-level metadata is broadcast identically to every chunk
            # (see LegalChunker._detect_doc_metadata + _attach_doc_meta).
            document.law_type = chunks[0].law_type
            document.act_number = chunks[0].act_number
            document.effective_year = chunks[0].effective_year

            # Upload the original PDF to Supabase Storage AFTER chunks
            # land in Postgres. Order matters: if the storage upload
            # fails we still have a fully-queryable document — only the
            # "view original" UX degrades. Doing it the other way round
            # would leak orphan objects on Postgres errors.
            if self._storage.enabled:
                scope_label = (
                    "tenant"
                    if visibility_scope is VisibilityScope.TENANT_PRIVATE
                    else "global"
                )
                remote_key = self._storage.build_key(
                    scope=scope_label,
                    document_id=document_id,
                    filename=original_filename,
                    organization_id=org_id,
                )
                if self._storage.upload(key=remote_key, data=file_bytes):
                    self._repo.set_remote_storage_key(document_id, remote_key)

            # Case-law branch: persist structured metadata + citation edges.
            detected_law_type = chunks[0].law_type if chunks else None
            if detected_law_type == "case_law":
                self._ingest_case_metadata(
                    document_id=document_id,
                    parsed=parsed,
                    declared_title=title,
                    organization_id=org_id,
                    visibility_scope=visibility_scope,
                )

            self._repo.mark_document_status(document_id, "ingested")
            self._db.commit()
            record_upload(scope=visibility_scope.value, outcome="ingested")
            logger.info(
                "ingestion.completed",
                document_id=str(document_id),
                chunks=len(records),
                pages=parsed.page_count,
                law_type=detected_law_type,
                visibility_scope=visibility_scope.value,
            )
            return document_id, len(records)
        except Exception as e:
            self._db.rollback()
            # Reopen a session-friendly state: mark failure in its own transaction.
            self._repo.mark_document_status(document_id, "failed", notes=str(e)[:500])
            self._db.commit()
            record_upload(scope=visibility_scope.value, outcome="failed")
            logger.exception("ingestion.failed", document_id=str(document_id))
            raise

    # ---- case-law sub-pipeline --------------------------------------------

    def _ingest_case_metadata(
        self,
        *,
        document_id: uuid.UUID,
        parsed: ParsedPdf,
        declared_title: str,
        organization_id: uuid.UUID | None,
        visibility_scope: VisibilityScope,
    ) -> None:
        """Persist a LegalCase row + parties + extracted citation edges.

        Failures here must NOT abort document ingestion — the chunks and
        embeddings are already valuable on their own. We log loudly and move on.
        """
        try:
            meta = self._case_extractor.extract(parsed, declared_title=declared_title)
        except Exception:
            logger.exception("case.extractor.failed", document_id=str(document_id))
            return

        body = "\n".join(p.text for p in parsed.pages[:40])
        try:
            citations = self._citation_extractor.extract(body)
        except Exception:
            logger.exception("case.citations.failed", document_id=str(document_id))
            citations = []

        try:
            # A SAVEPOINT, not the outer transaction — ``ingest_pdf``'s
            # document + chunk rows are already flushed but not yet
            # committed at this point. Without this, a DB-level failure
            # here (e.g. a too-long case_number) leaves the whole session
            # in Postgres's aborted-transaction state, and the NEXT
            # statement anywhere in ``ingest_pdf`` (mark_document_status)
            # raises a confusing PendingRollbackError that masks the real
            # cause and fails the entire job — even though the chunks and
            # embeddings, which this docstring says must survive, were
            # already fine. Caught in production: an unnormalized
            # case_number blew the 128-char column and took the whole
            # ingestion down with it.
            with self._db.begin_nested():
                case = self._case_repo.create_case(
                    CaseRecord(
                        document_id=document_id,
                        case_title=meta.case_title,
                        case_number=meta.case_number,
                        court=meta.court,
                        judges=meta.judges or None,
                        decision_date=meta.decision_date,
                        outcome=meta.outcome,
                        holding=meta.holding,
                        facts_summary=meta.facts_summary,
                        legal_topics=meta.legal_topics or None,
                        organization_id=organization_id,
                        visibility_scope=visibility_scope.value,
                        parties=[
                            CasePartyRecord(role=p.role, party_name=p.party_name)
                            for p in meta.parties
                        ],
                        citations=[
                            CaseCitationRecord(
                                cited_raw_text=c.raw_text,
                                cited_case_title=c.case_title,
                                cited_reporter=c.reporter,
                            )
                            for c in citations
                        ],
                    )
                )
                # Phase 5 M5 — try to resolve outgoing citation edges against
                # whatever cases are already in the corpus. Best-effort;
                # failure never breaks ingestion.
                try:
                    resolver = CitationResolutionService(self._db)
                    resolver.resolve_for_case(case.id)
                except Exception:
                    logger.exception(
                        "citation.resolution.failed", document_id=str(document_id)
                    )
            logger.info(
                "case.persisted",
                document_id=str(document_id),
                case_title=meta.case_title,
                court=meta.court,
                judges=len(meta.judges),
                topics=meta.legal_topics,
                citations=len(citations),
                enrichment_used=meta.enrichment_used,
            )
        except Exception:
            logger.exception("case.persist.failed", document_id=str(document_id))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_origin_scope(
    visibility_scope: VisibilityScope, document_origin: DocumentOrigin | None
) -> None:
    """Refuse mismatched origin/scope combinations at the service boundary.

    Examples this rejects:
    - A 'sample_case' uploaded to GLOBAL (would leak firm material publicly).
    - A 'constitution' uploaded as TENANT_PRIVATE (would hide authoritative law
      from everyone else).

    Origin is optional — when it's None, only the scope itself matters.
    """
    if document_origin is None:
        return
    if visibility_scope is VisibilityScope.GLOBAL and document_origin in TENANT_PRIVATE_ORIGINS:
        raise ValueError(
            f"document_origin={document_origin.value!r} is tenant-private; "
            "cannot be uploaded under GLOBAL visibility"
        )
    if (
        visibility_scope is VisibilityScope.TENANT_PRIVATE
        and document_origin in GLOBAL_ONLY_ORIGINS
    ):
        raise ValueError(
            f"document_origin={document_origin.value!r} is global-only; "
            "cannot be uploaded under TENANT_PRIVATE visibility"
        )


def _resolve_org_id(
    visibility_scope: VisibilityScope, tenant: TenantContext | None
) -> uuid.UUID | None:
    """Pick the organization_id to persist, honoring the scope.

    GLOBAL → always None (the tenant header is ignored even if present so
    admin operators can't accidentally tag a global Act with their own org).

    TENANT_PRIVATE → must have a tenant context with an organization_id.
    Refuse otherwise; better to 4xx than to write a private row no tenant
    can ever retrieve.
    """
    if visibility_scope is VisibilityScope.GLOBAL:
        return None
    if tenant is None or tenant.organization_id is None:
        raise ValueError(
            "TENANT_PRIVATE upload requires a tenant context with organization_id"
        )
    return tenant.organization_id
