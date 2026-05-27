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
from app.ingestion.embedding_service import EmbeddingService
from app.ingestion.legal_chunker import LegalChunker
from app.ingestion.pdf_parser import ParsedPdf, PdfParser
from app.services.citation_resolution_service import CitationResolutionService
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
                    organization_id=org_str,
                    visibility_scope=visibility_scope.value,
                    document_origin=origin_str,
                    confidentiality_level=confidentiality_level.value,
                    start_char=c.start_char,
                    end_char=c.end_char,
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
                    organization_id=org_id,
                    visibility_scope=visibility_scope.value,
                    document_origin=origin_str,
                    start_char=c.start_char,
                    end_char=c.end_char,
                )
                for c, pid in zip(chunks, point_ids, strict=True)
            ]
            self._repo.add_chunks(document_id, records)
            document.page_count = parsed.page_count

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
            # whatever cases are already in the corpus. Best-effort; failure
            # never breaks ingestion.
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
