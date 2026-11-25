"""Real-Qdrant + real-Postgres tenant isolation tests.

Where the existing security suite tests a Python re-implementation of
``build_tenant_filter``, these tests:

1. Ingest a Firm-A private document via the real ingestion pipeline
   (chunker → fake embeddings → live Qdrant upsert → Postgres rows).
2. Run the real :class:`HybridRetriever` as Firm B (different org_id).
3. Assert zero Firm-A chunks surface, across BOTH the vector and
   keyword arms of retrieval.

If the production code path drifts (someone forgets the filter, an
upsert leaks the wrong scope, the OR-clause in keyword_search loses a
parenthesis…) these tests fail, where the simulation-based suite would
still report score=1.0.
"""

from __future__ import annotations

import uuid

import pytest

from app.core.tenant_context import TenantContext
from app.db.enums import (
    ConfidentialityLevel,
    DocumentOrigin,
    TenantRole,
    VisibilityScope,
)
from app.rag.hybrid_retriever import HybridRetriever
from app.services.ingestion_service import IngestionService

_FIRM_A_MARKER = "FIRM-A-CONFIDENTIAL-MARKER-7F3A-INTEGRATION"
_FIRM_B_MARKER = "FIRM-B-INTERNAL-MARKER-4A11-INTEGRATION"


def _ingest_via_real_pipeline(
    *,
    db,
    vector_store,
    fake_embeddings,
    body: str,
    org_id: uuid.UUID | None,
    visibility: VisibilityScope,
    document_origin: DocumentOrigin,
    tracker,
) -> uuid.UUID:
    """Use the real ``IngestionService`` but with our run-scoped Qdrant
    collection + fake embeddings. Everything else is production code."""
    from app.ingestion.legal_chunker import LegalChunker
    from app.ingestion.pdf_parser import ParsedPdf, PdfPage

    class _FakeParser:
        def parse(self, _path):
            return ParsedPdf(pages=[PdfPage(page_number=1, text=body)])

    tenant: TenantContext | None = None
    if org_id is not None:
        tenant = TenantContext(
            organization_id=org_id,
            user_id=uuid.uuid4(),
            role=TenantRole.LAWYER,
        )

    service = IngestionService(
        db,
        parser=_FakeParser(),  # type: ignore[arg-type]
        chunker=LegalChunker(),
        embeddings=fake_embeddings,  # type: ignore[arg-type]
        vector_store=vector_store,  # type: ignore[arg-type]
    )
    doc_id, _chunks = service.ingest_pdf(
        file_bytes=b"%PDF-1.4\n%%EOF",  # parser ignores; real bytes not needed
        original_filename=f"integration_{uuid.uuid4().hex[:8]}.pdf",
        title=f"integration test {visibility.value}",
        doc_type="Integration",
        visibility_scope=visibility,
        document_origin=document_origin,
        confidentiality_level=(
            ConfidentialityLevel.PUBLIC
            if visibility is VisibilityScope.GLOBAL
            else ConfidentialityLevel.INTERNAL
        ),
        tenant=tenant,
    )
    tracker.document_ids.append(doc_id)
    if org_id is not None:
        tracker.org_ids.append(org_id)
    return doc_id


@pytest.mark.integration
def test_firm_b_cannot_retrieve_firm_a_chunks_via_hybrid_retriever(
    db_session, live_vector_store, fake_embeddings, row_tracker
):
    """The core leakage check, against the production retriever."""
    org_a = uuid.uuid4()
    org_b = uuid.uuid4()

    _ingest_via_real_pipeline(
        db=db_session,
        vector_store=live_vector_store.store,
        fake_embeddings=fake_embeddings,
        body=(
            f"{_FIRM_A_MARKER}. This is Firm A's internal tenancy eviction "
            f"playbook. Tenant arrears, notice procedure, court filing."
        ),
        org_id=org_a,
        visibility=VisibilityScope.TENANT_PRIVATE,
        document_origin=DocumentOrigin.INTERNAL_MEMO,
        tracker=row_tracker,
    )
    _ingest_via_real_pipeline(
        db=db_session,
        vector_store=live_vector_store.store,
        fake_embeddings=fake_embeddings,
        body=f"{_FIRM_B_MARKER}. Firm B criminal defense strategy.",
        org_id=org_b,
        visibility=VisibilityScope.TENANT_PRIVATE,
        document_origin=DocumentOrigin.INTERNAL_MEMO,
        tracker=row_tracker,
    )

    retriever = HybridRetriever(
        db=db_session, embeddings=fake_embeddings, vector_store=live_vector_store.store
    )

    # Firm B asks about tenancy eviction — semantically close to Firm A's doc.
    firm_b_ctx = TenantContext(
        organization_id=org_b,
        user_id=uuid.uuid4(),
        role=TenantRole.LAWYER,
    )
    result = retriever.retrieve(
        "tenancy eviction procedure for tenant arrears",
        tenant=firm_b_ctx,
    )

    for passage in result.passages:
        assert _FIRM_A_MARKER not in passage.content, (
            "LEAK: Firm A's private chunk surfaced for Firm B "
            f"(passage source={passage.source}, score={passage.score:.3f})"
        )
        assert passage.organization_id != str(org_a), (
            "LEAK: passage with org_id=org_a returned to Firm B "
            f"(visibility_scope={passage.visibility_scope})"
        )


@pytest.mark.integration
def test_anonymous_caller_sees_no_tenant_chunks(
    db_session, live_vector_store, fake_embeddings, row_tracker
):
    """Anonymous → GLOBAL only. Tenant marker must never appear."""
    org_a = uuid.uuid4()
    _ingest_via_real_pipeline(
        db=db_session,
        vector_store=live_vector_store.store,
        fake_embeddings=fake_embeddings,
        body=f"{_FIRM_A_MARKER}. Strictly confidential playbook content.",
        org_id=org_a,
        visibility=VisibilityScope.TENANT_PRIVATE,
        document_origin=DocumentOrigin.SAMPLE_CASE,
        tracker=row_tracker,
    )

    retriever = HybridRetriever(
        db=db_session, embeddings=fake_embeddings, vector_store=live_vector_store.store
    )
    result = retriever.retrieve("strictly confidential playbook", tenant=None)

    for passage in result.passages:
        assert _FIRM_A_MARKER not in passage.content
        assert passage.visibility_scope == "global", (
            "LEAK: anonymous request retrieved a non-global passage "
            f"(scope={passage.visibility_scope}, org={passage.organization_id})"
        )


@pytest.mark.integration
def test_firm_a_can_still_retrieve_its_own_chunks(
    db_session, live_vector_store, fake_embeddings, row_tracker
):
    """Positive control — without this we could pass the leakage test by
    just returning empty everywhere."""
    org_a = uuid.uuid4()
    _ingest_via_real_pipeline(
        db=db_session,
        vector_store=live_vector_store.store,
        fake_embeddings=fake_embeddings,
        body=f"{_FIRM_A_MARKER}. Firm A's own internal eviction playbook.",
        org_id=org_a,
        visibility=VisibilityScope.TENANT_PRIVATE,
        document_origin=DocumentOrigin.INTERNAL_MEMO,
        tracker=row_tracker,
    )

    retriever = HybridRetriever(
        db=db_session, embeddings=fake_embeddings, vector_store=live_vector_store.store
    )
    firm_a_ctx = TenantContext(
        organization_id=org_a,
        user_id=uuid.uuid4(),
        role=TenantRole.LAWYER,
    )
    result = retriever.retrieve("internal eviction playbook", tenant=firm_a_ctx)

    assert any(_FIRM_A_MARKER in p.content for p in result.passages), (
        "POSITIVE CONTROL FAILED: Firm A could not retrieve its own private doc"
    )
