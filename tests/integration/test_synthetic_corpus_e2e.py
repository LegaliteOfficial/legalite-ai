"""Phase 2.5 — synthetic end-to-end corpus proof.

Unlike test_tenant_isolation_live.py's `_ingest_via_real_pipeline` (which
injects already-parsed text via a fake PdfParser to keep that test focused
on tenant isolation), this test exercises the REAL PdfParser against a
real, valid PDF byte stream — the one piece of the pipeline the tenant
isolation suite deliberately skips. Everything downstream (chunker,
embeddings, Qdrant, retrieval, citation) is the same production code.

Fake embeddings only (deterministic, from conftest.py) — no OpenAI calls.

The fixture text is entirely fictional and clearly labelled as such. It
must never be used as legal evaluation ground truth (see
evaluation/datasets/README.md) — it exists only to prove the pipeline
works end to end, not to test legal accuracy.
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
from app.ingestion.pdf_parser import PdfParser
from app.rag.hybrid_retriever import HybridRetriever
from app.services.ingestion_service import IngestionService

_FICTIONAL_MARKER = "FICTIONAL-TEST-LAW-9K2Q-NOT-REAL-LEGISLATION"


def _make_synthetic_law_pdf() -> bytes:
    """A minimal, valid single-page PDF containing clearly-fictional test
    legislation. /Length is computed from the actual stream bytes —
    earlier in this project a hand-typed, guessed /Length value silently
    truncated the extracted text mid-word; computing it here avoids
    repeating that exact mistake.
    """
    lines = [
        f"TEST LAW - NOT REAL LAW ({_FICTIONAL_MARKER})",
        "SECTION 1 - Registration",
        "A fictional entity must submit Form A within 10 days.",
        "SECTION 2 - Renewal",
        "A fictional entity must renew its registration annually.",
    ]
    content_ops = ["BT", "/F1 12 Tf", "50 740 Td"]
    for i, line in enumerate(lines):
        if i > 0:
            content_ops.append("0 -20 Td")
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content_ops.append(f"({escaped}) Tj")
    content_ops.append("ET")
    stream = "\n".join(content_ops).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj".encode() + b"\n" + obj + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref_offset}\n%%EOF".encode()
    return bytes(out)


def test_synthetic_pdf_parses_with_the_real_parser() -> None:
    """Sanity check the fixture itself before trusting the full pipeline
    test below — if this fails, the problem is the fixture, not
    ingestion/retrieval."""
    import tempfile
    from pathlib import Path

    pdf_bytes = _make_synthetic_law_pdf()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        path = Path(f.name)
    try:
        parsed = PdfParser().parse(path)
        full_text = "\n".join(p.text for p in parsed.pages)
        assert _FICTIONAL_MARKER in full_text
        assert "SECTION 1" in full_text.upper()
        assert "SECTION 2" in full_text.upper()
        # The exact failure mode from earlier in this project: a bad
        # /Length silently truncates the stream mid-word.
        assert "registration annually" in full_text.lower()
    finally:
        path.unlink()


@pytest.mark.integration
def test_synthetic_corpus_end_to_end(db_session, live_vector_store, fake_embeddings, row_tracker):
    """PDF -> real PdfParser -> real LegalChunker -> fake embeddings ->
    live Qdrant upsert -> live Postgres -> real HybridRetriever ->
    citation traces back to the exact document/chunk ingested.

    Fails if: ingestion succeeds but Qdrant indexing fails (no hit on
    retrieval), Qdrant indexing succeeds but the tenant filter hides it
    from its own owner, or the retrieved passage's document_id/chunk_id
    don't correspond to a real Postgres row.
    """
    org_id = uuid.uuid4()
    tenant = TenantContext(organization_id=org_id, user_id=uuid.uuid4(), role=TenantRole.LAWYER)

    service = IngestionService(
        db_session,
        parser=PdfParser(),  # the REAL parser — this is the point of this test
        embeddings=fake_embeddings,  # type: ignore[arg-type]
        vector_store=live_vector_store.store,  # type: ignore[arg-type]
    )
    # Route through the run-scoped throwaway collection, same as the
    # tenant-isolation suite.
    live_vector_store.store._collection = live_vector_store.collection_name  # type: ignore[attr-defined]

    document_id, chunk_count = service.ingest_pdf(
        file_bytes=_make_synthetic_law_pdf(),
        original_filename="synthetic_test_law.pdf",
        title="Synthetic Test Law (fictional, not real legislation)",
        doc_type="Test Fixture",
        visibility_scope=VisibilityScope.TENANT_PRIVATE,
        document_origin=DocumentOrigin.INTERNAL_MEMO,
        confidentiality_level=ConfidentialityLevel.INTERNAL,
        tenant=tenant,
    )
    row_tracker.document_ids.append(document_id)
    row_tracker.org_ids.append(org_id)

    assert chunk_count > 0, "ingestion produced zero chunks — parser or chunker failed silently"

    # --- Postgres: the document and its chunks actually exist ---
    from sqlalchemy import text as sql_text

    doc_row = db_session.execute(
        sql_text("SELECT id, status FROM legal_documents WHERE id = :id"),
        {"id": str(document_id)},
    ).fetchone()
    assert doc_row is not None, "document row missing from Postgres after ingest_pdf returned"
    assert doc_row[1] == "ingested"

    chunk_rows = db_session.execute(
        sql_text("SELECT id, vector_id, content FROM legal_chunks WHERE document_id = :id"),
        {"id": str(document_id)},
    ).fetchall()
    assert len(chunk_rows) == chunk_count
    assert all(row[1] is not None for row in chunk_rows), (
        "a chunk has no vector_id — never upserted"
    )
    assert any(_FICTIONAL_MARKER in (row[2] or "") for row in chunk_rows)

    # --- Qdrant: the vectors are actually retrievable, not just referenced ---
    retriever = HybridRetriever(
        db=db_session, embeddings=fake_embeddings, vector_store=live_vector_store.store
    )
    result = retriever.retrieve(
        "What must a fictional entity do to renew its registration?",
        tenant=tenant,
    )
    matching = [p for p in result.passages if p.document_id == str(document_id)]
    assert matching, (
        "the ingested document exists in Postgres and Qdrant reports it has "
        "vectors, but HybridRetriever could not find it — indexing or "
        "retrieval is broken even though ingestion reported success"
    )

    # --- Citation traceability: document_id/chunk_id on the hit are real ---
    hit = matching[0]
    chunk_ids = {str(row[0]) for row in chunk_rows}
    assert hit.chunk_id in chunk_ids, "retrieved chunk_id does not match any real legal_chunks row"

    # --- Tenant isolation still holds for this freshly-ingested document ---
    other_org = uuid.uuid4()
    other_tenant = TenantContext(
        organization_id=other_org, user_id=uuid.uuid4(), role=TenantRole.LAWYER
    )
    other_result = retriever.retrieve(
        "What must a fictional entity do to renew its registration?",
        tenant=other_tenant,
    )
    leaked = [p for p in other_result.passages if p.document_id == str(document_id)]
    assert not leaked, "a TENANT_PRIVATE document leaked to a different organization_id"
