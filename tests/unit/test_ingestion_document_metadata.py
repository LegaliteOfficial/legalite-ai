"""Phase 1I regression test.

Bug: IngestionService.ingest_pdf created the ``legal_documents`` row
*before* parsing/chunking ran, so law_type/act_number/effective_year
(computed by LegalChunker from the PDF's own content) were never passed to
``create_document`` — those three columns were always NULL on the document,
even though the identical values were correctly attached to every row in
``legal_chunks``. This test drives ``ingest_pdf`` with fakes for every
external dependency (no real filesystem/DB/Qdrant/OpenAI needed) and
asserts the document object actually receives the detected metadata.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.db.enums import ConfidentialityLevel, VisibilityScope
from app.ingestion.document_classifier import AUTO_DOC_TYPE, DocumentClassifier
from app.ingestion.legal_chunker import LegalChunk
from app.ingestion.pdf_parser import ParsedPdf, PdfPage
from app.services.ingestion_service import IngestionService


class _FakeDocument:
    def __init__(self, doc_id: uuid.UUID) -> None:
        self.id = doc_id
        self.doc_type: str | None = None
        self.summary: str | None = None
        self.doc_type_source: str | None = None
        self.page_count: int | None = None
        self.law_type: str | None = None
        self.act_number: str | None = None
        self.effective_year: int | None = None


class _FakeRepo:
    """Stands in for DocumentRepository — records what the service does
    without touching a real database."""

    def __init__(self) -> None:
        self.create_document_kwargs: dict[str, object] | None = None
        self.document = _FakeDocument(uuid.uuid4())
        self.added_chunks: list[object] | None = None
        self.status_calls: list[tuple[str, str | None]] = []

    def create_document(self, **kwargs: object) -> _FakeDocument:
        self.create_document_kwargs = kwargs
        return self.document

    def add_chunks(self, document_id: uuid.UUID, records: list[object]) -> None:
        self.added_chunks = records

    def mark_document_status(
        self, document_id: uuid.UUID, status: str, notes: str | None = None
    ) -> None:
        self.status_calls.append((status, notes))

    def set_remote_storage_key(self, *args: object, **kwargs: object) -> None:
        pass


class _FakeParser:
    def parse(self, path: Path) -> ParsedPdf:
        return ParsedPdf(pages=[PdfPage(page_number=1, text="ACT 1995\nSECTION 1\nBody text.")])


class _FakeChunker:
    """Returns chunks pre-populated exactly as the real LegalChunker would
    after detecting doc-level metadata from the PDF header — the real
    detection logic is tested separately in test_legal_chunker.py; this
    fake isolates the ingestion-service wiring bug from chunker internals."""

    def chunk(self, parsed: ParsedPdf, doc_title: str | None = None) -> list[LegalChunk]:
        return [
            LegalChunk(
                content="Body text.",
                section="1",
                law_type="act",
                act_number="1995",
                effective_year=1995,
            )
        ]


class _FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeVectorStore:
    def upsert(self, embeddings: list[list[float]], payloads: list[object]) -> list[str]:
        return [str(uuid.uuid4()) for _ in embeddings]


class _FakeDb:
    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


@pytest.fixture()
def isolated_upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """ingest_pdf writes the raw upload to settings.upload_dir, which
    defaults to /app/data/uploads — not writable (or meaningful) outside
    the app's own container. Point it at a pytest tmp dir instead so this
    test is portable to CI and any dev machine."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_ingest_pdf_populates_document_level_metadata(isolated_upload_dir: Path) -> None:
    repo = _FakeRepo()
    service = IngestionService(
        db=_FakeDb(),  # type: ignore[arg-type]
        parser=_FakeParser(),  # type: ignore[arg-type]
        chunker=_FakeChunker(),  # type: ignore[arg-type]
        embeddings=_FakeEmbeddings(),  # type: ignore[arg-type]
        vector_store=_FakeVectorStore(),  # type: ignore[arg-type]
    )
    service._repo = repo  # type: ignore[assignment]

    document_id, chunk_count = service.ingest_pdf(
        file_bytes=b"%PDF-1.4 fake",
        original_filename="act-1995.pdf",
        title="Some Act, 1995",
        doc_type="Act",
        visibility_scope=VisibilityScope.GLOBAL,
        confidentiality_level=ConfidentialityLevel.PUBLIC,
    )

    assert chunk_count == 1
    assert document_id == repo.document.id
    # The bug: these three were always None because create_document() ran
    # before chunking, and nothing ever wrote the detected values back.
    assert repo.document.law_type == "act"
    assert repo.document.act_number == "1995"
    assert repo.document.effective_year == 1995
    assert repo.status_calls[-1] == ("ingested", None)


class _RecordingVectorStore(_FakeVectorStore):
    def __init__(self) -> None:
        self.payloads: list[object] = []

    def upsert(self, embeddings: list[list[float]], payloads: list[object]) -> list[str]:
        self.payloads = payloads
        return super().upsert(embeddings, payloads)


def test_ingest_pdf_auto_doc_type_is_classified_before_indexing(
    isolated_upload_dir: Path,
) -> None:
    """An ``auto`` upload must land under the DETECTED type everywhere —
    the document row and every vector payload — not the sentinel."""
    repo = _FakeRepo()
    store = _RecordingVectorStore()
    service = IngestionService(
        db=_FakeDb(),  # type: ignore[arg-type]
        parser=_FakeParser(),  # type: ignore[arg-type]
        chunker=_FakeChunker(),  # type: ignore[arg-type]
        embeddings=_FakeEmbeddings(),  # type: ignore[arg-type]
        vector_store=store,  # type: ignore[arg-type]
        classifier=DocumentClassifier(llm_enabled=False),
    )
    service._repo = repo  # type: ignore[assignment]

    service.ingest_pdf(
        file_bytes=b"%PDF-1.4 fake",
        original_filename="act-1995.pdf",
        title="Some Act, 1995",
        doc_type=AUTO_DOC_TYPE,
        visibility_scope=VisibilityScope.GLOBAL,
        confidentiality_level=ConfidentialityLevel.PUBLIC,
    )

    assert repo.document.doc_type == "Act"
    assert repo.document.doc_type_source == "heuristic"
    assert repo.document.summary is None
    assert [p.doc_type for p in store.payloads] == ["Act"]
