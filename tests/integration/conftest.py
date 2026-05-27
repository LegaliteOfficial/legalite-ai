"""Fixtures for the live integration suite.

These tests hit a **real** Postgres and a **real** Qdrant — the same ones
the production code talks to. They use:

- A deterministic fake embedding (SHA-256 → 1536 floats in [-1, 1]) so we
  don't burn OpenAI tokens and tests stay reproducible.
- A run-scoped Qdrant collection name (``legalite_test_<random>``) so
  test points never live in the production ``ghana_legal_docs``
  collection.
- A teardown that DELETEs every row created during the test by tracking
  inserted document_ids + org_ids — never relies on transactions because
  the production code commits explicitly.

Skipped automatically when ``LEGALITE_INTEGRATION_ENABLED=true`` is not
set in the environment, so ``pytest -q`` (fast unit suite) stays free of
live-network dependencies. The CI gate that exercises real isolation
runs with ``LEGALITE_INTEGRATION_ENABLED=true pytest -q tests/integration``.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip everything in tests/integration unless explicitly enabled."""
    if os.environ.get("LEGALITE_INTEGRATION_ENABLED", "").lower() in {"1", "true", "yes"}:
        return
    skip_marker = pytest.mark.skip(
        reason="set LEGALITE_INTEGRATION_ENABLED=true to run integration tests"
    )
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Deterministic fake embedding service
# ---------------------------------------------------------------------------


_DIM = 1536  # matches text-embedding-3-small


def _hashed_vector(text: str) -> list[float]:
    """SHA-256-based deterministic vector. Two identical inputs always
    produce the same vector; small differences produce small but distinct
    vectors. Good enough for retrieval-shape assertions."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Stretch the 32-byte digest across 1536 dims via xor-folding the
    # sequence index. Normalise to roughly unit length so cosine returns
    # values in [-1, 1] and the existing thresholds still trigger.
    vec = []
    for i in range(_DIM):
        byte = h[i % 32]
        vec.append((byte - 127.5) / 127.5)
    # Cheap L2-normalise so Qdrant cosine ranks correlate with dot product.
    import math

    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class _FakeEmbeddings:
    """Stand-in for app.ingestion.embedding_service.EmbeddingService."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_hashed_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _hashed_vector(text)


@pytest.fixture
def fake_embeddings() -> _FakeEmbeddings:
    return _FakeEmbeddings()


# ---------------------------------------------------------------------------
# Live Qdrant collection (one per test run)
# ---------------------------------------------------------------------------


@dataclass
class _TestVectorStore:
    """Wraps a fresh ``QdrantVectorStore`` pointed at a run-scoped collection."""

    collection_name: str
    store: object  # QdrantVectorStore — Any so import stays lazy

    def cleanup(self) -> None:
        try:
            self.store._client.delete_collection(collection_name=self.collection_name)  # type: ignore[attr-defined]
        except Exception:
            pass


@pytest.fixture
def live_vector_store() -> Iterator[_TestVectorStore]:
    from app.vector_store.qdrant_client import QdrantVectorStore

    collection = f"legalite_test_{uuid.uuid4().hex[:12]}"
    store = QdrantVectorStore()
    # Point the new instance at our throwaway collection.
    store._collection = collection  # type: ignore[attr-defined]
    store.ensure_collection()

    handle = _TestVectorStore(collection_name=collection, store=store)
    yield handle
    handle.cleanup()


# ---------------------------------------------------------------------------
# Live Postgres session + per-test row tracking
# ---------------------------------------------------------------------------


@dataclass
class _Tracker:
    """Collects rows the test created so the teardown can DELETE them."""

    document_ids: list[uuid.UUID] = field(default_factory=list)
    org_ids: list[uuid.UUID] = field(default_factory=list)


@pytest.fixture
def db_session() -> Iterator[object]:
    from app.db.session import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def row_tracker(db_session) -> Iterator[_Tracker]:
    from sqlalchemy import text

    tracker = _Tracker()
    yield tracker

    # Strict cleanup — by document_id (cascades to chunks/cases/parties/citations
    # via FK ON DELETE CASCADE) and by org_id for chat_sessions + ingestion_jobs.
    try:
        if tracker.document_ids:
            ids = [str(d) for d in tracker.document_ids]
            db_session.execute(
                text("DELETE FROM legal_documents WHERE id = ANY(:ids)"),
                {"ids": ids},
            )
        if tracker.org_ids:
            ids = [str(o) for o in tracker.org_ids]
            db_session.execute(
                text("DELETE FROM chat_sessions WHERE organization_id = ANY(:ids)"),
                {"ids": ids},
            )
            db_session.execute(
                text("DELETE FROM ingestion_jobs WHERE organization_id = ANY(:ids)"),
                {"ids": ids},
            )
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise
