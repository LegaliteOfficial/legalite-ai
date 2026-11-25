"""CaseSimilarityEngine — semantic-vs-topic blending logic.

We test the engine with stubbed dependencies (fake embedder, fake vector
store, fake repo) so the unit tests stay DB-free. The blend / ranking
behavior is what we actually want to validate; the surrounding I/O is
covered by integration tests once a real Postgres is in CI.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from app.rag.case_similarity_engine import CaseSimilarityEngine
from app.vector_store.qdrant_client import VectorHit


@dataclass
class _FakeCase:
    id: uuid.UUID
    document_id: uuid.UUID
    case_title: str
    legal_topics: list[str] = field(default_factory=list)
    court: str | None = None
    judges: list[str] | None = None
    decision_date: Any = None
    outcome: str | None = None
    case_number: str | None = None
    holding: str | None = None
    facts_summary: str | None = None
    jurisdiction: str = "GH"
    created_at: Any = None
    parties: list[Any] = field(default_factory=list)


class _StubEmbeddings:
    def embed_query(self, _: str) -> list[float]:
        return [0.0] * 1536


class _StubVectorStore:
    def __init__(self, hits: list[VectorHit]) -> None:
        self._hits = hits

    def search(
        self,
        _query: list[float],
        top_k: int,
        *,
        query_filter: Any = None,
    ) -> list[VectorHit]:
        # Accept and ignore the tenant filter — these tests cover ranking,
        # not the boundary itself (that lives in test_tenant_filter.py).
        return self._hits[:top_k]


def _hit(doc_id: str, score: float, law_type: str = "case_law") -> VectorHit:
    return VectorHit(
        score=score,
        point_id=str(uuid.uuid4()),
        payload={
            "document_id": doc_id,
            "law_type": law_type,
            "content": "...",
        },
    )


def _build_engine(hits: list[VectorHit], cases: list[_FakeCase]) -> CaseSimilarityEngine:
    engine = CaseSimilarityEngine.__new__(CaseSimilarityEngine)
    engine._db = MagicMock()  # type: ignore[attr-defined]
    engine._embeddings = _StubEmbeddings()  # type: ignore[attr-defined]
    engine._vector_store = _StubVectorStore(hits)  # type: ignore[attr-defined]
    engine._cases = MagicMock()  # type: ignore[attr-defined]

    # Override _cases_by_document_ids to return our fakes directly.
    # Accepts the optional organization_id kwarg added when tenant context
    # was wired through the engine — tests don't exercise the tenancy
    # branch, so we ignore it.
    def _by_docs(doc_ids, **_kwargs):
        wanted = set(doc_ids)
        return [c for c in cases if c.document_id in wanted]

    engine._cases_by_document_ids = _by_docs  # type: ignore[attr-defined]
    return engine


def test_ranks_by_blended_score() -> None:
    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()
    case_a = _FakeCase(
        id=uuid.uuid4(),
        document_id=doc_a,
        case_title="A v B (employment dismissal)",
        legal_topics=["employment"],
    )
    case_b = _FakeCase(
        id=uuid.uuid4(),
        document_id=doc_b,
        case_title="C v D (criminal arrest)",
        legal_topics=["criminal"],
    )

    # Case A: lower semantic but topic match. Case B: higher semantic but
    # topic mismatch. With default weights (0.65 semantic / 0.35 topic) and
    # a strong topic signal, A should still come out ahead when semantic is
    # comparable.
    hits = [
        _hit(str(doc_a), score=0.70),
        _hit(str(doc_b), score=0.72),
    ]

    engine = _build_engine(hits, [case_a, case_b])
    results = engine.find_by_text(
        "employee dismissed without notice for pregnancy",
        top_k=5,
    )

    assert len(results) == 2
    # Topic overlap pushes A above B even though B has slightly higher semantic.
    assert results[0].case.id == case_a.id
    assert results[0].topic_overlap == 1
    assert results[1].topic_overlap == 0


def test_filters_out_non_case_law_chunks() -> None:
    doc_case = uuid.uuid4()
    doc_act = uuid.uuid4()
    case = _FakeCase(
        id=uuid.uuid4(),
        document_id=doc_case,
        case_title="A v B",
        legal_topics=["contract"],
    )
    hits = [
        _hit(str(doc_act), score=0.99, law_type="act"),  # should be ignored
        _hit(str(doc_case), score=0.50, law_type="case_law"),
    ]
    engine = _build_engine(hits, [case])
    results = engine.find_by_text("breach of contract", top_k=5)
    assert len(results) == 1
    assert results[0].case.id == case.id


def test_returns_empty_when_no_case_law_hits() -> None:
    hits = [_hit(str(uuid.uuid4()), score=0.9, law_type="constitution")]
    engine = _build_engine(hits, [])
    assert engine.find_by_text("anything", top_k=3) == []


def test_aggregates_multiple_chunks_to_one_case() -> None:
    doc = uuid.uuid4()
    case = _FakeCase(
        id=uuid.uuid4(),
        document_id=doc,
        case_title="X v Y",
        legal_topics=[],
    )
    # Three chunks from the same case — the case should appear once with the
    # max chunk score and matched_chunks == 3.
    hits = [
        _hit(str(doc), score=0.40),
        _hit(str(doc), score=0.85),
        _hit(str(doc), score=0.55),
    ]
    engine = _build_engine(hits, [case])
    results = engine.find_by_text("question", top_k=5)
    assert len(results) == 1
    assert results[0].matched_chunks == 3
    assert results[0].semantic_score == 0.85
