"""Hybrid retrieval: dense vectors (Qdrant) + keyword/BM25-style FTS (Postgres).

Why both:
- Vector search captures *semantic* matches: "boss fired me while pregnant"
  matches "maternity dismissal".
- Keyword search captures *exact* legal terms: "warrant", "section 17(2)(a)",
  "Act 651". Legal language is dense with specific terminology that paraphrase
  models can miss.

We merge results via **Reciprocal Rank Fusion** (RRF):

    RRF(d) = Σ 1 / (k + rank_i(d))

where `k` is a smoothing constant (default 60). RRF is rank-only — it does
not rely on the underlying scores being comparable — which makes it robust
when one source returns cosine similarity and another returns ts_rank_cd.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models.legal_chunk import LegalChunk
from app.db.repositories.document_repository import DocumentRepository
from app.ingestion.embedding_service import EmbeddingService
from app.rag.retriever import RetrievedPassage
from app.vector_store.qdrant_client import QdrantVectorStore

logger = get_logger(__name__)


@dataclass(slots=True)
class HybridResult:
    passages: list[RetrievedPassage]
    n_vector: int
    n_keyword: int
    n_fused: int


class HybridRetriever:
    def __init__(
        self,
        *,
        db: Session,
        embeddings: EmbeddingService,
        vector_store: QdrantVectorStore,
    ) -> None:
        self._db = db
        self._repo = DocumentRepository(db)
        self._embeddings = embeddings
        self._vector_store = vector_store

    # ---- public API --------------------------------------------------------

    def retrieve(
        self,
        question: str,
        *,
        law_type: str | None = None,
        top_k_final: int | None = None,
    ) -> HybridResult:
        settings = get_settings()
        topk_v = settings.hybrid_topk_vector
        topk_k = settings.hybrid_topk_keyword
        topk_f = top_k_final or settings.hybrid_topk_fused

        vector_passages = self._vector_search(question, topk_v)
        keyword_passages = self._keyword_search(question, topk_k, law_type=law_type)

        fused = _reciprocal_rank_fusion(
            [vector_passages, keyword_passages],
            k=settings.rrf_k,
        )[:topk_f]

        logger.info(
            "hybrid.retrieved",
            question_len=len(question),
            vector=len(vector_passages),
            keyword=len(keyword_passages),
            fused=len(fused),
            law_type=law_type,
        )
        return HybridResult(
            passages=fused,
            n_vector=len(vector_passages),
            n_keyword=len(keyword_passages),
            n_fused=len(fused),
        )

    # ---- internals ---------------------------------------------------------

    def _vector_search(self, question: str, top_k: int) -> list[RetrievedPassage]:
        query_vec = self._embeddings.embed_query(question)
        hits = self._vector_store.search(query_vec, top_k=top_k)
        return [RetrievedPassage.from_hit(h) for h in hits]

    def _keyword_search(
        self, question: str, top_k: int, law_type: str | None
    ) -> list[RetrievedPassage]:
        rows = self._repo.keyword_search(question, top_k=top_k, law_type=law_type)
        return [_chunk_to_passage(c, score=s) for c, s in rows]


def _chunk_to_passage(chunk: LegalChunk, score: float) -> RetrievedPassage:
    return RetrievedPassage(
        score=score,
        content=chunk.content,
        law=chunk.title,
        article=chunk.article,
        section=chunk.section,
        chapter=chunk.chapter,
        clause=chunk.clause,
        subsection=chunk.subsection,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        vector_id=chunk.vector_id or str(chunk.id),
        part=chunk.part,
        law_type=chunk.law_type,
        act_number=chunk.act_number,
        effective_year=chunk.effective_year,
        legal_topic=chunk.legal_topic,
        source="keyword",
    )


def _passage_key(p: RetrievedPassage) -> str:
    """Stable identity for fusion / dedup. Prefer vector_id; fall back to content hash."""
    if p.vector_id and not p.vector_id.startswith("None"):
        try:
            return str(uuid.UUID(p.vector_id))
        except ValueError:
            return p.vector_id
    return f"text:{hash(p.content)}"


def _reciprocal_rank_fusion(
    rankings: Iterable[list[RetrievedPassage]], k: int
) -> list[RetrievedPassage]:
    """Combine multiple ranked lists into a single ranking using RRF."""
    score_by_key: dict[str, float] = {}
    passage_by_key: dict[str, RetrievedPassage] = {}

    for ranking in rankings:
        for rank, p in enumerate(ranking, start=1):
            key = _passage_key(p)
            score_by_key[key] = score_by_key.get(key, 0.0) + 1.0 / (k + rank)
            # Keep the richest version of the passage we see (vector hit has full payload).
            existing = passage_by_key.get(key)
            if existing is None or (existing.source == "keyword" and p.source == "vector"):
                passage_by_key[key] = p

    fused_items = sorted(
        passage_by_key.values(),
        key=lambda p: score_by_key[_passage_key(p)],
        reverse=True,
    )
    # Mark fused source + overwrite score with RRF score so downstream stages
    # see a comparable signal.
    out: list[RetrievedPassage] = []
    for p in fused_items:
        key = _passage_key(p)
        p.score = score_by_key[key]
        p.source = "hybrid"
        out.append(p)
    return out
