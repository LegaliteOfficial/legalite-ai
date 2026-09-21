"""Vector retrieval over the Ghana legal corpus.

`RetrievedPassage` is the shared shape used by every retriever (vector,
keyword, hybrid). Keeping one type makes the rerank → confidence → prompt
pipeline trivially composable.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.ingestion.embedding_service import EmbeddingService
from app.vector_store.qdrant_client import QdrantVectorStore, VectorHit


@dataclass(slots=True)
class RetrievedPassage:
    score: float
    content: str
    law: str | None
    article: str | None
    section: str | None
    chapter: str | None
    clause: str | None
    subsection: str | None
    page_start: int | None
    page_end: int | None
    vector_id: str
    # Extended metadata (None when retrieved from a source that doesn't carry it)
    part: str | None = None
    law_type: str | None = None
    act_number: str | None = None
    effective_year: int | None = None
    legal_topic: str | None = None
    # Provenance: which retrievers surfaced this passage
    source: str = "vector"  # vector | keyword | hybrid
    # Rerank score (set by the Reranker; None if rerank disabled)
    rerank_score: float | None = None
    # The retriever-native score before Reciprocal Rank Fusion mutates
    # ``score``. For vector hits this is the cosine similarity (0-1 scale,
    # comparable to ConfidenceEngine's cosine thresholds in
    # app/core/config.py). For keyword hits this is the Postgres FTS rank,
    # which is on a different scale — so callers should only treat
    # ``original_score`` as a confidence signal when the passage's
    # pre-fusion ``source`` was ``"vector"``.
    original_score: float | None = None
    # Tenancy provenance — preserved across RRF so the structured /ask
    # response can group passages by source_scope (public vs private).
    organization_id: str | None = None
    visibility_scope: str = "global"
    document_origin: str | None = None
    # Authority hierarchy (set by ``app.rag.authority.annotate``). When the
    # passage hasn't been classified yet these stay at their defaults — never
    # rely on them being correct without a prior ``annotate()`` call.
    legal_authority_rank: int = 0
    source_scope: str = "global"  # global | tenant_private
    authority_tier: str | None = None
    # Phase 5 M9 — character offsets for UI source highlighting.
    start_char: int | None = None
    end_char: int | None = None
    # Source-document linkage. ``document_id`` is the foreign key the
    # frontend uses to hit GET /documents/{id} for the "view source"
    # drawer. ``chunk_id`` lets the drawer scroll-to and highlight the
    # exact passage that was cited (it matches LegalChunk.id, NOT the
    # Qdrant point id in ``vector_id``). Both are None for legacy
    # passages whose payload predates Phase B.
    document_id: str | None = None
    chunk_id: str | None = None

    @classmethod
    def from_hit(cls, hit: VectorHit) -> RetrievedPassage:
        p = hit.payload
        return cls(
            score=hit.score,
            original_score=hit.score,
            content=str(p.get("content", "")),
            law=p.get("law") or p.get("title"),
            article=p.get("article"),
            section=p.get("section"),
            chapter=p.get("chapter"),
            clause=p.get("clause"),
            subsection=p.get("subsection"),
            page_start=p.get("page_start"),
            page_end=p.get("page_end"),
            vector_id=hit.point_id,
            part=p.get("part"),
            law_type=p.get("law_type") or p.get("doc_type"),
            act_number=p.get("act_number"),
            effective_year=p.get("effective_year"),
            legal_topic=p.get("legal_topic"),
            source="vector",
            organization_id=p.get("organization_id"),
            visibility_scope=p.get("visibility_scope") or "global",
            document_origin=p.get("document_origin"),
            start_char=p.get("start_char"),
            end_char=p.get("end_char"),
            document_id=p.get("document_id"),
            chunk_id=p.get("chunk_id"),
        )


class Retriever:
    """Pure dense-vector retriever. Kept for backward compatibility / fallback."""

    def __init__(
        self,
        embeddings: EmbeddingService,
        vector_store: QdrantVectorStore,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedPassage]:
        settings = get_settings()
        k = top_k or settings.rag_top_k
        query_vec = self._embeddings.embed_query(question)
        hits = self._vector_store.search(query_vec, top_k=k)
        return [RetrievedPassage.from_hit(h) for h in hits]
