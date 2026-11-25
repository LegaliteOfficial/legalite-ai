"""Find cases similar to a fact pattern or an existing case.

Two query modes:

- **By query text** ("a tenant who fell into rent arrears..."): we embed the
  text, vector-search the chunk corpus filtered to ``law_type=case_law``, and
  fold the per-chunk hits up to their ``LegalCase`` via ``document_id``.
- **By case_id**: we build the query from the case's
  ``case_title + holding + facts_summary`` (whatever is populated), then run
  the same pipeline excluding the source case.

Final score is a weighted blend of:

- semantic similarity (max chunk score for that case, rescaled 0..1), and
- topic overlap (count of shared topics ÷ |query topics|).

Weights live in settings so we can tune without code changes. We deliberately
don't pull in the Cohere reranker here — at case-level granularity the score
gap between candidates is already large, and reranking adds latency for
marginal gain.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.tenant_context import TenantContext
from app.db.models.legal_case import LegalCase
from app.db.repositories.case_repository import CaseRepository
from app.ingestion.embedding_service import EmbeddingService
from app.vector_store.qdrant_client import QdrantVectorStore, build_tenant_filter

logger = get_logger(__name__)


@dataclass(slots=True)
class CaseSimilarityHit:
    case: LegalCase
    score: float
    semantic_score: float
    topic_overlap: int
    shared_topics: list[str]
    matched_chunks: int


class CaseSimilarityEngine:
    def __init__(
        self,
        *,
        db: Session,
        embeddings: EmbeddingService,
        vector_store: QdrantVectorStore,
    ) -> None:
        self._db = db
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._cases = CaseRepository(db)

    # ---- public API --------------------------------------------------------

    def find_by_text(
        self,
        text: str,
        *,
        top_k: int | None = None,
        topics: Sequence[str] | None = None,
        exclude_case_ids: Sequence[uuid.UUID] = (),
        tenant: TenantContext | None = None,
    ) -> list[CaseSimilarityHit]:
        settings = get_settings()
        k = top_k or settings.case_similarity_top_k
        query_topics = list(topics) if topics else _extract_topics(text)

        org_id = tenant.organization_id if tenant else None
        query_vec = self._embeddings.embed_query(text)
        # Pull a generous candidate window so we have enough chunks to fold up
        # into distinct cases. Cap at 4x to avoid pathological corpora.
        candidate_pool = max(k * 6, 30)
        qfilter = build_tenant_filter(organization_id=org_id)
        hits = self._vector_store.search(query_vec, top_k=candidate_pool, query_filter=qfilter)

        # Group chunk hits by document_id and take the max chunk score per doc.
        # That score becomes the "semantic_score" for the candidate case.
        doc_to_score: dict[str, float] = {}
        doc_to_match_count: dict[str, int] = {}
        for h in hits:
            if (h.payload.get("law_type") or "").lower() != "case_law":
                continue
            doc_id = h.payload.get("document_id")
            if not doc_id:
                continue
            doc_to_score[doc_id] = max(doc_to_score.get(doc_id, 0.0), float(h.score))
            doc_to_match_count[doc_id] = doc_to_match_count.get(doc_id, 0) + 1

        if not doc_to_score:
            return []

        try:
            doc_uuids = [uuid.UUID(d) for d in doc_to_score]
        except ValueError:
            doc_uuids = []
        cases = self._cases_by_document_ids(doc_uuids, organization_id=org_id)

        exclude = set(exclude_case_ids)
        scored: list[CaseSimilarityHit] = []
        for case in cases:
            if case.id in exclude:
                continue
            semantic = doc_to_score.get(str(case.document_id), 0.0)
            shared = _shared_topics(query_topics, case.legal_topics or [])
            topic_norm = len(shared) / max(len(query_topics), 1) if query_topics else 0.0
            score = _blend(
                semantic_score=semantic,
                topic_score=topic_norm,
                semantic_weight=settings.case_similarity_semantic_weight,
                topic_weight=settings.case_similarity_topic_weight,
            )
            scored.append(
                CaseSimilarityHit(
                    case=case,
                    score=score,
                    semantic_score=semantic,
                    topic_overlap=len(shared),
                    shared_topics=shared,
                    matched_chunks=doc_to_match_count.get(str(case.document_id), 0),
                )
            )

        scored.sort(key=lambda h: h.score, reverse=True)
        result = scored[:k]
        logger.info(
            "case_similarity.ranked",
            candidates=len(scored),
            returned=len(result),
            query_topics=query_topics,
        )
        return result

    def find_by_case(
        self,
        case_id: uuid.UUID,
        *,
        top_k: int | None = None,
        tenant: TenantContext | None = None,
    ) -> list[CaseSimilarityHit]:
        org_id = tenant.organization_id if tenant else None
        case = self._cases.get_case(case_id, organization_id=org_id)
        if case is None:
            return []
        query = _build_query_from_case(case)
        if not query.strip():
            return []
        return self.find_by_text(
            query,
            top_k=top_k,
            topics=case.legal_topics or [],
            exclude_case_ids=[case.id],
            tenant=tenant,
        )

    # ---- internals ---------------------------------------------------------

    def _cases_by_document_ids(
        self,
        document_ids: Sequence[uuid.UUID],
        *,
        organization_id: uuid.UUID | None = None,
    ) -> list[LegalCase]:
        if not document_ids:
            return []
        from sqlalchemy import or_, select

        from app.db.enums import VisibilityScope

        stmt = select(LegalCase).where(LegalCase.document_id.in_(list(document_ids)))
        if organization_id is None:
            stmt = stmt.where(LegalCase.visibility_scope == VisibilityScope.GLOBAL.value)
        else:
            stmt = stmt.where(
                or_(
                    LegalCase.visibility_scope == VisibilityScope.GLOBAL.value,
                    (LegalCase.visibility_scope == VisibilityScope.TENANT_PRIVATE.value)
                    & (LegalCase.organization_id == organization_id),
                )
            )
        return list(self._db.scalars(stmt).all())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blend(
    *,
    semantic_score: float,
    topic_score: float,
    semantic_weight: float,
    topic_weight: float,
) -> float:
    total = semantic_weight + topic_weight or 1.0
    sw = semantic_weight / total
    tw = topic_weight / total
    # Qdrant cosine scores already sit in [0, 1] for normalized embeddings.
    semantic_clamped = max(0.0, min(semantic_score, 1.0))
    topic_clamped = max(0.0, min(topic_score, 1.0))
    return sw * semantic_clamped + tw * topic_clamped


def _shared_topics(query_topics: Sequence[str], case_topics: Sequence[str]) -> list[str]:
    q = {(t or "").lower() for t in query_topics}
    c = {(t or "").lower() for t in case_topics}
    return sorted(q & c)


def _build_query_from_case(case: LegalCase) -> str:
    parts: list[str] = []
    if case.case_title:
        parts.append(case.case_title)
    if case.holding:
        parts.append(case.holding)
    if case.facts_summary:
        parts.append(case.facts_summary)
    return "\n".join(parts)


# Shared topic vocabulary — kept in sync with chunker + extractor.
# Duplicating the list (rather than importing) keeps this module lightweight
# and free of side effects (the extractor module triggers config loading).
_TOPIC_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
    (
        "employment",
        re.compile(r"\b(employer|employee|dismissal|wages?|labou?r|maternity)\b", re.IGNORECASE),
    ),
    ("landlord_tenant", re.compile(r"\b(landlord|tenant|tenancy|eviction|rent)\b", re.IGNORECASE)),
    (
        "criminal",
        re.compile(r"\b(offen[cs]e|imprisonment|conviction|warrant|arrest|bail)\b", re.IGNORECASE),
    ),
    (
        "constitutional",
        re.compile(r"\b(fundamental\s+rights?|liberty|equality|due\s+process)\b", re.IGNORECASE),
    ),
    ("family", re.compile(r"\b(marriage|divorce|custody|guardian|matrimonial)\b", re.IGNORECASE)),
    (
        "contract",
        re.compile(r"\b(contract|breach|consideration|offer|acceptance)\b", re.IGNORECASE),
    ),
    (
        "property",
        re.compile(r"\b(conveyance|easement|deed|title|mortgage|land|trespass)\b", re.IGNORECASE),
    ),
    ("tort", re.compile(r"\b(negligence|nuisance|defamation|duty\s+of\s+care)\b", re.IGNORECASE)),
    (
        "corporate",
        re.compile(r"\b(company|directors?|shareholders?|winding\s+up|corporate)\b", re.IGNORECASE),
    ),
]


def _extract_topics(text: str) -> list[str]:
    return [topic for topic, pat in _TOPIC_KEYWORDS if pat.search(text)]
