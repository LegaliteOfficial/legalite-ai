"""Service layer for case-law intelligence endpoints.

Reads run against ``CaseRepository``. Similarity goes through
``CaseSimilarityEngine``. ``compare_cases`` is intentionally rule-based for
this first cut: it surfaces field-level differences without invoking an LLM,
so the comparison is auditable and free.
"""

from __future__ import annotations

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
from app.rag.case_similarity_engine import CaseSimilarityEngine, CaseSimilarityHit
from app.schemas.cases import (
    CaseCompareResponse,
    CaseComparisonField,
    CaseDetailOut,
    CaseListResponse,
    CaseSimilarHitOut,
    CaseSimilarResponse,
    CaseSummarizeResponse,
    CaseSummaryOut,
    CitationGraphNode,
    CitationGraphResponse,
    PartyOut,
)
from app.services.citation_resolution_service import get_citation_graph
from app.vector_store.qdrant_client import QdrantVectorStore

logger = get_logger(__name__)


@dataclass(slots=True)
class _CompareContext:
    cases: list[LegalCase]
    by_id: dict[str, LegalCase]


class CaseIntelligenceService:
    def __init__(
        self,
        db: Session,
        *,
        embeddings: EmbeddingService | None = None,
        vector_store: QdrantVectorStore | None = None,
        similarity: CaseSimilarityEngine | None = None,
    ) -> None:
        self._db = db
        self._repo = CaseRepository(db)
        if similarity is not None:
            self._similarity = similarity
        else:
            from app.vector_store.qdrant_client import get_vector_store

            self._similarity = CaseSimilarityEngine(
                db=db,
                embeddings=embeddings or EmbeddingService(),
                vector_store=vector_store or get_vector_store(),
            )

    # ---- /cases ------------------------------------------------------------

    def list_cases(
        self,
        *,
        court: str | None = None,
        topic: str | None = None,
        limit: int = 50,
        offset: int = 0,
        tenant: TenantContext | None = None,
    ) -> CaseListResponse:
        org_id = tenant.organization_id if tenant else None
        cases = self._repo.list_cases(
            court=court,
            topic=topic,
            limit=limit,
            offset=offset,
            organization_id=org_id,
        )
        return CaseListResponse(
            cases=[_to_summary(c) for c in cases],
            limit=limit,
            offset=offset,
        )

    # ---- /cases/{id} -------------------------------------------------------

    def get_case_detail(
        self,
        case_id: uuid.UUID,
        *,
        tenant: TenantContext | None = None,
    ) -> CaseDetailOut | None:
        org_id = tenant.organization_id if tenant else None
        case = self._repo.get_case(case_id, organization_id=org_id)
        return None if case is None else _to_detail(case)

    # ---- /cases/similar ----------------------------------------------------

    def find_similar(
        self,
        *,
        case_id: uuid.UUID | None,
        query: str | None,
        top_k: int | None = None,
        tenant: TenantContext | None = None,
    ) -> CaseSimilarResponse:
        if case_id is None and not (query and query.strip()):
            raise ValueError("Provide either case_id or query")

        org_id = tenant.organization_id if tenant else None
        if case_id is not None:
            hits = self._similarity.find_by_case(case_id, top_k=top_k, tenant=tenant)
            source_case = self._repo.get_case(case_id, organization_id=org_id)
            query_topics = list(source_case.legal_topics or []) if source_case else []
        else:
            assert query is not None  # narrowed by the check above
            hits = self._similarity.find_by_text(query, top_k=top_k, tenant=tenant)
            query_topics = _topics_from_hits(hits)

        return CaseSimilarResponse(
            query_topics=query_topics,
            hits=[_to_hit_out(h) for h in hits],
        )

    # ---- /cases/summarize --------------------------------------------------

    def summarize(
        self,
        case_id: uuid.UUID,
        *,
        tenant: TenantContext | None = None,
    ) -> CaseSummarizeResponse | None:
        org_id = tenant.organization_id if tenant else None
        case = self._repo.get_case(case_id, organization_id=org_id)
        if case is None:
            return None
        return CaseSummarizeResponse(case=_to_detail(case))

    # ---- /cases/compare ----------------------------------------------------

    def citation_graph(
        self,
        case_id: uuid.UUID,
        *,
        tenant: TenantContext | None = None,
    ) -> CitationGraphResponse | None:
        org_id = tenant.organization_id if tenant else None
        case = self._repo.get_case(case_id, organization_id=org_id)
        if case is None:
            return None
        settings = get_settings()
        graph = get_citation_graph(
            self._db,
            case_id,
            organization_id=org_id,
            max_chain_depth=settings.citation_resolution_chain_depth,
        )
        return CitationGraphResponse(
            cites=[_graph_node(n) for n in graph["cites"]],
            cited_by=[_graph_node(n) for n in graph["cited_by"]],
            precedent_chain=[_graph_node(n) for n in graph["precedent_chain"]],
        )

    def compare(
        self,
        case_ids: Sequence[uuid.UUID],
        *,
        tenant: TenantContext | None = None,
    ) -> CaseCompareResponse | None:
        org_id = tenant.organization_id if tenant else None
        cases = self._repo.get_cases(list(case_ids), organization_id=org_id)
        if len(cases) < 2:
            return None

        # Preserve user-supplied order so the response columns match the request.
        by_id = {str(c.id): c for c in cases}
        ordered = [by_id[str(cid)] for cid in case_ids if str(cid) in by_id]
        ctx = _CompareContext(cases=ordered, by_id={str(c.id): c for c in ordered})

        return CaseCompareResponse(
            cases=[_to_detail(c) for c in ordered],
            differences=_diff_fields(ctx),
            overlap=_topic_overlap_all(ordered),
        )


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------


def _to_summary(case: LegalCase) -> CaseSummaryOut:
    return CaseSummaryOut(
        id=case.id,
        document_id=case.document_id,
        case_title=case.case_title,
        case_number=case.case_number,
        court=case.court,
        judges=list(case.judges or []),
        decision_date=case.decision_date,
        outcome=case.outcome,
        legal_topics=list(case.legal_topics or []),
        jurisdiction=case.jurisdiction,
        created_at=case.created_at,
    )


def _to_detail(case: LegalCase) -> CaseDetailOut:
    base = _to_summary(case).model_dump()
    return CaseDetailOut(
        **base,
        holding=case.holding,
        facts_summary=case.facts_summary,
        parties=[
            PartyOut(role=p.role, party_name=p.party_name)  # type: ignore[arg-type]
            for p in (case.parties or [])
        ],
    )


def _to_hit_out(hit: CaseSimilarityHit) -> CaseSimilarHitOut:
    return CaseSimilarHitOut(
        case=_to_summary(hit.case),
        score=hit.score,
        semantic_score=hit.semantic_score,
        topic_overlap=hit.topic_overlap,
        shared_topics=hit.shared_topics,
        matched_chunks=hit.matched_chunks,
    )


def _graph_node(n: object) -> CitationGraphNode:
    """Adapter from the resolver's GraphNode dataclass to the API schema."""
    return CitationGraphNode(
        case_id=n.case_id,
        case_title=n.case_title,
        case_number=getattr(n, "case_number", None),
        court=getattr(n, "court", None),
        confidence=float(getattr(n, "confidence", 0.0)),
    )


def _topics_from_hits(hits: Sequence[CaseSimilarityHit]) -> list[str]:
    seen: dict[str, None] = {}
    for h in hits:
        for t in h.shared_topics:
            seen.setdefault(t, None)
    return list(seen.keys())


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


_COMPARE_FIELDS: list[tuple[str, str]] = [
    ("court", "Court"),
    ("decision_date", "Decision date"),
    ("outcome", "Outcome"),
    ("holding", "Holding"),
    ("case_number", "Case number"),
]


def _diff_fields(ctx: _CompareContext) -> list[CaseComparisonField]:
    diffs: list[CaseComparisonField] = []
    for attr, label in _COMPARE_FIELDS:
        values: dict[str, str | None] = {}
        seen_values: set[str | None] = set()
        for case in ctx.cases:
            v = getattr(case, attr)
            v_str = None if v is None else str(v)
            values[str(case.id)] = v_str
            seen_values.add(v_str)
        # Only surface fields where at least one case differs from another.
        if len(seen_values) > 1:
            diffs.append(CaseComparisonField(field=label, values=values))
    return diffs


def _topic_overlap_all(cases: Sequence[LegalCase]) -> list[str]:
    sets = [set(c.legal_topics or []) for c in cases]
    if not sets:
        return []
    intersection = sets[0]
    for s in sets[1:]:
        intersection &= s
    return sorted(intersection)
