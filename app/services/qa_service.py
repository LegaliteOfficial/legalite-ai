"""Top-level Q&A orchestration.

Pipeline (v2):
  question → memory.load
           → MultiDocumentReasoner.gather (hybrid retrieval per law_type + rerank)
           → AnswerGenerator.generate (refusal-aware, grounded prompt)
           → citation_formatter (dedup + sources_used)
           → memory.append
           → AskResponse

Every stage is injectable so tests and the evaluation harness can substitute
deterministic stubs.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator

import structlog
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.core.telemetry import (
    observe,
    record_ask,
    record_refusal,
    record_tenant_request,
    retrieval_latency,
    traced,
    update_source_ratio,
)
from app.core.tenant_context import TenantContext
from app.db.repositories.chat_repository import ChatRepository
from app.ingestion.embedding_service import EmbeddingService
from app.rag.answer_generator import AnswerGenerator, GeneratedAnswer
from app.rag.citation_formatter import format_citations, format_sources
from app.rag.conversational_responder import ConversationalResponder
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.legal_memory import LegalMemory
from app.rag.multi_document_reasoner import MultiDocumentReasoner
from app.rag.prompt_builder import DISCLAIMER, REFUSAL_TEXT
from app.rag.query_planner import (
    CONVERSATIONAL_INTENTS,
    QueryIntent,
    QueryPlan,
    QueryPlanner,
)
from app.rag.query_rewriter import QueryRewriter
from app.rag.streaming import SSEEvent
from app.rag.reranker import BaseReranker, build_reranker
from app.schemas.qa import AskResponse, Citation, StructuredAnswer
from app.vector_store.qdrant_client import QdrantVectorStore

logger = get_logger(__name__)


class QAService:
    def __init__(
        self,
        db: Session,
        *,
        embeddings: EmbeddingService | None = None,
        vector_store: QdrantVectorStore | None = None,
        reranker: BaseReranker | None = None,
        generator: AnswerGenerator | None = None,
        planner: QueryPlanner | None = None,
        rewriter: QueryRewriter | None = None,
        conversational: ConversationalResponder | None = None,
    ) -> None:
        from app.vector_store.qdrant_client import get_vector_store

        embeddings = embeddings or EmbeddingService()
        vector_store = vector_store or get_vector_store()

        self._retriever = HybridRetriever(
            db=db, embeddings=embeddings, vector_store=vector_store
        )
        self._reranker = reranker or build_reranker()
        # The rewriter is constructed eagerly so its in-process cache is
        # shared across requests on the same QAService instance.
        self._rewriter = rewriter if rewriter is not None else QueryRewriter()
        self._reasoner = MultiDocumentReasoner(
            retriever=self._retriever,
            reranker=self._reranker,
            rewriter=self._rewriter,
        )
        self._generator = generator or AnswerGenerator()
        self._memory = LegalMemory(ChatRepository(db))
        self._planner = planner or QueryPlanner()
        self._conversational = conversational or ConversationalResponder()

    def ask(
        self,
        question: str,
        session_id: str | None = None,
        *,
        tenant: TenantContext | None = None,
    ) -> AskResponse:
        t0 = time.perf_counter()
        sid = self._memory.get_or_create_session(session_id, tenant=tenant)
        with structlog.contextvars.bound_contextvars(
            session_id=str(sid),
            tenant_org=str(tenant.organization_id) if tenant and tenant.organization_id else None,
        ):
            memory = self._memory.load_recent(sid)
            self._memory.append_user(sid, question)

            # Phase 5 M2 — classify intent. We log the result for observability
            # and short-circuit on INSUFFICIENT_CONTEXT (ambiguous pronoun
            # with no memory). Other intents still flow through the unified
            # retriever; a future increment can branch CASE_SIMILARITY /
            # CASE_SUMMARY to dedicated pipelines without changing this contract.
            plan = self._planner.plan(question, has_memory=bool(memory.turns))
            logger.info(
                "qa.planned",
                intent=plan.intent.value,
                confidence=plan.confidence,
                source=plan.source,
                rationale=plan.rationale,
            )
            if plan.intent == QueryIntent.INSUFFICIENT_CONTEXT:
                record_refusal("planner")
                return _planner_refusal(sid, plan)

            org_label = (
                str(tenant.organization_id) if tenant and tenant.organization_id else None
            )
            scope_label = "tenant" if org_label else "anonymous"
            record_tenant_request(org_label, "/ask")

            # Conversational short-circuit — greetings, thanks, "who are
            # you", etc. skip retrieval and the answer generator
            # entirely. Returns instantly with a templated reply.
            if plan.intent in CONVERSATIONAL_INTENTS:
                conv = self._conversational.respond(question, plan.intent)
                self._memory.append_assistant(
                    sid,
                    conv.answer,
                    metadata={
                        "confidence": conv.confidence,
                        "confidence_score": 1.0,
                        "refused": False,
                        "citations": [],
                        "conversational": True,
                    },
                )
                total_ms = int((time.perf_counter() - t0) * 1000)
                logger.info(
                    "qa.answered.conversational",
                    intent=plan.intent.value,
                    total_latency_ms=total_ms,
                )
                return AskResponse(
                    answer=conv.answer,
                    citations=[],
                    confidence=conv.confidence,
                    sources_used=[],
                    reasoning_summary=plan.rationale,
                    session_id=str(sid),
                    structured_answer=conv.structured,
                    query_intent=plan.intent.value,
                    query_intent_confidence=plan.confidence,
                )

            with (
                traced(
                    "qa.retrieve",
                    intent=plan.intent.value,
                    scope=scope_label,
                ),
                observe(retrieval_latency, scope=scope_label),
            ):
                cross = self._reasoner.gather(question, tenant=tenant)
            passages = cross.passages

            # Source distribution gauge — useful for spotting if tenants
            # accidentally start retrieving only their private corpus.
            global_n = sum(1 for p in passages if p.visibility_scope == "global")
            update_source_ratio(global_n, len(passages) - global_n)

            with traced(
                "qa.generate",
                intent=plan.intent.value,
                scope=scope_label,
                passage_count=len(passages),
            ):
                result: GeneratedAnswer = self._generator.generate(
                    question, passages, memory=memory.turns
                )

            # Outcome metrics.
            if result.refused:
                record_refusal("generator")
            record_ask(
                scope=scope_label,
                intent=plan.intent.value,
                confidence=result.confidence,
            )

            citations: list[Citation] = (
                [] if result.refused else format_citations(result.used_passages)
            )
            sources = (
                [] if result.refused else format_sources(result.used_passages)
            )

            self._memory.append_assistant(
                sid,
                result.answer,
                metadata={
                    "confidence": result.confidence,
                    "confidence_score": result.confidence_score,
                    "refused": result.refused,
                    "citations": [c.model_dump() for c in citations],
                },
            )

            total_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "qa.answered",
                confidence=result.confidence,
                refused=result.refused,
                retrieved=len(passages),
                bucket_counts=cross.bucket_counts,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                gen_latency_ms=result.latency_ms,
                total_latency_ms=total_ms,
            )

            # Phase A — per-request trace covering the full pipeline.
            # Logged at INFO so it's visible without flipping log levels;
            # field names match the "Retrieval diagnostics" list in the
            # Phase A spec so anyone debugging a "why did it refuse"
            # incident can grep one log line and see every stage.
            logger.info(
                "qa.trace",
                original_query=question[:512],
                query_variants=cross.query_variants,
                intent=plan.intent.value,
                intent_confidence=plan.confidence,
                retrieved_count=len(passages),
                top_retrieval_scores=[
                    round(p.original_score, 3) if p.original_score is not None else None
                    for p in passages[:5]
                ],
                top_rerank_scores=[
                    round(p.rerank_score, 3) if p.rerank_score is not None else None
                    for p in passages[:5]
                ],
                final_confidence=result.confidence,
                final_confidence_score=round(result.confidence_score, 3),
                refused=result.refused,
                citation_count=len(citations),
            )

            return AskResponse(
                answer=result.answer,
                citations=citations,
                confidence=result.confidence,
                sources_used=sources,
                reasoning_summary=result.reasoning_summary,
                session_id=str(sid),
                structured_answer=result.structured,
                query_intent=plan.intent.value,
                query_intent_confidence=plan.confidence,
            )

    def ask_stream(
        self,
        question: str,
        session_id: str | None = None,
        *,
        tenant: TenantContext | None = None,
    ) -> Iterator[SSEEvent]:
        """SSE event stream for the same pipeline as :meth:`ask`.

        Implementation note: we deliberately reuse :meth:`ask` rather than
        re-implementing the pipeline. The terminal ``completed`` /
        ``refused`` event carries the full AskResponse. Intermediate events
        announce milestones (retrieval started, sources found, reasoning)
        so the client can render progress immediately.

        Calling this method does NOT issue a second LLM call.
        """
        sid_uuid = self._memory.get_or_create_session(session_id)
        sid = str(sid_uuid)

        yield SSEEvent(
            event="retrieval_started",
            data={"session_id": sid, "question_chars": len(question)},
        )

        response = self.ask(question, session_id=session_id, tenant=tenant)

        # We can't fan out events mid-pipeline without a second LLM call,
        # but we CAN emit the structured milestones derived from the
        # finished response. These let the UI render the answer in
        # sections rather than as one block.
        yield SSEEvent(
            event="sources_found",
            data={
                "count": len(response.sources_used),
                "scopes": _scope_distribution(response.sources_used),
            },
        )

        if response.structured_answer is None or response.confidence == "low":
            # Refusal terminal event — preserves the refusal contract.
            yield SSEEvent(
                event="refused",
                data={
                    "answer": response.answer,
                    "confidence": response.confidence,
                    "structured_answer": response.structured_answer.model_dump()
                    if response.structured_answer
                    else None,
                    "session_id": response.session_id,
                    "query_intent": response.query_intent,
                },
            )
            return

        sa = response.structured_answer
        yield SSEEvent(event="reasoning", data={"summary": response.reasoning_summary})
        yield SSEEvent(
            event="direct_answer",
            data={"direct_answer": sa.direct_answer},
        )
        yield SSEEvent(
            event="citations",
            data={
                "applicable_law": [c.model_dump() for c in sa.applicable_law],
                "relevant_public_cases": [c.model_dump() for c in sa.relevant_public_cases],
                "firm_similar_cases": [c.model_dump() for c in sa.firm_similar_cases],
                "citation_ids": sa.citations,
            },
        )
        yield SSEEvent(
            event="completed",
            data=response.model_dump(mode="json"),
        )

    def get_or_create_session(
        self,
        session_id: str | None,
        *,
        tenant: TenantContext | None = None,
    ) -> uuid.UUID:
        return self._memory.get_or_create_session(session_id, tenant=tenant)


def _scope_distribution(sources: list) -> dict[str, int]:
    counts: dict[str, int] = {"global": 0, "tenant_private": 0}
    for s in sources:
        scope = getattr(s, "source_scope", "global") or "global"
        counts[scope] = counts.get(scope, 0) + 1
    return counts


def _planner_refusal(session_id: uuid.UUID, plan: QueryPlan) -> AskResponse:
    """Short-circuit response when the planner says the question is too vague.

    Mirrors the canonical refusal shape from AnswerGenerator so clients see
    a consistent contract regardless of who refused.
    """
    structured = StructuredAnswer(
        direct_answer=REFUSAL_TEXT,
        legal_reasoning=(
            "Your question references something without prior context "
            "(e.g. 'this case', 'that law'). Re-ask with the specific "
            "statute, case, or fact pattern."
        ),
        confidence_assessment="Refused before retrieval: ambiguous reference.",
    )
    return AskResponse(
        answer=f"{REFUSAL_TEXT}\n\n{DISCLAIMER}",
        citations=[],
        confidence="low",
        sources_used=[],
        reasoning_summary=plan.rationale,
        session_id=str(session_id),
        structured_answer=structured,
        query_intent=plan.intent.value,
        query_intent_confidence=plan.confidence,
    )
