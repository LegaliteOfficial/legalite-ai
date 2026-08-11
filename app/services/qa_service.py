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
from app.rag.legal_memory import LegalMemory, MemoryContext
from app.rag.multi_document_reasoner import CrossDocResult, MultiDocumentReasoner
from app.rag.prompt_builder import DISCLAIMER, REFUSAL_TEXT
from app.rag.query_planner import (
    CONVERSATIONAL_INTENTS,
    QueryIntent,
    QueryPlan,
    QueryPlanner,
)
from app.rag.query_rewriter import QueryRewriter
from app.rag.retriever import RetrievedPassage
from app.rag.streaming import SSEEvent
from app.rag.reranker import BaseReranker, build_reranker
from app.schemas.qa import AskResponse, Citation, StructuredAnswer
from app.services.corpus_inventory import CorpusInventoryService
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
        # Module-level cache lives behind this; constructing it per
        # request is cheap because the actual Postgres roundtrip only
        # fires on the first call after the cache TTL expires.
        self._corpus_inventory = CorpusInventoryService(db)

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
            prep = self._prepare(question, sid, tenant=tenant, t0=t0)
            if isinstance(prep, AskResponse):
                return prep
            plan, scope_label, memory, passages, cross = prep

            with traced(
                "qa.generate",
                intent=plan.intent.value,
                scope=scope_label,
                passage_count=len(passages),
            ):
                result: GeneratedAnswer = self._generator.generate(
                    question, passages, memory=memory.turns
                )

            return self._finalize(sid, question, plan, scope_label, passages, cross, result, t0)

    def ask_stream(
        self,
        question: str,
        session_id: str | None = None,
        *,
        tenant: TenantContext | None = None,
    ) -> Iterator[SSEEvent]:
        """SSE event stream that runs the real pipeline live rather than
        replaying a finished :meth:`ask` response.

        Once retrieval hands off to the LLM, ``answer_delta`` events carry
        ``direct_answer`` text as OpenAI generates it — see
        ``AnswerGenerator.generate_stream``. Those deltas are provisional:
        the same post-generation safety checks that gate the non-streaming
        path (grounding, refusal-sentence detection) still run on the
        *complete* answer once the model finishes. If they reject it, a
        terminal ``refused`` event fires with the canonical refusal text,
        which may differ from what the deltas already showed — the client
        MUST replace whatever it rendered from the deltas with the
        refusal event's payload, not merge the two.

        Once the first byte of an SSE response has been sent, FastAPI can
        no longer turn an exception into a proper 500 — the connection
        would just drop, leaving the client with a truncated stream and
        no explanation. Everything after the ``retrieval_started`` yield
        below therefore runs inside a try/except that turns any failure
        into a terminal ``error`` event instead.
        """
        t0 = time.perf_counter()
        sid = self._memory.get_or_create_session(session_id, tenant=tenant)
        with structlog.contextvars.bound_contextvars(
            session_id=str(sid),
            tenant_org=str(tenant.organization_id) if tenant and tenant.organization_id else None,
        ):
            yield SSEEvent(
                event="retrieval_started",
                data={"session_id": str(sid), "question_chars": len(question)},
            )

            try:
                prep = self._prepare(question, sid, tenant=tenant, t0=t0)
                if isinstance(prep, AskResponse):
                    # Planner refusal or conversational short-circuit — no
                    # LLM call happens on this path, so there is nothing
                    # to stream.
                    response = prep
                    yield SSEEvent(
                        event="sources_found",
                        data={"count": 0, "scopes": {"global": 0, "tenant_private": 0}},
                    )
                    yield SSEEvent(event="completed", data=response.model_dump(mode="json"))
                    return
                plan, scope_label, memory, passages, cross = prep

                yield SSEEvent(
                    event="sources_found",
                    data={
                        "count": len(passages),
                        "scopes": _passage_scope_distribution(passages),
                    },
                )

                result: GeneratedAnswer | None = None
                with traced(
                    "qa.generate",
                    intent=plan.intent.value,
                    scope=scope_label,
                    passage_count=len(passages),
                ):
                    for item in self._generator.generate_stream(
                        question, passages, memory=memory.turns
                    ):
                        if isinstance(item, str):
                            yield SSEEvent(event="answer_delta", data={"text": item})
                        else:
                            result = item
                assert result is not None  # generate_stream always yields exactly one GeneratedAnswer

                response = self._finalize(
                    sid, question, plan, scope_label, passages, cross, result, t0
                )

                if result.refused:
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
                            # Refusals still persist a chat_messages row
                            # (see _finalize), so feedback (thumbs
                            # up/down) works on refused answers the same
                            # as on normal ones.
                            "message_id": response.message_id,
                        },
                    )
                    return

                sa = response.structured_answer
                assert sa is not None  # non-refused _finalize_from_raw always sets this
                yield SSEEvent(event="reasoning", data={"summary": response.reasoning_summary})
                yield SSEEvent(
                    event="citations",
                    data={
                        "applicable_law": [c.model_dump() for c in sa.applicable_law],
                        "relevant_public_cases": [
                            c.model_dump() for c in sa.relevant_public_cases
                        ],
                        "firm_similar_cases": [
                            c.model_dump() for c in sa.firm_similar_cases
                        ],
                        "citation_ids": sa.citations,
                    },
                )
                yield SSEEvent(
                    event="completed",
                    data=response.model_dump(mode="json"),
                )
            except Exception:
                logger.exception("qa.ask_stream_failed", question_chars=len(question))
                yield SSEEvent(
                    event="error",
                    data={
                        "message": (
                            "The AI service hit an unexpected error while answering. "
                            "Please try again."
                        ),
                    },
                )

    def get_or_create_session(
        self,
        session_id: str | None,
        *,
        tenant: TenantContext | None = None,
    ) -> uuid.UUID:
        return self._memory.get_or_create_session(session_id, tenant=tenant)

    def _prepare(
        self,
        question: str,
        sid: uuid.UUID,
        *,
        tenant: TenantContext | None,
        t0: float,
    ) -> (
        AskResponse
        | tuple[QueryPlan, str, MemoryContext, list[RetrievedPassage], CrossDocResult]
    ):
        """Session/memory setup, intent planning, and retrieval — the part
        of the pipeline that is identical whether the answer ends up
        streamed or not. Returns an early :class:`AskResponse` for the
        planner-refusal and conversational short-circuit paths (neither
        calls the answer LLM, so there's nothing for a caller to stream);
        otherwise returns ``(plan, scope_label, memory, passages, cross)``
        for the caller to hand to the generator.
        """
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

        org_label = str(tenant.organization_id) if tenant and tenant.organization_id else None
        scope_label = "tenant" if org_label else "anonymous"
        record_tenant_request(org_label, "/ask")

        # Conversational short-circuit — greetings, thanks, "who are
        # you", etc. skip retrieval and the answer generator entirely.
        # Returns instantly with a templated reply.
        #
        # CONVERSATIONAL_UNCLEAR additionally consumes the corpus
        # inventory so the response includes a capability tour + seed
        # questions ("here's what I can actually answer"). Inventory hit
        # is cached at module level, so warm-path calls don't touch
        # Postgres.
        if plan.intent in CONVERSATIONAL_INTENTS:
            inventory = None
            if plan.intent == QueryIntent.CONVERSATIONAL_UNCLEAR:
                try:
                    inventory = self._corpus_inventory.get()
                except Exception:  # noqa: BLE001
                    # Inventory failures must NEVER block a reply. The
                    # responder degrades to its empty-corpus fallback if
                    # ``inventory`` is None.
                    logger.exception("qa.inventory_failed")
            conv = self._conversational.respond(question, plan.intent, inventory=inventory)
            msg_id = self._memory.append_assistant(
                sid,
                conv.answer,
                metadata={
                    "confidence": conv.confidence,
                    "confidence_score": 1.0,
                    "refused": False,
                    "citations": [],
                    "conversational": True,
                    "query_intent": plan.intent.value,
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
                message_id=str(msg_id),
            )

        with (
            traced("qa.retrieve", intent=plan.intent.value, scope=scope_label),
            observe(retrieval_latency, scope=scope_label),
        ):
            cross = self._reasoner.gather(question, tenant=tenant)
        passages = cross.passages

        # Source distribution gauge — useful for spotting if tenants
        # accidentally start retrieving only their private corpus.
        global_n = sum(1 for p in passages if p.visibility_scope == "global")
        update_source_ratio(global_n, len(passages) - global_n)

        return plan, scope_label, memory, passages, cross

    def _finalize(
        self,
        sid: uuid.UUID,
        question: str,
        plan: QueryPlan,
        scope_label: str,
        passages: list[RetrievedPassage],
        cross: CrossDocResult,
        result: GeneratedAnswer,
        t0: float,
    ) -> AskResponse:
        """Citation formatting, memory persistence, and response assembly —
        the part of the pipeline after generation that is identical
        whether ``result`` came from :meth:`AnswerGenerator.generate` or
        the terminal item of :meth:`AnswerGenerator.generate_stream`.
        """
        if result.refused:
            record_refusal("generator")
        record_ask(scope=scope_label, intent=plan.intent.value, confidence=result.confidence)

        citations: list[Citation] = (
            [] if result.refused else format_citations(result.used_passages)
        )
        sources = [] if result.refused else format_sources(result.used_passages)

        msg_id = self._memory.append_assistant(
            sid,
            result.answer,
            metadata={
                "confidence": result.confidence,
                "confidence_score": result.confidence_score,
                "refused": result.refused,
                "citations": [c.model_dump() for c in citations],
                "query_intent": plan.intent.value,
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

        # Phase A — per-request trace covering the full pipeline. Logged
        # at INFO so it's visible without flipping log levels; field
        # names match the "Retrieval diagnostics" list in the Phase A
        # spec so anyone debugging a "why did it refuse" incident can
        # grep one log line and see every stage.
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
            message_id=str(msg_id),
        )


def _passage_scope_distribution(passages: list[RetrievedPassage]) -> dict[str, int]:
    """Scope breakdown straight off retrieval.

    Uses ``visibility_scope`` (set by the retriever) rather than
    ``source_scope`` (set later by ``app.rag.authority.annotate`` inside
    the generator) because this runs before generation — before
    ``annotate`` has had a chance to run. Both carry the same
    "global" | "tenant_private" values.
    """
    counts: dict[str, int] = {"global": 0, "tenant_private": 0}
    for p in passages:
        scope = p.visibility_scope or "global"
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
