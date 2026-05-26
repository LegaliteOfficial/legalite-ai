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

import structlog
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.core.tenant_context import TenantContext
from app.db.repositories.chat_repository import ChatRepository
from app.ingestion.embedding_service import EmbeddingService
from app.rag.answer_generator import AnswerGenerator, GeneratedAnswer
from app.rag.citation_formatter import format_citations, format_sources
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.legal_memory import LegalMemory
from app.rag.multi_document_reasoner import MultiDocumentReasoner
from app.rag.reranker import BaseReranker, build_reranker
from app.schemas.qa import AskResponse, Citation
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
    ) -> None:
        from app.vector_store.qdrant_client import get_vector_store

        embeddings = embeddings or EmbeddingService()
        vector_store = vector_store or get_vector_store()

        self._retriever = HybridRetriever(
            db=db, embeddings=embeddings, vector_store=vector_store
        )
        self._reranker = reranker or build_reranker()
        self._reasoner = MultiDocumentReasoner(
            retriever=self._retriever, reranker=self._reranker
        )
        self._generator = generator or AnswerGenerator()
        self._memory = LegalMemory(ChatRepository(db))

    def ask(
        self,
        question: str,
        session_id: str | None = None,
        *,
        tenant: TenantContext | None = None,
    ) -> AskResponse:
        t0 = time.perf_counter()
        sid = self._memory.get_or_create_session(session_id)
        with structlog.contextvars.bound_contextvars(
            session_id=str(sid),
            tenant_org=str(tenant.organization_id) if tenant and tenant.organization_id else None,
        ):
            memory = self._memory.load_recent(sid)
            self._memory.append_user(sid, question)

            cross = self._reasoner.gather(question, tenant=tenant)
            passages = cross.passages

            result: GeneratedAnswer = self._generator.generate(
                question, passages, memory=memory.turns
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

            return AskResponse(
                answer=result.answer,
                citations=citations,
                confidence=result.confidence,
                sources_used=sources,
                reasoning_summary=result.reasoning_summary,
                session_id=str(sid),
                structured_answer=result.structured,
            )

    def get_or_create_session(self, session_id: str | None) -> uuid.UUID:
        return self._memory.get_or_create_session(session_id)
