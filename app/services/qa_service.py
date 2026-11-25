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
from collections.abc import Callable, Iterator

import structlog
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.telemetry import (
    observe,
    record_ask,
    record_hallucination_floor_if_ungrounded,
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
from app.rag.authority import classify as classify_authority
from app.rag.chat_agent import (
    AgentResult,
    AgentStatus,
    ChatAgent,
    ChatTurn,
    clean_history_text,
    passage_header,
)
from app.rag.citation_formatter import format_citations, format_sources
from app.rag.conversation_memory import ConversationSummarizer, build_memory_message
from app.rag.conversational_responder import ConversationalResponder
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.legal_memory import LegalMemory, MemoryContext, Summarizer
from app.rag.multi_document_reasoner import CrossDocResult, MultiDocumentReasoner
from app.rag.prompt_builder import DISCLAIMER, REFUSAL_TEXT, MemoryTurn
from app.rag.query_planner import (
    _LEGAL_SUBSTANCE_RE,
    CONVERSATIONAL_INTENTS,
    QueryIntent,
    QueryPlan,
    QueryPlanner,
)
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import BaseReranker, build_reranker
from app.rag.retriever import RetrievedPassage
from app.rag.streaming import SSEEvent
from app.rag.workspace_consent import (
    AccessResolution,
    ConsentDecision,
    ConsentStore,
    InMemoryConsentStore,
    PermissionRequest,
    SessionConsentStore,
    resolve_access,
    token_subject,
)
from app.rag.workspace_tools import WorkspaceToolkit
from app.schemas.qa import (
    AskResponse,
    Citation,
    PermissionRequestOut,
    StructuredAnswer,
    StructuredCitation,
)
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
        agent: ChatAgent | None = None,
        use_agent: bool | None = None,
        workspace_factory: Callable[[str], WorkspaceToolkit] | None = None,
        summarizer: Summarizer | None = None,
        consent_store: ConsentStore | None = None,
    ) -> None:
        from app.vector_store.qdrant_client import get_vector_store

        embeddings = embeddings or EmbeddingService()
        vector_store = vector_store or get_vector_store()

        self._retriever = HybridRetriever(db=db, embeddings=embeddings, vector_store=vector_store)
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
        # Chat agent (Ask Legalite v4). ``use_agent=None`` follows the
        # ASK_AGENT_ENABLED setting (auto-on with a real OpenAI key). An
        # explicitly injected agent always wins — that's how tests opt in.
        self._summarizer: Summarizer | None = summarizer
        if agent is not None:
            self._agent: ChatAgent | None = agent
        elif use_agent if use_agent is not None else get_settings().agent_enabled:
            self._agent = ChatAgent()
            if self._summarizer is None:
                self._summarizer = ConversationSummarizer()
        else:
            self._agent = None
        # Tests inject a factory; production builds a GraphQL-backed toolkit.
        self._workspace_factory = workspace_factory
        self._consent_store = consent_store

    def ask(
        self,
        question: str,
        session_id: str | None = None,
        *,
        tenant: TenantContext | None = None,
        context: str | None = None,
        user_token: str | None = None,
        client_history: list[MemoryTurn] | None = None,
        workspace_consent: ConsentDecision | None = None,
    ) -> AskResponse:
        if self._agent is not None:
            final: AskResponse | None = None
            for item in self._run_agent(
                question,
                session_id,
                tenant=tenant,
                context=context,
                user_token=user_token,
                client_history=client_history,
                workspace_consent=workspace_consent,
            ):
                if isinstance(item, AskResponse):
                    final = item
                elif item.event == "error":
                    raise RuntimeError(item.data.get("message", "agent failed"))
            assert final is not None
            return final

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
        context: str | None = None,
        user_token: str | None = None,
        client_history: list[MemoryTurn] | None = None,
        workspace_consent: ConsentDecision | None = None,
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
        if self._agent is not None:
            for item in self._run_agent(
                question,
                session_id,
                tenant=tenant,
                context=context,
                user_token=user_token,
                client_history=client_history,
                workspace_consent=workspace_consent,
            ):
                if isinstance(item, SSEEvent):
                    yield item
            return

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
                assert (
                    result is not None
                )  # generate_stream always yields exactly one GeneratedAnswer

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
                        "relevant_public_cases": [c.model_dump() for c in sa.relevant_public_cases],
                        "firm_similar_cases": [c.model_dump() for c in sa.firm_similar_cases],
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

    # ------------------------------------------------------------------
    # Chat agent path (Ask Legalite v4)
    # ------------------------------------------------------------------

    def _run_agent(
        self,
        question: str,
        session_id: str | None,
        *,
        tenant: TenantContext | None,
        context: str | None,
        user_token: str | None = None,
        client_history: list[MemoryTurn] | None = None,
        workspace_consent: ConsentDecision | None = None,
    ) -> Iterator[SSEEvent | AskResponse]:
        """Run one chat turn through :class:`ChatAgent`.

        Yields the SSE events the streaming route forwards, and — just
        before the terminal ``completed`` event — the final
        :class:`AskResponse` object itself, so :meth:`ask` can share this
        code path without re-parsing JSON. Event names are a superset of
        the legacy contract (``retrieval_started`` → ``sources_found`` →
        ``answer_delta``* → ``citations`` → ``completed`` | ``error``), plus
        ``status`` events carrying a human-readable progress line.

        Workspace consent: when the user's data is needed and they have not
        allowed it (see ``app/rag/workspace_consent.py``) the turn ends with
        a ``permission_required`` event instead of an answer. Nothing is
        written to memory for a paused turn; the client resends the same
        question with ``workspace_consent`` and the turn runs in full.
        """
        assert self._agent is not None
        settings = get_settings()
        t0 = time.perf_counter()
        question = (question or "").strip()
        if not question:
            raise ValueError("question must not be empty")
        page_context = (context or "").strip()[: settings.ask_context_max_chars] or None

        sid = self._memory.get_or_create_session(session_id, tenant=tenant)
        with structlog.contextvars.bound_contextvars(
            session_id=str(sid),
            tenant_org=str(tenant.organization_id) if tenant and tenant.organization_id else None,
        ):
            yield SSEEvent(
                event="retrieval_started",
                data={"session_id": str(sid), "question_chars": len(question)},
            )
            answered = False
            user_saved = False
            plan: QueryPlan | None = None
            try:
                if client_history:
                    # No-op unless the server has no record of this chat.
                    self._memory.import_history(sid, client_history)
                conv = self._memory.load_conversation(
                    sid,
                    window_turns=settings.agent_memory_window_turns,
                    max_tokens=settings.agent_memory_max_tokens,
                    summarizer=self._summarizer,
                )
                memory_note = build_memory_message(conv)
                history = [
                    ChatTurn(
                        role="assistant" if t.role == "assistant" else "user",
                        content=clean_history_text(t.content, settings.agent_memory_turn_max_chars),
                    )
                    for t in conv.recent
                ]

                plan = self._planner.plan(question, has_memory=bool(history))
                org_label = (
                    str(tenant.organization_id) if tenant and tenant.organization_id else None
                )
                scope_label = "tenant" if org_label else "anonymous"
                record_tenant_request(org_label, "/ask")

                conversational, force_search = route_agent_turn(
                    plan, question, page_context=page_context
                )
                logger.info(
                    "qa.agent.planned",
                    intent=plan.intent.value,
                    confidence=plan.confidence,
                    force_search=force_search,
                    conversational=conversational,
                )

                def _search(query: str) -> list[RetrievedPassage]:
                    with (
                        traced("qa.retrieve", intent=plan.intent.value, scope=scope_label),
                        observe(retrieval_latency, scope=scope_label),
                    ):
                        return self._reasoner.gather(query, tenant=tenant).passages

                toolkit = self._workspace_for(user_token)
                access = self._resolve_workspace_access(
                    sid, question, toolkit, user_token, workspace_consent
                )
                workspace = toolkit if access.access in ("granted", "ask") else None
                profile = (
                    toolkit.profile_context(tools_available=False)
                    if toolkit is not None and workspace is None
                    else None
                )
                if access.access == "granted" and access.decision == "allow":
                    yield SSEEvent(
                        event="status",
                        data={
                            "phase": "workspace",
                            "message": "Workspace access allowed for this request",
                        },
                    )
                result: AgentResult | None = None
                with traced("qa.agent", intent=plan.intent.value, scope=scope_label):
                    for item in self._agent.run_stream(
                        question,
                        history=history,
                        search_fn=_search,
                        inventory_fn=self._corpus_inventory.get,
                        force_search=force_search,
                        conversational=conversational,
                        tenant_label=f"role: {tenant.role.value}" if org_label and tenant else None,
                        page_context=page_context,
                        workspace=workspace,
                        workspace_access=access.access,
                        profile=profile,
                        conversation_memory=memory_note,
                    ):
                        if isinstance(item, AgentResult):
                            result = item
                        elif isinstance(item, str):
                            answered = True
                            yield SSEEvent(event="answer_delta", data={"text": item})
                        else:
                            yield from _status_events(item)
                assert result is not None

                if result.mode == "permission_required":
                    response = self._pause_for_permission(sid, question, plan, result, user_token)
                    assert response.permission_request is not None
                    yield SSEEvent(
                        event="permission_required",
                        data=response.permission_request.model_dump(mode="json"),
                    )
                    yield response
                    yield SSEEvent(event="completed", data=response.model_dump(mode="json"))
                    return

                self._memory.append_user(sid, question)
                user_saved = True
                response = self._finalize_agent(sid, question, plan, scope_label, result, t0)
                response.workspace_access = access.access
                sa = response.structured_answer
                assert sa is not None
                yield SSEEvent(
                    event="citations",
                    data={
                        "applicable_law": [c.model_dump() for c in sa.applicable_law],
                        "relevant_public_cases": [c.model_dump() for c in sa.relevant_public_cases],
                        "firm_similar_cases": [c.model_dump() for c in sa.firm_similar_cases],
                        "citation_ids": sa.citations,
                    },
                )
                yield response
                yield SSEEvent(event="completed", data=response.model_dump(mode="json"))
            except Exception:
                logger.exception("qa.agent_failed", question_chars=len(question))
                if not user_saved:
                    # Keep the user's words in the thread even when the
                    # answer failed, as before the consent gate.
                    try:
                        self._memory.append_user(sid, question)
                        user_saved = True
                    except Exception:  # noqa: BLE001
                        logger.exception("qa.agent_save_user_failed")
                # Graceful degradation: if nothing was streamed yet and the
                # turn was small talk, answer from the deterministic
                # templates rather than showing an error for "hi".
                if not answered and plan is not None and plan.intent in CONVERSATIONAL_INTENTS:
                    try:
                        response = self._conversational_fallback(sid, question, plan)
                        yield response
                        yield SSEEvent(event="completed", data=response.model_dump(mode="json"))
                        return
                    except Exception:  # noqa: BLE001
                        logger.exception("qa.agent_fallback_failed")
                yield SSEEvent(
                    event="error",
                    data={
                        "message": (
                            "The AI service hit an unexpected error while answering. "
                            "Please try again."
                        ),
                    },
                )

    def _finalize_agent(
        self,
        sid: uuid.UUID,
        question: str,
        plan: QueryPlan,
        scope_label: str,
        result: AgentResult,
        t0: float,
    ) -> AskResponse:
        cited = result.cited_passages
        citations = format_citations(cited)
        sources = format_sources(cited)
        structured = _agent_structured_answer(result)
        record_ask(scope=scope_label, intent=plan.intent.value, confidence=result.confidence)
        searches = [
            str(tc.arguments.get("query", ""))
            for tc in result.tool_calls
            if tc.name == "search_legal_library"
        ]

        msg_id = self._memory.append_assistant(
            sid,
            result.answer,
            metadata={
                "confidence": result.confidence,
                "confidence_score": result.confidence_score,
                "refused": False,
                "citations": [c.model_dump() for c in citations],
                "query_intent": plan.intent.value,
                "answer_mode": result.mode,
                "agent": True,
                "model": result.model,
                "searches": searches,
                "workspace_tools": [
                    tc.name
                    for tc in result.tool_calls
                    if tc.name not in ("search_legal_library", "describe_legal_library")
                ],
                "proposed_actions": [a.to_dict() for a in result.proposed_actions],
            },
        )
        total_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "qa.agent.answered",
            intent=plan.intent.value,
            mode=result.mode,
            confidence=result.confidence,
            model=result.model,
            rounds=result.rounds,
            searches=searches,
            passages=len(result.passages),
            cited=result.cited_ids,
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
            reasoning_summary="",
            session_id=str(sid),
            structured_answer=structured,
            query_intent=plan.intent.value,
            query_intent_confidence=plan.confidence,
            message_id=str(msg_id),
            answer_mode=result.mode,
            searches=searches,
            model=result.model,
            proposed_actions=[a.to_dict() for a in result.proposed_actions],
        )

    # -- workspace consent ----------------------------------------------

    def _consent(self) -> ConsentStore:
        if self._consent_store is None:
            if hasattr(self._memory, "get_meta") and hasattr(self._memory, "set_meta"):
                self._consent_store = SessionConsentStore(self._memory)
            else:
                self._consent_store = InMemoryConsentStore()
        return self._consent_store

    def _resolve_workspace_access(
        self,
        sid: uuid.UUID,
        question: str,
        toolkit: WorkspaceToolkit | None,
        user_token: str | None,
        decision: ConsentDecision | None,
    ) -> AccessResolution:
        if toolkit is None:
            return resolve_access(
                workspace_available=False, preference="ask", decision=None, redeemed=False
            )
        preference = toolkit.access_preference()
        redeemed = False
        if decision is not None:
            redeemed = self._consent().redeem(
                sid, decision.request_id, subject=token_subject(user_token), question=question
            )
        res = resolve_access(
            workspace_available=True,
            preference=preference,
            decision=decision,
            redeemed=redeemed,
        )
        logger.info(
            "qa.workspace_access",
            access=res.access,
            preference=res.preference,
            decision=res.decision,
            notes=res.notes or None,
        )
        return res

    def _pause_for_permission(
        self,
        sid: uuid.UUID,
        question: str,
        plan: QueryPlan,
        result: AgentResult,
        user_token: str | None,
    ) -> AskResponse:
        request_id, expires_at = self._consent().issue(
            sid, subject=token_subject(user_token), question=question
        )
        req = PermissionRequest(
            request_id=request_id,
            scopes=result.permission_scopes,
            tools=result.permission_tools,
            expires_at=expires_at,
        )
        logger.info("qa.workspace_permission_requested", tools=req.tools)
        return AskResponse(
            answer=result.answer,
            citations=[],
            confidence="high",
            session_id=str(sid),
            structured_answer=StructuredAnswer(direct_answer=result.answer),
            query_intent=plan.intent.value,
            query_intent_confidence=plan.confidence,
            answer_mode="permission_required",
            model=result.model,
            permission_request=PermissionRequestOut(**req.to_dict()),
            workspace_access="ask",
        )

    def _workspace_for(self, user_token: str | None) -> WorkspaceToolkit | None:
        """Bind workspace tools to this request's user, or ``None`` when the
        caller is anonymous or the integration is not configured."""
        settings = get_settings()
        if (
            not user_token
            or not settings.agent_workspace_tools_enabled
            or not settings.legalite_backend_graphql_url
        ):
            return None
        if self._workspace_factory is not None:
            return self._workspace_factory(user_token)
        from app.integrations.workspace_client import WorkspaceClient

        return WorkspaceToolkit(WorkspaceClient(bearer_token=user_token))

    def _conversational_fallback(
        self, sid: uuid.UUID, question: str, plan: QueryPlan
    ) -> AskResponse:
        conv = self._conversational.respond(question, plan.intent, inventory=None)
        msg_id = self._memory.append_assistant(
            sid,
            conv.answer,
            metadata={"conversational": True, "query_intent": plan.intent.value, "fallback": True},
        )
        return AskResponse(
            answer=conv.answer,
            citations=[],
            confidence=conv.confidence,
            session_id=str(sid),
            structured_answer=conv.structured,
            query_intent=plan.intent.value,
            query_intent_confidence=plan.confidence,
            message_id=str(msg_id),
            answer_mode="conversational",
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
    ) -> AskResponse | tuple[QueryPlan, str, MemoryContext, list[RetrievedPassage], CrossDocResult]:
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

        citations: list[Citation] = [] if result.refused else format_citations(result.used_passages)
        sources = [] if result.refused else format_sources(result.used_passages)
        record_hallucination_floor_if_ungrounded(
            confidence=result.confidence, has_citations=bool(citations)
        )

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


# ---------------------------------------------------------------------------
# Chat agent helpers
# ---------------------------------------------------------------------------

# Intents for which the first agent round is forced to search the library
# (when the question also carries legal vocabulary). Grounding for clearly
# legal questions should never depend on the model choosing to search.
_FORCE_SEARCH_INTENTS: frozenset[QueryIntent] = frozenset(
    {
        QueryIntent.STATUTE_QUESTION,
        QueryIntent.CASE_SIMILARITY,
        QueryIntent.CASE_COMPARISON,
        QueryIntent.CASE_SUMMARY,
        QueryIntent.PROCEDURAL_QUESTION,
        QueryIntent.TENANT_CONTEXTUAL,
    }
)

_CASE_TIERS = frozenset({"public_case_law"})


def route_agent_turn(
    plan: QueryPlan, question: str, *, page_context: str | None
) -> tuple[bool, bool]:
    """Decide (conversational, force_search) for one agent turn.

    ``conversational`` → fast model, no forced search. ``force_search`` →
    the first round must call the library search. Kept pure so the live
    evaluation harness routes turns exactly as production does.
    """
    conversational = (
        plan.intent in CONVERSATIONAL_INTENTS
        and plan.intent != QueryIntent.CONVERSATIONAL_UNCLEAR
        and page_context is None
    )
    force_search = (
        plan.intent in _FORCE_SEARCH_INTENTS
        and plan.confidence >= 0.5
        and bool(_LEGAL_SUBSTANCE_RE.search(question))
    )
    return conversational, force_search


def _status_events(status: AgentStatus) -> Iterator[SSEEvent]:
    """Translate agent progress into SSE events.

    ``sources_found`` keeps the legacy shape so older clients still flip
    their "answering" phase; ``status`` is the new human-readable line.
    """
    if status.kind == "searching":
        yield SSEEvent(
            event="status",
            data={
                "phase": "searching",
                "message": f"Searching the Legalite library: “{status.query}”",
                "query": status.query,
            },
        )
    elif status.kind == "sources":
        yield SSEEvent(
            event="sources_found",
            data={
                "count": status.count,
                "scopes": status.scopes or {"global": 0, "tenant_private": 0},
            },
        )
    elif status.kind == "answering":
        yield SSEEvent(event="status", data={"phase": "answering", "message": "Writing the answer"})
    elif status.kind == "workspace":
        yield SSEEvent(event="status", data={"phase": "workspace", "message": status.query})
    elif status.kind == "action" and status.action is not None:
        yield SSEEvent(event="action_proposed", data=status.action.to_dict())


def _agent_structured_answer(result: AgentResult) -> StructuredAnswer:
    """Map the agent's markdown answer + cited passages onto the existing
    7-section ``StructuredAnswer`` so every client keeps working.

    The whole markdown answer is ``direct_answer``; the citation groups are
    derived from the authority tier of each *cited* passage (only passages
    the answer actually references, never everything retrieved).
    """
    law: list[StructuredCitation] = []
    cases: list[StructuredCitation] = []
    firm: list[StructuredCitation] = []
    for n in result.cited_ids:
        p = result.passages[n - 1]
        tier = classify_authority(p)
        excerpt = " ".join(p.content.split())
        cite = StructuredCitation(
            citation_id=n,
            summary=(excerpt[:220] + "…") if len(excerpt) > 220 else excerpt,
            law_name=p.law or passage_header(p),
            section=p.section or "",
            article=p.article or "",
            case_title=(p.law or "") if (tier.tier in _CASE_TIERS or not tier.is_global) else "",
            document_id=p.document_id,
            chunk_id=p.chunk_id,
            page_number=p.page_start,
        )
        if not tier.is_global:
            firm.append(cite)
        elif tier.tier in _CASE_TIERS:
            cases.append(cite)
        else:
            law.append(cite)

    if result.mode == "general":
        note = (
            "Not found in the Legalite library — answered from general knowledge. "
            "Verify against the current law before relying on it."
            if any(tc.name == "search_legal_library" for tc in result.tool_calls)
            else ""
        )
    elif result.mode == "workspace":
        note = "Based on your Legalite workspace."
    elif result.mode == "grounded":
        n = len(result.cited_ids)
        note = f"Grounded in {n} passage{'s' if n != 1 else ''} from the Legalite library."
    else:
        note = ""

    return StructuredAnswer(
        direct_answer=result.answer,
        applicable_law=law,
        relevant_public_cases=cases,
        firm_similar_cases=firm,
        legal_reasoning="",
        confidence_assessment=note,
        citations=list(result.cited_ids),
    )
