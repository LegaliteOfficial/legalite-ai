"""FastAPI application entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    admin,
    archive,
    cases,
    document_view,
    documents,
    drafting,
    feedback,
    health,
    jobs,
    metrics,
    private_documents,
    qa,
    sessions,
)
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.observability import ObservabilityMiddleware
from app.core.rate_limit import configure_rate_limiting
from app.core.service_auth import signature_required


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    logger = get_logger(__name__)
    settings = get_settings()
    try:
        from app.vector_store.qdrant_client import get_vector_store

        get_vector_store()
    except Exception:  # noqa: BLE001
        logger.exception("startup.qdrant_unavailable")

    # One-time, unmissable log of whether reranking is actually active.
    # build_reranker() already warns on every /ask request when
    # COHERE_API_KEY is missing (app/rag/reranker.py) — that's easy to miss
    # in request-volume logs, so this makes the same fact visible exactly
    # once at boot instead.
    from app.rag.reranker import CohereReranker, build_reranker

    reranker = build_reranker()
    logger.info(
        "startup.reranker_status",
        configured_provider=settings.rerank_provider,
        active_implementation=type(reranker).__name__,
        enabled=isinstance(reranker, CohereReranker),
    )

    logger.info(
        "app.starting",
        env=settings.app_env,
        debug=settings.debug,
        llm_model=settings.openai_llm_model,
        embed_model=settings.openai_embedding_model,
    )
    yield
    logger.info("app.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    logger = get_logger(__name__)
    app = FastAPI(
        title="Legalite AI",
        description=(
            "AI-powered Ghana Legal Assistant. Answers are grounded ONLY in "
            "uploaded Ghana legal materials. Informational use only — not legal advice."
        ),
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # CORS — tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Observability: request-id + structured per-request logging
    app.add_middleware(ObservabilityMiddleware)

    # Rate limiting (audit Item 6). Per-tenant via x-organization-id, falling
    # back to client IP for anonymous traffic.
    configure_rate_limiting(app)

    if not signature_required():
        logger.warning(
            "service_auth.unsigned_mode_enabled",
            note=(
                "LEGALITE_GATEWAY_SECRET is not set — gateway headers are "
                "trusted at face value. Only safe behind a private network."
            ),
        )

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(document_view.router)
    app.include_router(qa.router)
    app.include_router(drafting.router)
    app.include_router(feedback.router)
    app.include_router(sessions.router)
    app.include_router(cases.router)
    app.include_router(admin.router)
    app.include_router(private_documents.router)
    app.include_router(archive.router)
    app.include_router(jobs.router)
    app.include_router(metrics.router)
    return app


app = create_app()
