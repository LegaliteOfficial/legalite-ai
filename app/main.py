"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, health, qa, sessions
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.observability import ObservabilityMiddleware


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    logger = get_logger(__name__)
    settings = get_settings()

    # Pre-warm Qdrant collection so the first request doesn't pay the cost.
    try:
        from app.vector_store.qdrant_client import get_vector_store

        get_vector_store()
    except Exception:  # noqa: BLE001
        logger.exception("startup.qdrant_unavailable")

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

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(qa.router)
    app.include_router(sessions.router)
    return app


app = create_app()
