"""Application configuration.

All settings come from environment variables (or a `.env` file in dev). We
validate them eagerly at startup via Pydantic so misconfiguration fails loud
instead of surfacing as a runtime error in the middle of a request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

AppEnv = Literal["development", "staging", "production"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application -------------------------------------------------------
    app_name: str = "legalite-ai"
    app_env: AppEnv = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    debug: bool = False

    # --- OpenAI ------------------------------------------------------------
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536

    # --- RAG tuning --------------------------------------------------------
    rag_top_k: int = 6
    rag_chunk_size: int = 900
    rag_chunk_overlap: int = 120
    rag_min_confidence_score: float = 0.35

    # Hybrid retrieval + rerank
    hybrid_topk_vector: int = 20
    hybrid_topk_keyword: int = 20
    hybrid_topk_fused: int = 20
    rag_topk_final: int = 5
    rrf_k: int = 60

    # Reranker
    rerank_provider: Literal["cohere", "none"] = "cohere"
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-english-v3.0"
    rerank_timeout_seconds: int = 8

    # Conversational memory
    memory_window_turns: int = 6
    memory_max_tokens: int = 2000

    # --- Qdrant ------------------------------------------------------------
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "ghana_legal_docs"

    # --- PostgreSQL --------------------------------------------------------
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "legalite"
    postgres_password: str = "legalite_dev_password"
    postgres_db: str = "legalite"
    database_url: str | None = None  # explicit override wins

    # --- Storage -----------------------------------------------------------
    upload_dir: str = "/app/data/uploads"
    max_upload_mb: int = 50

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return _force_psycopg3(self.database_url)
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


def _force_psycopg3(url: str) -> str:
    """Ensure the SQLAlchemy URL uses the psycopg (v3) driver, since psycopg2 isn't installed.

    Accepts `postgresql://...` or `postgres://...` and rewrites both to
    `postgresql+psycopg://...`. Leaves explicit driver suffixes alone.
    """
    if url.startswith("postgresql+"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    return url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings. Use this everywhere — never instantiate Settings directly."""
    return Settings()  # type: ignore[call-arg]
