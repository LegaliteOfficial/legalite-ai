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

    # Character limits for LegalChunker's oversized-unit fallback split
    # (app/ingestion/legal_chunker.py). Structure-aware chunking (splitting
    # on detected article/section/clause markers) does NOT use these at
    # all — they only bound a single hierarchy unit that's too large to
    # keep whole. Previously named RAG_CHUNK_SIZE=900 / RAG_CHUNK_OVERLAP=120
    # and silently multiplied by 4 inside LegalChunker.__init__ (an
    # undocumented token->char heuristic) to get these same effective
    # values (900*4=3600, 120*4=480) — renamed so the number you set here
    # is the number actually enforced, no hidden multiplier.
    rag_max_chunk_chars: int = 3600
    rag_chunk_overlap_chars: int = 480

    # Confidence / refusal thresholds — previously a single
    # RAG_MIN_CONFIDENCE_SCORE that was read into ConfidenceEngine and then
    # never actually used; the real gates were hardcoded literals scattered
    # through app/rag/confidence_engine.py. These five names are exactly
    # those literals, given names and defaults equal to the values already
    # live in production — this is a wiring fix, not a threshold change.
    # Do not retune these without the golden-set evaluation harness
    # (evaluation/runner.py) to measure the effect first.
    rag_catastrophic_cosine_threshold: float = 0.10
    rag_catastrophic_rerank_threshold: float = 0.05
    rag_confidence_high_threshold: float = 0.5
    rag_confidence_medium_threshold: float = 0.3
    rag_citation_consistency_high_threshold: float = 0.6

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

    # Case-law intelligence
    case_llm_enrichment_enabled: bool | None = None  # None = auto-detect (off in tests)
    # Upload-time doc_type classification (bulk uploads without a doc_type)
    doc_classifier_llm_enabled: bool | None = None  # None = auto-detect (off in tests)
    case_similarity_top_k: int = 8
    case_similarity_topic_weight: float = 0.35
    case_similarity_semantic_weight: float = 0.65
    # Citation resolution — minimum confidence to auto-link an edge.
    citation_resolution_threshold: float = 0.6
    citation_resolution_chain_depth: int = 4

    # Query planner
    query_planner_llm_fallback_enabled: bool = False
    query_planner_min_heuristic_confidence: float = 0.5

    # Query understanding (Phase A). When enabled, every retrieval-bound
    # question is rewritten into 2–3 semantic variants before search,
    # broadening recall on conversationally-phrased queries. Adds one LLM
    # call (~300-500ms) per cache miss; identical questions are cached
    # in-process. Disable to revert to single-query retrieval.
    query_rewriter_enabled: bool = True
    query_rewriter_max_variants: int = 3

    # Ask Legalite chat agent. When enabled, /ask runs a tool-calling
    # conversational agent (app/rag/chat_agent.py) instead of the strict
    # one-shot RAG generator: the model answers any question — legal or
    # not — and calls the hybrid retriever as a tool whenever law is
    # involved, citing retrieved passages inline as [n].
    # None = auto-detect: on iff a real-looking OpenAI key is configured
    # (so unit tests, which use the "test-key" sentinel, keep the legacy
    # pipeline unless they opt in explicitly).
    ask_agent_enabled: bool | None = None
    # Model for the agent. Falls back to OPENAI_LLM_MODEL when unset. The
    # agent is the user-facing "brain" — use the strongest model the
    # budget allows (e.g. gpt-4.1 / gpt-4o / a reasoning model).
    agent_llm_model: str | None = None
    # Optional cheaper model for purely conversational turns (greetings,
    # thanks, small talk) the planner is confident about. None = same
    # model as agent_llm_model.
    agent_fast_llm_model: str | None = None
    agent_temperature: float = 0.3
    agent_max_output_tokens: int = 2000
    # Upper bound on tool-calling rounds per turn (each round = one LLM
    # call). Bounds latency + cost on runaway tool loops.
    agent_max_tool_rounds: int = 4
    # How many passages a single search_legal_library call returns, and
    # how much of each passage's text is shown to the model.
    agent_search_top_k: int = 6
    agent_passage_max_chars: int = 1800
    # Conversation memory fed to the agent as real chat turns.
    agent_memory_window_turns: int = 12
    agent_memory_max_tokens: int = 6000
    agent_memory_turn_max_chars: int = 2500
    # Optional on-screen context (e.g. the open document) sent alongside
    # a question. Hard cap so one request can't blow the context window.
    ask_context_max_chars: int = 20000

    # Observability
    prometheus_enabled: bool = True
    otel_enabled: bool = False
    otel_service_name: str = "legalite-ai"
    otel_exporter_otlp_endpoint: str | None = None

    # Background ingestion
    redis_url: str = "redis://redis:6379/0"
    ingestion_queue_name: str = "legalite_ingest"
    ingestion_max_retries: int = 3
    ingestion_job_timeout_seconds: int = 1800

    # Bulk upload — how many PDFs a single ``POST /upload-law/bulk``
    # request may carry. Tuned for:
    #   - HTTP body size (25 × 50 MB = ~1.25 GB upper bound; well
    #     under most reverse-proxy/CDN limits)
    #   - Per-request commit budget (one INSERT into ingestion_jobs
    #     per file; sequential commits are ~10ms each)
    #   - Worker queue throughput — 25 enqueues fan out cleanly to
    #     the existing Arq worker without backpressure
    # Increase cautiously: the multipart parser holds every file in
    # memory before the handler runs.
    bulk_upload_max_files: int = 25

    # Cloud object storage (Supabase Storage). When ``supabase_url`` and
    # ``supabase_service_key`` are BOTH set, the ingestion path uploads
    # each PDF to ``supabase_storage_bucket`` and the /documents/{id}
    # endpoint returns a short-lived signed URL alongside the extracted
    # text. When either is unset the system degrades gracefully —
    # ingestion still succeeds, ``remote_storage_key`` stays NULL, and
    # the endpoint returns text-only.
    supabase_url: str | None = None
    supabase_service_key: str | None = None
    supabase_storage_bucket: str = "legal-documents"
    # Signed-URL TTLs. Global docs are public-record law; the URL can
    # safely live long enough to be cacheable. Tenant-private docs MUST
    # use a short TTL — if a URL leaks, the blast radius is bounded.
    storage_signed_url_ttl_global_seconds: int = 86400  # 24 hours
    storage_signed_url_ttl_private_seconds: int = 300   # 5 minutes

    # Service-to-service authentication (legalite-backend → legalite-ai).
    # When set, the gateway must sign tenant headers with HMAC-SHA256 and
    # include x-legalite-signature + x-legalite-timestamp. When None
    # (default), the trust-the-network model from Phase 4 applies — useful
    # for local dev, NOT acceptable in production.
    legalite_gateway_secret: str | None = None
    gateway_signature_max_skew_seconds: int = 300

    # Rate limiting (slowapi).
    rate_limit_enabled: bool = True
    rate_limit_ask: str = "30/minute"
    rate_limit_upload: str = "10/minute"
    rate_limit_default: str = "120/minute"

    # OCR fallback for scanned PDFs (audit Item 5).
    # When ``None`` (default), the parser tries to import pytesseract on first
    # use and silently disables OCR if the import fails. Explicit ``true``
    # raises on failure (use this in prod). Explicit ``false`` disables OCR
    # even when pytesseract is installed (useful in tests).
    ocr_enabled: bool | None = None
    # Below this many characters extracted from a page we treat it as scanned
    # and try OCR. Empirically, ~40 chars is a header line on a real PDF page.
    ocr_min_text_chars: int = 40
    # Tesseract DPI for rendering — higher = better OCR but slower.
    ocr_render_dpi: int = 200
    ocr_lang: str = "eng"
    ocr_timeout_seconds: int = 60

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

    @property
    def agent_enabled(self) -> bool:
        """Resolve ``ask_agent_enabled`` (None = auto-detect from the key)."""
        if self.ask_agent_enabled is not None:
            return self.ask_agent_enabled
        return bool(self.openai_api_key) and self.openai_api_key != "test-key"


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
