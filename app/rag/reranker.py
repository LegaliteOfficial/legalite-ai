"""Cross-encoder–style reranking with a graceful no-op fallback.

The retrieval stage (hybrid vector + FTS) prioritises *recall*: get ~20
candidates that mention the topic. The reranker reads the question alongside
each candidate and scores how well the passage actually *answers* it. This
typically gives a large bump in precision@k for legal Q&A, where many
passages mention a term but only a few are responsive.

Providers (configurable via `RERANK_PROVIDER`):
- `cohere`: uses the Cohere Rerank API. Requires `COHERE_API_KEY`.
- `none`:   no-op pass-through. Use this for tests or when the key is unset.

If the configured provider fails (network, timeout, invalid key), we log and
fall back to the original ordering rather than failing the request — the
upstream `confidence_engine` will see a missing `rerank_score` and adjust.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.telemetry import record_rerank_latency
from app.rag.retriever import RetrievedPassage

logger = get_logger(__name__)


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, passages: list[RetrievedPassage], top_n: int
    ) -> list[RetrievedPassage]:
        """Return passages sorted best-first, truncated to `top_n`."""


class NoOpReranker(BaseReranker):
    """Pass-through: returns the first `top_n` passages unchanged."""

    def rerank(
        self, query: str, passages: list[RetrievedPassage], top_n: int
    ) -> list[RetrievedPassage]:
        return passages[:top_n]


class CohereReranker(BaseReranker):
    def __init__(self) -> None:
        # Imported lazily so the dep is only required when actually using Cohere.
        import cohere  # noqa: PLC0415

        settings = get_settings()
        if not settings.cohere_api_key:
            raise RuntimeError("COHERE_API_KEY is required for the cohere reranker")
        self._client = cohere.ClientV2(api_key=settings.cohere_api_key)
        self._model = settings.cohere_rerank_model
        self._timeout = settings.rerank_timeout_seconds

    def rerank(
        self, query: str, passages: list[RetrievedPassage], top_n: int
    ) -> list[RetrievedPassage]:
        if not passages:
            return []
        documents = [p.content for p in passages]
        start = time.perf_counter()
        try:
            response = self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
                # cohere-python v5 accepts request_options for timeout
                request_options={"timeout_in_seconds": self._timeout},
            )
        except Exception:
            record_rerank_latency(time.perf_counter() - start, success=False)
            logger.exception("rerank.cohere_failed", model=self._model, candidates=len(passages))
            return passages[:top_n]
        record_rerank_latency(time.perf_counter() - start, success=True)

        out: list[RetrievedPassage] = []
        for r in response.results:
            idx = r.index
            if 0 <= idx < len(passages):
                p = passages[idx]
                p.rerank_score = float(r.relevance_score)
                out.append(p)
        # Guard against the API returning fewer than expected
        if not out:
            return passages[:top_n]
        return out


def build_reranker() -> BaseReranker:
    """Factory: return the configured reranker, falling back to no-op on misconfig."""
    settings = get_settings()
    provider = settings.rerank_provider
    if provider == "none":
        return NoOpReranker()
    if provider == "cohere":
        if not settings.cohere_api_key:
            logger.warning("rerank.cohere_unconfigured", reason="COHERE_API_KEY missing")
            return NoOpReranker()
        try:
            return CohereReranker()
        except Exception:
            logger.exception("rerank.init_failed", provider=provider)
            return NoOpReranker()
    logger.warning("rerank.unknown_provider", provider=provider)
    return NoOpReranker()
