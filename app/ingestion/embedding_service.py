"""OpenAI embedding service with batching and retry.

We batch because the embeddings endpoint accepts up to 2048 inputs per call,
and a single legal Act can produce thousands of chunks. Retries handle the
common transient failures (rate limits, brief network blips).
"""

from __future__ import annotations

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_BATCH_SIZE = 96  # comfortable under OpenAI limits while keeping latency low


class EmbeddingService:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model
        self._expected_dim = settings.openai_embedding_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            out.extend(self._embed_batch(batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=batch)
        vectors = [item.embedding for item in response.data]
        if vectors and len(vectors[0]) != self._expected_dim:
            raise RuntimeError(
                f"Embedding dim mismatch: model returned {len(vectors[0])}, "
                f"settings expect {self._expected_dim}"
            )
        return vectors
