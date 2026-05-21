# Architecture notes

## Why this shape

- **Thin routes, thick services.** Routes only handle HTTP concerns
  (validation, status codes). All RAG / ingestion logic lives in
  `app/services/*`, which is independently importable from scripts or
  workers.
- **Repositories isolate SQLAlchemy.** Anything that talks to Postgres goes
  through `app/db/repositories/`. This keeps the rest of the codebase free
  of ORM imports and makes the persistence layer swappable.
- **Vector store wrapper.** `QdrantVectorStore` is the only module that
  imports `qdrant_client`. Retriever and services depend on the wrapper, so
  we can later support a second backend (pgvector, Weaviate, ...) without
  touching call sites.
- **Embedding + LLM split.** `EmbeddingService` and `AnswerGenerator` are
  separate classes. Embeddings are deterministic and cacheable; chat
  completions are not. Splitting them lets each scale and retry differently.
- **Strict prompt + client-side refusal.** The system prompt forbids the LLM
  from speculating. On top of that, `AnswerGenerator` refuses outright when
  the top retrieval score is below `RAG_MIN_CONFIDENCE_SCORE` — so the LLM
  is never invoked on weak evidence.

## Chunking strategy

Legal documents have a deep hierarchy. Generic recursive splitters break
citations across boundaries (e.g. cutting an article in half). The
`LegalChunker`:

1. Scans lines for hierarchy markers in this priority order:
   `Title → Chapter → Article → Section → Clause → Subsection`.
2. Emits one chunk per leaf unit, carrying the full hierarchy path so
   citations are always complete.
3. Falls back to size-based splitting (with overlap) for unusually long
   units, so a single huge article still fits the context window.
4. Preserves page numbers per chunk for accurate citation back to the PDF.

## Confidence signal

`AnswerGenerator._score_to_confidence` maps the top Qdrant score:

- `< RAG_MIN_CONFIDENCE_SCORE` → **low** → refusal
- `>= 0.60` → **high**
- otherwise → **medium**

These thresholds are starting points and should be tuned against a curated
eval set once one exists.
