# Legalite AI — Ghana Legal Assistant

AI-powered retrieval system for Ghana legal documents (Constitution, Acts,
Regulations, Case Law). Answers user questions using **only** the legal
materials uploaded into the system, with citations to law, article, section,
and page. Refuses to answer when evidence is insufficient.

> **Not legal advice.** Outputs are informational only.

---

## Architecture

```
┌────────────────── INGESTION ──────────────────┐
│ POST /upload-law                              │
│   PDF → PyMuPDF → Chunker v2 (Title/Chapter/  │
│   Part/Article/Section/Clause/Subsection +    │
│   law_type, act_number, year, topic)          │
│   → OpenAI embeddings → Qdrant                │
│   + Postgres (FTS tsvector + metadata)        │
└───────────────────────────────────────────────┘

┌────────────────── Q&A (v2) ───────────────────┐
│ POST /ask                                     │
│   memory.load (Postgres-backed)               │
│   → MultiDocumentReasoner                     │
│       └─ per law_type (constitution, act,     │
│          regulation, case_law):               │
│             HybridRetriever                   │
│               ├─ Qdrant vector search         │
│               └─ Postgres FTS (tsvector+GIN)  │
│             merge via Reciprocal Rank Fusion  │
│       └─ Reranker (Cohere / no-op fallback)   │
│   → ConfidenceEngine (pre-LLM gate)           │
│   → grounded prompt → LLM                     │
│   → ConfidenceEngine (post-LLM grounding)     │
│   → CitationFormatter (dedup + rank)          │
│   → memory.append                             │
│   → AskResponse (answer, citations,           │
│     sources_used, confidence, reasoning)      │
└───────────────────────────────────────────────┘
```

### Tech stack

| Layer            | Choice                                                      |
| ---------------- | ----------------------------------------------------------- |
| API              | FastAPI 0.115, Pydantic v2, Uvicorn                          |
| RAG framework    | LlamaIndex 0.12, OpenAI SDK                                 |
| Embeddings       | OpenAI `text-embedding-3-small` (1536-d)                    |
| LLM              | OpenAI `gpt-4o-mini` (configurable)                         |
| Vector DB        | Qdrant (Cosine distance)                                    |
| Relational DB    | PostgreSQL 16 + SQLAlchemy 2 + Alembic                      |
| PDF extraction   | PyMuPDF (page-aware)                                        |
| Logging          | structlog (JSON in prod, console in dev)                    |
| Tooling          | Ruff, Black, MyPy (strict), Pytest, Pre-commit              |
| Runtime          | Python 3.12, Docker + Docker Compose                        |

---

## Project layout

```
project-root/
│
├── app/
│   ├── api/
│   │   ├── routes/              # health.py, documents.py, qa.py
│   │   └── dependencies/        # FastAPI Depends() providers
│   │
│   ├── core/
│   │   ├── config.py            # Pydantic Settings (env-driven)
│   │   ├── logging.py           # structlog setup
│   │   └── security.py          # API-key dependency stub
│   │
│   ├── ingestion/
│   │   ├── pdf_parser.py        # PyMuPDF → per-page text
│   │   ├── legal_chunker.py     # Title/Chapter/Article/Section/Clause/Subsection
│   │   └── embedding_service.py # OpenAI embeddings (batched + retry)
│   │
│   ├── rag/
│   │   ├── retriever.py         # Embed query → Qdrant search
│   │   ├── prompt_builder.py    # Strict grounded prompt
│   │   ├── citation_formatter.py
│   │   └── answer_generator.py  # OpenAI chat completion + refusal
│   │
│   ├── vector_store/
│   │   └── qdrant_client.py     # Collection mgmt, upsert, search
│   │
│   ├── db/
│   │   ├── models/              # LegalDocument, LegalChunk
│   │   ├── repositories/        # DocumentRepository
│   │   ├── session.py
│   │   ├── base.py
│   │   └── migrations/          # Alembic env + 0001_initial.py
│   │
│   ├── schemas/                 # Pydantic API DTOs
│   ├── services/                # IngestionService, QAService
│   └── main.py                  # FastAPI app factory
│
├── docker/Dockerfile
├── docker-compose.yml
├── alembic.ini
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .pre-commit-config.yaml
├── tests/                       # unit + integration
└── scripts/bootstrap.sh
```

---

## Migrating to v2

If you already had v1 running, apply the new schema and rebuild the image
(new dependency: `cohere`):

```bash
docker compose down
docker compose up --build       # alembic upgrade head runs automatically
```

The 0002 migration adds:
- New columns on `legal_chunks`: `part`, `law_type`, `act_number`,
  `effective_year`, `legal_topic`, and a generated `content_tsv` (tsvector)
  with a GIN index for Postgres full-text search.
- `chat_sessions` + `chat_messages` for conversational memory.

Existing documents you've already ingested will keep working — the new
columns are nullable and the FTS index is built over existing content
automatically. To re-tag old chunks with v2 metadata, re-upload them.

## Quick start (Docker — recommended)

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, QDRANT_URL/API_KEY, POSTGRES_PASSWORD

# 2. Bring up the stack (app + postgres + qdrant)
docker compose up --build

# 3. The API is live
curl http://localhost:8000/health
open http://localhost:8000/docs
```

Migrations run automatically on container start (`alembic upgrade head`).

> **Never commit `.env`** — it contains live API keys. Only `.env.example`
> belongs in version control.

## Quick start (local Python)

```bash
./scripts/bootstrap.sh
docker compose up -d postgres qdrant
alembic upgrade head
uvicorn app.main:app --reload
```

---

## API

### `GET /health`
```json
{ "status": "ok", "version": "0.1.0" }
```

### `POST /upload-law` (multipart/form-data)
Fields:
- `file` — PDF
- `title` — e.g. `"Constitution of Ghana"`
- `doc_type` — `Constitution | Act | Regulation | Case Law | ...`

Response (201):
```json
{
  "document_id": "…uuid…",
  "title": "Constitution of Ghana",
  "doc_type": "Constitution",
  "page_count": 0,
  "chunks_indexed": 312,
  "status": "ingested"
}
```

### `POST /ask` (v2 — backward compatible)
Request:
```json
{
  "question": "Can a landlord evict a tenant without notice?",
  "session_id": "optional-uuid-for-conversational-memory"
}
```
Response (v1 fields + v2 additions):
```json
{
  "answer": "Direct answer ...\n\nLegal explanation [1][2] ...\n\nCitations:\n- [1] Rent Act, s.17\n\nConfidence note: ...\n\nThis information is for educational purposes and not legal advice.",
  "citations": [
    {
      "law": "Rent Act, 1963 (Act 220)",          // legacy
      "article": "",                                // legacy
      "section": "17",                              // legacy
      "page": "12",                                 // legacy
      "law_name": "Rent Act, 1963 (Act 220)",
      "law_type": "act",
      "act_number": "220",
      "chapter": "",
      "part": "",
      "clause": "",
      "subsection": "",
      "page_number": 12,
      "relevance": 0.91
    }
  ],
  "confidence": "high",
  "disclaimer": "This information is for educational purposes and not legal advice.",
  "sources_used": [
    {
      "law_name": "Rent Act, 1963 (Act 220)",
      "law_type": "act",
      "section": "17",
      "article": "",
      "page": "12",
      "excerpt": "Where a tenant has been in arrears...",
      "score": 0.91
    }
  ],
  "reasoning_summary": "final=0.78 consistency=1.00 grounded=True",
  "session_id": "9f2c..."
}
```

When the retrieved evidence is too weak, the system responds with:
> "I could not confidently determine this from the provided Ghana legal materials.
>
> This information is for educational purposes and not legal advice."

### `POST /sessions` and `GET /sessions/{id}`

Manage conversational memory. Pass the returned `session_id` to subsequent
`/ask` calls so follow-up questions ("what if I was pregnant?") inherit
context.

---

## Safety contract

The system prompt in [`app/rag/prompt_builder.py`](app/rag/prompt_builder.py)
enforces:

1. Answer **only** from retrieved passages.
2. Refuse with the exact sentence when evidence is insufficient.
3. Never invent statutes, articles, sections, or page numbers.
4. Always cite passages inline (`[n]`).
5. Be conservative and neutral; never give legal advice.
6. Always append the disclaimer.

A secondary check in [`AnswerGenerator`](app/rag/answer_generator.py) refuses
client-side when the top retrieval score is below `RAG_MIN_CONFIDENCE_SCORE`,
so the LLM is never invoked on weak grounding.

---

## Development

```bash
# Lint, format, type-check
ruff check . && black --check . && mypy app

# Tests
pytest -q

# New migration
alembic revision --autogenerate -m "add new column"
alembic upgrade head
```

Pre-commit hooks run all formatters/linters on every commit:
```bash
pre-commit install
```

---

## Configuration reference

All settings come from environment variables. See [`.env.example`](.env.example)
for the full list. The most important:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required. |
| `OPENAI_LLM_MODEL` | Default `gpt-4o-mini`. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` (1536-d). |
| `QDRANT_URL` / `QDRANT_API_KEY` | Local Compose or Qdrant Cloud. |
| `QDRANT_COLLECTION` | Defaults to `ghana_legal_docs`. |
| `RAG_TOP_K` | Passages retrieved per question. |
| `RAG_MIN_CONFIDENCE_SCORE` | Below this → automatic refusal. |
| `DATABASE_URL` | Optional explicit URL; otherwise built from `POSTGRES_*`. |

---

## Intelligence upgrades (v2)

This codebase ships with the following production-grade RAG features beyond
naive vector search:

| Feature | Module | What it does |
| --- | --- | --- |
| Hybrid retrieval | [app/rag/hybrid_retriever.py](app/rag/hybrid_retriever.py) | Qdrant (semantic) + Postgres FTS (exact legal terms), merged via Reciprocal Rank Fusion |
| Reranking | [app/rag/reranker.py](app/rag/reranker.py) | Cohere Rerank with no-op fallback when key is unset |
| Legal-aware chunking | [app/ingestion/legal_chunker.py](app/ingestion/legal_chunker.py) | Title/Chapter/Part/Article/Section/Clause/Subsection + law_type, act_number, year, topic |
| Citation engine | [app/rag/citation_formatter.py](app/rag/citation_formatter.py) | Dedup by structural identity, rank by relevance, expose `sources_used` with excerpts |
| Confidence scoring | [app/rag/confidence_engine.py](app/rag/confidence_engine.py) | Pre-LLM gate (retrieval/rerank thresholds) + post-LLM grounding & consistency check; forces refusal on weak evidence |
| Conversational memory | [app/rag/legal_memory.py](app/rag/legal_memory.py) | Postgres-backed chat sessions, windowed + token-capped |
| Cross-document reasoning | [app/rag/multi_document_reasoner.py](app/rag/multi_document_reasoner.py) | Per-law_type buckets so Constitution / Acts / Regulations / Case law all get a seat at the table |
| Prompt v2 | [app/rag/prompt_builder.py](app/rag/prompt_builder.py) | Structured 5-part answer, explicit conflict surfacing, refusal contract |
| Observability | [app/core/observability.py](app/core/observability.py) | request_id middleware + per-stage latency/tokens logs |
| Evaluation harness | [evaluation/](evaluation/) | recall@k, citation accuracy, refusal correctness, hallucination floor |

### Running the evaluation

```bash
python -m evaluation.runner --dataset evaluation/datasets/sample.json
```

The sample dataset includes one "must refuse" case for sanity-checking the
refusal contract. Replace with a curated Ghana-law set as you build it.

## Next implementation steps

1. **Auth & multi-tenancy** — wire `verify_api_key` into protected routes,
   add user/org tables, partition Qdrant payloads by tenant.
2. **Background ingestion** — move `IngestionService` behind a queue (Celery
   or Arq) so large PDFs don't block the request thread.
3. **Hybrid retrieval** — combine dense embeddings with BM25 over Postgres
   `tsvector` for higher recall on rare legal terms.
4. **Re-ranking** — add a cross-encoder reranker (e.g. Cohere Rerank) before
   building the prompt to push the most relevant passages to the top.
5. **Evaluation harness** — curate a Ghana-law eval set; score answer
   faithfulness, citation accuracy, and refusal rate against it.
6. **Citation linking** — store offsets so the UI can highlight the exact
   span in the source PDF, not just the page number.
7. **Streaming answers** — switch `/ask` to SSE for snappier UX.
8. **Observability** — OpenTelemetry traces around retrieval, prompt
   construction, and LLM calls; Prometheus metrics for hit rates and refusals.
# legalite-ai
