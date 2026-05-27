# syntax=docker/dockerfile:1.7
# =============================================================================
# Legalite AI - Production Image
# Multi-stage build: small final image, no build toolchain at runtime.
# =============================================================================

# ---- Stage 1: builder ----
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System deps needed only to *build* wheels (psycopg, pymupdf, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Generous timeout + retry budget — llama-index pulls heavy transitive deps
# (nltk, tiktoken, sentencepiece, etc.) and the 15-second pip default chokes
# on slow / flaky links. BuildKit cache mount keeps wheels around across
# rebuilds so a transient network blip costs minutes, not a full re-download.
RUN --mount=type=cache,id=legalite-pip,target=/root/.cache/pip \
    pip install --prefix=/install \
        --timeout 600 \
        --retries 5 \
        -r requirements.txt


# ---- Stage 2: runtime ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_HOME=/app

# Runtime-only system deps. tesseract-ocr is the OCR engine used by the
# pytesseract Python wrapper for scanned-PDF fallback (Phase 5.1 / audit Item 5).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system --gid app --home ${APP_HOME} app

WORKDIR ${APP_HOME}

# Copy installed Python deps from builder
COPY --from=builder /install /usr/local

# Copy app source
COPY --chown=app:app . .

RUN mkdir -p /app/data/uploads && chown -R app:app /app/data
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
