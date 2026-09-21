"""Production telemetry — Prometheus metrics + OpenTelemetry hooks.

Two surfaces in one module:

- **Prometheus metrics** (always enabled). Counters / histograms covering
  retrieval, reranking, LLM latency, refusal rate, tenant request counts,
  Qdrant / Postgres failures, upload outcomes, and the
  ``tenant_isolation_score`` produced by the eval harness.

- **OpenTelemetry** (opt-in via ``OTEL_ENABLED=true``). Provides a
  no-op-safe ``traced()`` context manager that records a span when an
  OTel SDK is wired up at process start, otherwise becomes a cheap noop.
  Spans accept tenant + request-level attributes so traces stay scoped.

Importing this module is safe even when the optional ``opentelemetry``
packages are absent — we fall back to a stub Tracer.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prometheus registry & metrics
# ---------------------------------------------------------------------------

# Dedicated registry so tests can read it without contaminating the default.
REGISTRY = CollectorRegistry()


# RAG retrieval / generation
retrieval_latency = Histogram(
    "legalite_retrieval_latency_seconds",
    "End-to-end retrieval latency (vector + keyword + fusion).",
    labelnames=("scope",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)
rerank_latency = Histogram(
    "legalite_rerank_latency_seconds",
    "Cohere reranker call duration. Zero observations when reranker disabled "
    "(NoOpReranker never calls record_rerank_latency).",
    labelnames=("status",),  # success | failure
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)
embedding_latency = Histogram(
    "legalite_embedding_latency_seconds",
    "OpenAI embedding call duration (per batch attempt).",
    labelnames=("status",),  # success | failure
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)
llm_latency = Histogram(
    "legalite_llm_latency_seconds",
    "Chat-completion latency for /ask answer generation.",
    buckets=(0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0),
    registry=REGISTRY,
)

# Outcome counters
refusal_total = Counter(
    "legalite_refusal_total",
    "Refusals split by reason category.",
    labelnames=("reason",),  # weak_retrieval | parse_failed | llm_error | planner | model_refused
    registry=REGISTRY,
)
ask_total = Counter(
    "legalite_ask_total",
    "Total /ask requests handled.",
    labelnames=("scope", "intent", "confidence"),
    registry=REGISTRY,
)
hallucination_floor = Counter(
    "legalite_hallucination_floor_total",
    "Heuristic count of answers with no [n] citation markers despite passages "
    "being available — should always be 0 if the refusal contract holds.",
    registry=REGISTRY,
)

# Self-learning loop (Tier 1) — every feedback POST emits one increment.
# Labelling by (thumbs × query_intent × confidence × refused) lets the
# dashboard answer: "what fraction of HIGH-confidence STATUTE_QUESTIONs
# got thumbed down?", "are refused answers more likely to get downvotes
# than non-refused?", "which intent is the loudest gap?". Cardinality
# bound: 2 × ~13 intents × 3 confidences × 2 = ~156 series. Within Prom
# safety limits.
feedback_total = Counter(
    "legalite_feedback_total",
    "User feedback (👍/👎) on assistant answers — the input to the "
    "self-improvement loop.",
    labelnames=("thumbs", "query_intent", "confidence", "refused"),
    registry=REGISTRY,
)

# Tenancy / isolation
tenant_requests = Counter(
    "legalite_tenant_requests_total",
    "Requests per organization.",
    labelnames=("organization_id", "endpoint"),
    registry=REGISTRY,
)
global_vs_private_ratio = Gauge(
    "legalite_global_vs_private_ratio",
    "Fraction of retrieved chunks that were GLOBAL across the last sample. "
    "1.0 means tenants are getting pure public law; 0.0 means pure private.",
    registry=REGISTRY,
)
tenant_isolation_score = Gauge(
    "legalite_tenant_isolation_score",
    "Tenant-isolation eval score in [0, 1]. CI fails on < 1.0.",
    registry=REGISTRY,
)

# Infrastructure
qdrant_failures = Counter(
    "legalite_qdrant_failures_total",
    "Exceptions raised by the Qdrant client.",
    labelnames=("operation",),  # search | upsert | delete | ensure_collection
    registry=REGISTRY,
)
postgres_failures = Counter(
    "legalite_postgres_failures_total",
    "Exceptions raised by SQLAlchemy operations we instrument.",
    labelnames=("operation",),
    registry=REGISTRY,
)

# Upload / ingestion
upload_total = Counter(
    "legalite_uploads_total",
    "Upload requests grouped by outcome.",
    labelnames=("scope", "outcome"),  # outcome: ingested | failed
    registry=REGISTRY,
)
ingestion_job_state = Counter(
    "legalite_ingestion_job_state_total",
    "Background ingestion job state transitions.",
    labelnames=("state",),  # queued | processing | completed | failed | retried
    registry=REGISTRY,
)

# Citation graph
citation_resolution_outcome = Counter(
    "legalite_citation_resolution_total",
    "Edges processed by the citation resolver.",
    labelnames=("outcome",),  # resolved | unresolved
    registry=REGISTRY,
)

# OCR fallback (Phase 5.1 / audit Item 5)
ocr_pages_total = Counter(
    "legalite_ocr_pages_total",
    "Pages processed by the OCR fallback during PDF ingestion.",
    labelnames=("outcome",),  # text_extracted | ocr_extracted | ocr_failed | ocr_disabled
    registry=REGISTRY,
)
ocr_latency = Histogram(
    "legalite_ocr_latency_seconds",
    "Tesseract OCR call duration (per page).",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _TimerCtx:
    histogram: Histogram
    start: float


@contextlib.contextmanager
def observe(histogram: Histogram, **labels: str) -> Iterator[_TimerCtx]:
    """Time a block and record the duration on a histogram.

    Usage::

        with observe(retrieval_latency, scope="global"):
            ...

    Labels on the histogram must match; missing labels raise immediately
    so misuse fails loudly in tests rather than silently dropping data.
    """
    metric = histogram.labels(**labels) if labels else histogram
    ctx = _TimerCtx(histogram=histogram, start=time.perf_counter())
    try:
        yield ctx
    finally:
        metric.observe(time.perf_counter() - ctx.start)


def record_refusal(reason: str) -> None:
    refusal_total.labels(reason=reason).inc()


def record_ask(scope: str, intent: str, confidence: str) -> None:
    ask_total.labels(scope=scope, intent=intent, confidence=confidence).inc()


def record_tenant_request(organization_id: str | None, endpoint: str) -> None:
    org_label = organization_id or "anonymous"
    tenant_requests.labels(organization_id=org_label, endpoint=endpoint).inc()


def record_upload(scope: str, outcome: str) -> None:
    upload_total.labels(scope=scope, outcome=outcome).inc()


def record_citation_resolution(resolved: bool) -> None:
    citation_resolution_outcome.labels(
        outcome="resolved" if resolved else "unresolved"
    ).inc()


def record_ocr_page(outcome: str) -> None:
    """Track per-page OCR outcome — one of: text_extracted, ocr_extracted,
    ocr_failed, ocr_disabled."""
    ocr_pages_total.labels(outcome=outcome).inc()


def record_qdrant_failure(operation: str) -> None:
    """One shared call site so every Qdrant operation reports the same way.

    Before this, only ``search()`` incremented ``qdrant_failures`` — a
    failing ``upsert``/``delete``/``ensure_collection`` raised with no
    telemetry at all. Call this from an ``except`` block and re-raise;
    it never swallows the exception itself.
    """
    qdrant_failures.labels(operation=operation).inc()


def record_postgres_failure(operation: str) -> None:
    """Mirror of :func:`record_qdrant_failure` for instrumented SQLAlchemy
    calls on the RAG path. Call from an ``except`` block and re-raise."""
    postgres_failures.labels(operation=operation).inc()


def record_embedding_latency(seconds: float, *, success: bool) -> None:
    embedding_latency.labels(status="success" if success else "failure").observe(seconds)


def record_rerank_latency(seconds: float, *, success: bool) -> None:
    rerank_latency.labels(status="success" if success else "failure").observe(seconds)


def record_hallucination_floor_if_ungrounded(*, confidence: str, has_citations: bool) -> None:
    """Same heuristic as ``evaluation.metrics.hallucination_rate``, applied
    per live request instead of per eval case: an answer that wasn't
    refused (confidence != "low") but produced zero citations is a case
    we *know* is ungrounded — a floor, not a real hallucination detector.
    A genuine hallucination (wrong claim backed by a real-looking but
    incorrect citation) is invisible to this check by design; that needs
    the golden-set comparison in the eval harness, not a production metric.
    """
    if confidence != "low" and not has_citations:
        hallucination_floor.inc()


def update_isolation_score(value: float) -> None:
    """Called by the tenant-isolation eval harness to publish its score."""
    tenant_isolation_score.set(max(0.0, min(1.0, value)))


def update_source_ratio(global_count: int, private_count: int) -> None:
    total = global_count + private_count
    if total == 0:
        return
    global_vs_private_ratio.set(global_count / total)


def render_metrics() -> tuple[bytes, str]:
    """Render the Prometheus exposition format. Returns ``(body, content_type)``."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


# ---------------------------------------------------------------------------
# OpenTelemetry — optional, noop-safe
# ---------------------------------------------------------------------------


class _NoopSpan:
    def set_attribute(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover
        return

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class _NoopTracer:
    """Used when OTel is disabled or the SDK isn't installed."""

    @contextlib.contextmanager
    def start_as_current_span(
        self, _name: str, **_kwargs: object
    ) -> Iterator[_NoopSpan]:
        yield _NoopSpan()


_tracer_instance: object | None = None


def get_tracer() -> object:
    """Lazily initialize an OTel tracer if configured, else a noop tracer.

    Initialization failures (missing OTLP endpoint, missing SDK) are
    swallowed and degrade to noop so observability bugs never break /ask.
    """
    global _tracer_instance
    if _tracer_instance is not None:
        return _tracer_instance

    settings = get_settings()
    if not settings.otel_enabled:
        _tracer_instance = _NoopTracer()
        return _tracer_instance

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": settings.otel_service_name})
        provider = TracerProvider(resource=resource)

        if settings.otel_exporter_otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        _tracer_instance = trace.get_tracer("legalite.rag")
    except Exception:  # noqa: BLE001
        logger.exception("otel.init_failed")
        _tracer_instance = _NoopTracer()
    return _tracer_instance


@contextlib.contextmanager
def traced(name: str, **attributes: object) -> Iterator[object]:
    """Record an OTel span when enabled, else a noop. Always safe to call.

    ``attributes`` are set on the span when present. Use sparingly — high
    cardinality (e.g. raw question text) belongs in structured logs, not
    trace attributes.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:  # type: ignore[attr-defined]
        for k, v in attributes.items():
            try:
                span.set_attribute(k, v)
            except Exception:  # pragma: no cover
                pass
        yield span
