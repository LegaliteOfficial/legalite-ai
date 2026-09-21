"""Telemetry primitives — metric registration + render (Phase 5 M6)."""

from __future__ import annotations

from app.core.telemetry import (
    hallucination_floor,
    observe,
    postgres_failures,
    record_ask,
    record_citation_resolution,
    record_embedding_latency,
    record_hallucination_floor_if_ungrounded,
    record_postgres_failure,
    record_qdrant_failure,
    record_refusal,
    record_rerank_latency,
    record_upload,
    render_metrics,
    retrieval_latency,
    update_isolation_score,
    update_source_ratio,
)


def test_render_metrics_contains_known_families() -> None:
    body, content_type = render_metrics()
    body_str = body.decode("utf-8")
    assert "text/plain" in content_type or "openmetrics" in content_type
    # Every histogram + counter from telemetry.py must appear in the output
    # (the # HELP / # TYPE preamble is emitted for every registered metric).
    expected_families = (
        "legalite_retrieval_latency_seconds",
        "legalite_llm_latency_seconds",
        "legalite_refusal_total",
        "legalite_ask_total",
        "legalite_tenant_requests_total",
        "legalite_qdrant_failures_total",
        "legalite_uploads_total",
        "legalite_tenant_isolation_score",
        "legalite_citation_resolution_total",
        # Previously dead (zero call sites) — Phase 1 wired these up.
        # Listed explicitly so a future regression that stops calling
        # record_embedding_latency/record_rerank_latency/etc. still shows
        # the family (histograms/counters render even at zero
        # observations), but the increment tests below would catch an
        # actual call-site removal.
        "legalite_embedding_latency_seconds",
        "legalite_rerank_latency_seconds",
        "legalite_postgres_failures_total",
        "legalite_hallucination_floor_total",
    )
    for fam in expected_families:
        assert fam in body_str, f"missing metric family {fam}"


def test_record_embedding_latency_increments_by_status() -> None:
    record_embedding_latency(0.1, success=True)
    record_embedding_latency(0.2, success=False)
    body = render_metrics()[0].decode("utf-8")
    assert 'legalite_embedding_latency_seconds_count{status="success"}' in body
    assert 'legalite_embedding_latency_seconds_count{status="failure"}' in body


def test_record_rerank_latency_increments_by_status() -> None:
    record_rerank_latency(0.05, success=True)
    body = render_metrics()[0].decode("utf-8")
    assert 'legalite_rerank_latency_seconds_count{status="success"}' in body


def test_record_qdrant_and_postgres_failures() -> None:
    record_qdrant_failure("upsert")
    record_qdrant_failure("ensure_collection")
    record_postgres_failure("keyword_search")
    body = render_metrics()[0].decode("utf-8")
    assert 'legalite_qdrant_failures_total{operation="upsert"}' in body
    assert 'legalite_qdrant_failures_total{operation="ensure_collection"}' in body
    assert 'legalite_postgres_failures_total{operation="keyword_search"}' in body


def test_hallucination_floor_only_fires_when_ungrounded() -> None:
    before = postgres_failures.labels(operation="keyword_search")._value.get()  # type: ignore[attr-defined]
    hallucination_before = hallucination_floor._value.get()  # type: ignore[attr-defined]

    # Grounded (has citations) — must NOT increment regardless of confidence.
    record_hallucination_floor_if_ungrounded(confidence="high", has_citations=True)
    # Refused/low-confidence with no citations — expected shape, not a hallucination.
    record_hallucination_floor_if_ungrounded(confidence="low", has_citations=False)
    assert hallucination_floor._value.get() == hallucination_before  # type: ignore[attr-defined]

    # Confident answer with zero citations — exactly the ungrounded case this floor exists to catch.
    record_hallucination_floor_if_ungrounded(confidence="high", has_citations=False)
    assert hallucination_floor._value.get() == hallucination_before + 1  # type: ignore[attr-defined]
    # Unrelated counter must be untouched by the above.
    assert postgres_failures.labels(operation="keyword_search")._value.get() == before  # type: ignore[attr-defined]


def test_record_ask_and_refusal_increment_counters() -> None:
    record_ask(scope="tenant", intent="statute_question", confidence="high")
    record_refusal(reason="planner")
    body = render_metrics()[0].decode("utf-8")
    assert 'legalite_ask_total{' in body
    assert 'legalite_refusal_total{reason="planner"}' in body


def test_observe_records_histogram() -> None:
    before = retrieval_latency.labels(scope="anonymous")._sum.get()  # type: ignore[attr-defined]
    with observe(retrieval_latency, scope="anonymous"):
        pass
    after = retrieval_latency.labels(scope="anonymous")._sum.get()  # type: ignore[attr-defined]
    assert after >= before


def test_update_isolation_score_clamps_to_unit_interval() -> None:
    update_isolation_score(1.5)
    update_isolation_score(-0.2)
    update_isolation_score(0.85)
    body = render_metrics()[0].decode("utf-8")
    assert "legalite_tenant_isolation_score 0.85" in body


def test_record_upload_and_citation_outcomes() -> None:
    record_upload(scope="global", outcome="ingested")
    record_citation_resolution(resolved=True)
    record_citation_resolution(resolved=False)
    body = render_metrics()[0].decode("utf-8")
    assert 'legalite_uploads_total{outcome="ingested",scope="global"}' in body
    assert 'legalite_citation_resolution_total{outcome="resolved"}' in body
    assert 'legalite_citation_resolution_total{outcome="unresolved"}' in body


def test_update_source_ratio_safe_with_zero_total() -> None:
    update_source_ratio(0, 0)  # must not raise / divide-by-zero
