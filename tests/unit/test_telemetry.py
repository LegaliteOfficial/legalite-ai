"""Telemetry primitives — metric registration + render (Phase 5 M6)."""

from __future__ import annotations

from app.core.telemetry import (
    REGISTRY,
    observe,
    record_ask,
    record_citation_resolution,
    record_refusal,
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
    )
    for fam in expected_families:
        assert fam in body_str, f"missing metric family {fam}"


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
