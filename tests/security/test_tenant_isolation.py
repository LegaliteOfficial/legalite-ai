"""Security suite — tenant isolation runner is CI-gated.

CI fails if ``tenant_isolation_score < 1.0``.
"""

from __future__ import annotations

from evaluation.tenant_isolation.runner import run


def test_tenant_isolation_score_is_one() -> None:
    summary = run()
    assert summary["tenant_isolation_score"] == 1.0, summary
    assert summary["leaked_queries"] == 0
    assert summary["firm_a_self_access_markers_visible"] > 0, (
        "positive control failed — Firm A should see its own private docs"
    )


def test_anonymous_filter_shape() -> None:
    summary = run()
    qf = summary["qdrant_filter"]
    # Anonymous filter must have exactly one branch (GLOBAL).
    assert qf["anonymous"]["should_branches"] == 1
    assert qf["anonymous"]["has_org_clause"] is False
    # Tenant filter must have two (GLOBAL + tenant_private).
    assert qf["firm_b"]["should_branches"] == 2
    assert qf["firm_b"]["has_org_clause"] is True
