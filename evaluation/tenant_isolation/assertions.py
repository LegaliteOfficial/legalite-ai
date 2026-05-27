"""Leakage assertions used by both the runner and the security pytest."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.rag.retriever import RetrievedPassage


@dataclass(slots=True, frozen=True)
class LeakageReport:
    leaked: bool
    leaked_count: int
    details: list[str]


def assert_no_leakage(
    passages: Iterable[RetrievedPassage],
    *,
    requesting_org_id: str | None,
    forbidden_markers: Iterable[str] = (),
) -> LeakageReport:
    """Assert no chunk in ``passages`` belongs to another tenant.

    Two complementary checks:

    1. **Identity** — every passage with ``visibility_scope='tenant_private'``
       must have ``organization_id == requesting_org_id``.
    2. **Content** — none of the forbidden markers (well-known UUID-like
       strings injected into Firm A's docs) may appear in the passage
       content returned to Firm B.

    Both checks must pass for ``leaked = False``.
    """
    details: list[str] = []
    bad = 0
    forbidden = {m for m in forbidden_markers if m}

    for p in passages:
        scope = (p.visibility_scope or "global").lower()
        if scope == "tenant_private":
            if p.organization_id != requesting_org_id:
                bad += 1
                details.append(
                    f"tenant_private chunk org={p.organization_id!r} "
                    f"surfaced for caller {requesting_org_id!r} "
                    f"(law_type={p.law_type}, vector_id={p.vector_id})"
                )
        content = p.content or ""
        for marker in forbidden:
            if marker in content:
                bad += 1
                details.append(f"forbidden marker '{marker}' in retrieved content")

    return LeakageReport(leaked=bad > 0, leaked_count=bad, details=details)


def isolation_score_from_reports(reports: Iterable[LeakageReport]) -> float:
    """Aggregate per-query reports into a single score in [0, 1].

    Reported as ``1 - (leaked_queries / total_queries)``. Any leak → < 1.0,
    which is the CI failure trigger.
    """
    reports = list(reports)
    if not reports:
        return 1.0
    leaked = sum(1 for r in reports if r.leaked)
    return 1.0 - (leaked / len(reports))
