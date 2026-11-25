"""Field-level diff + topic overlap used by /cases/compare.

We test the pure helpers directly — they're free of DB or HTTP concerns.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.services.case_intelligence_service import _CompareContext, _diff_fields, _topic_overlap_all


@dataclass
class _MiniCase:
    id: uuid.UUID
    court: str | None
    decision_date: date | None
    outcome: str | None
    holding: str | None
    case_number: str | None
    legal_topics: list[str] = field(default_factory=list)
    # The diff helper doesn't touch these, but the dataclass needs them so
    # getattr() on the model surface works.
    case_title: str = ""
    document_id: uuid.UUID = field(default_factory=uuid.uuid4)
    jurisdiction: str = "GH"
    judges: list[str] | None = None
    facts_summary: str | None = None
    created_at: Any = None
    parties: list[Any] = field(default_factory=list)


def test_diff_surfaces_only_differing_fields() -> None:
    a = _MiniCase(
        id=uuid.uuid4(),
        court="Supreme Court",
        decision_date=date(2020, 1, 1),
        outcome="appeal allowed",
        holding="Same",
        case_number="J4/1/2020",
    )
    b = _MiniCase(
        id=uuid.uuid4(),
        court="Court of Appeal",  # differs
        decision_date=date(2020, 1, 1),  # same
        outcome="appeal dismissed",  # differs
        holding="Same",  # same
        case_number="J4/2/2020",  # differs
    )
    ctx = _CompareContext(cases=[a, b], by_id={str(a.id): a, str(b.id): b})  # type: ignore[arg-type]
    diffs = _diff_fields(ctx)
    fields = {d.field for d in diffs}
    assert "Court" in fields
    assert "Outcome" in fields
    assert "Case number" in fields
    # Same across both — must be omitted.
    assert "Decision date" not in fields
    assert "Holding" not in fields


def test_topic_overlap_returns_intersection() -> None:
    a = _MiniCase(
        id=uuid.uuid4(),
        court=None,
        decision_date=None,
        outcome=None,
        holding=None,
        case_number=None,
        legal_topics=["contract", "property", "tort"],
    )
    b = _MiniCase(
        id=uuid.uuid4(),
        court=None,
        decision_date=None,
        outcome=None,
        holding=None,
        case_number=None,
        legal_topics=["property", "tort", "criminal"],
    )
    c = _MiniCase(
        id=uuid.uuid4(),
        court=None,
        decision_date=None,
        outcome=None,
        holding=None,
        case_number=None,
        legal_topics=["property"],
    )
    overlap = _topic_overlap_all([a, b, c])  # type: ignore[arg-type]
    assert overlap == ["property"]


def test_topic_overlap_empty_when_no_common_topic() -> None:
    a = _MiniCase(
        id=uuid.uuid4(),
        court=None,
        decision_date=None,
        outcome=None,
        holding=None,
        case_number=None,
        legal_topics=["contract"],
    )
    b = _MiniCase(
        id=uuid.uuid4(),
        court=None,
        decision_date=None,
        outcome=None,
        holding=None,
        case_number=None,
        legal_topics=["criminal"],
    )
    assert _topic_overlap_all([a, b]) == []  # type: ignore[arg-type]
