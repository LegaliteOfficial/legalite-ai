"""Snapshot of "what's actually in the corpus" — for capability-aware UX.

When the assistant can't answer a question, the worst thing it can do
is shrug and say "I don't know". The next-worst thing is to list every
file in the database. The right behaviour is to surface enough about
its current coverage that the user can ask a question the corpus
actually contains.

This service produces that snapshot — counts per doc_type plus a
handful of recent titles per doc_type — and caches it at module level
so the conversational responder can call it on every "unclear" turn
without hammering Postgres.

Cache shape
-----------
- Module-level singleton snapshot, recomputed on first miss after the
  TTL expires.
- TTL is intentionally short (5 minutes by default). The corpus grows
  via uploads from the same dashboard the assistant lives in; stale
  data here would mean "I just uploaded this Act, why doesn't the
  assistant know it exists?". 5 minutes is the sweet spot between
  reflecting fresh uploads and not pounding Postgres for every
  conversational turn.
- The snapshot is GLOBAL-only by design. Tenant-private documents
  shouldn't surface in a "here's what I have" tour exposed to
  anonymous /ask callers — that would leak the existence of private
  uploads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.enums import VisibilityScope
from app.db.models.legal_document import LegalDocument


@dataclass(frozen=True, slots=True)
class CorpusSnapshot:
    """Frozen view of the global corpus at a point in time.

    ``counts_by_doc_type`` is the headline number for each category
    the user has uploaded. ``sample_titles_by_doc_type`` is up to
    ``_SAMPLES_PER_TYPE`` recent titles per category — used to seed
    realistic example questions ("Summarize Republic v. Danquah") so
    the assistant's suggestions reference real artefacts instead of
    placeholders.
    """

    counts_by_doc_type: dict[str, int]
    sample_titles_by_doc_type: dict[str, list[str]]
    total_docs: int
    snapshot_at: float

    @property
    def is_empty(self) -> bool:
        return self.total_docs == 0


# How many sample titles to pull per doc_type. 3 is enough to drive
# question seeding without bloating the snapshot or the conversational
# response — neither the LLM nor the user needs more than a few real
# examples to anchor their suggestions.
_SAMPLES_PER_TYPE = 3

# Default cache TTL. Overridable per-instance for tests.
_DEFAULT_TTL_SECONDS = 300

# Module-level cache. Shared across QAService instances in the same
# process so the conversational responder doesn't requery Postgres on
# every "unclear" turn.
_cache: CorpusSnapshot | None = None


class CorpusInventoryService:
    """Read the corpus snapshot used by the capability-aware responses.

    Construct with the request's DB session; the actual data is cached
    at module level so subsequent calls within the TTL skip Postgres
    entirely.
    """

    def __init__(self, db: Session, *, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self._db = db
        self._ttl = ttl_seconds

    def get(self) -> CorpusSnapshot:
        global _cache
        now = time.monotonic()
        if _cache is not None and (now - _cache.snapshot_at) < self._ttl:
            return _cache
        _cache = self._build(now)
        return _cache

    # ---- internals ---------------------------------------------------------

    def _build(self, snapshot_at: float) -> CorpusSnapshot:
        """Two queries: counts grouped by doc_type + sample titles per type.

        The COUNT query is cheap and covers the headline numbers. The
        sample titles query has a LIMIT per type so total rows stay
        bounded regardless of corpus size; we issue it per type
        sequentially because GROUP BY + window functions to combine
        them in one query would be more complex than the savings
        justify at typical corpus sizes (<10K docs).
        """
        count_rows = self._db.execute(
            select(LegalDocument.doc_type, func.count())
            .where(LegalDocument.status == "ingested")
            .where(LegalDocument.visibility_scope == VisibilityScope.GLOBAL.value)
            .group_by(LegalDocument.doc_type)
        ).all()
        counts: dict[str, int] = {}
        for doc_type, n in count_rows:
            counts[str(doc_type or "Uncategorised")] = int(n)

        samples: dict[str, list[str]] = {}
        for doc_type in counts:
            sample_rows = self._db.execute(
                select(LegalDocument.title)
                .where(LegalDocument.doc_type == doc_type)
                .where(LegalDocument.status == "ingested")
                .where(LegalDocument.visibility_scope == VisibilityScope.GLOBAL.value)
                .order_by(LegalDocument.created_at.desc())
                .limit(_SAMPLES_PER_TYPE)
            ).scalars()
            samples[doc_type] = [str(t) for t in sample_rows if t]

        return CorpusSnapshot(
            counts_by_doc_type=counts,
            sample_titles_by_doc_type=samples,
            total_docs=sum(counts.values()),
            snapshot_at=snapshot_at,
        )


def reset_cache() -> None:
    """Drop the module-level cache. Test hook + manual refresh helper."""
    global _cache
    _cache = None
