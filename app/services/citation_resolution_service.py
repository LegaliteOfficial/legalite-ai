"""Resolve case citations to ingested ``legal_cases`` rows.

After a batch of judgments is ingested, this service walks every
``case_citations`` row whose ``cited_case_id`` is still NULL and tries to
match the raw citation text against the corpus.

Matching strategy (highest-confidence rules first):

1. **Case number exact match** — e.g. ``Suit No. J4/12/2020`` extracted
   from the raw citation appears in ``legal_cases.case_number``.
2. **Reporter + year + number** — ``[2020] GHASC 5`` decomposed into
   year/reporter/num, looked up against ``legal_cases.case_number``.
3. **Title fuzzy match** — case title normalized + scored against
   ``legal_cases.case_title`` using token-overlap + Jaro-like similarity.

Only edges whose score >= ``CITATION_RESOLUTION_THRESHOLD`` are auto-linked.
Lower-scored matches are left unresolved so a human can audit and link
manually rather than risk a wrong precedent edge.

Tenant safety
-------------
A tenant-private citing case can reference a global case (forward edge).
A global case must NEVER be auto-linked to a tenant-private cited case
(that would let one firm's private memo show up as "cited by" a public
judgment for everyone). The resolver enforces this explicitly.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.telemetry import record_citation_resolution
from app.db.enums import VisibilityScope
from app.db.models.case_citation import CaseCitation
from app.db.models.legal_case import LegalCase

logger = get_logger(__name__)


_REPORTER_RE = re.compile(
    r"\[(?P<year>\d{4}(?:-\d{2,4})?)\]\s+(?P<reporter>[A-Z]{3,7})\s+(?P<num>\d+)",
    re.IGNORECASE,
)
_SUIT_NUMBER_RE = re.compile(
    r"\b(?:Suit|Civil\s+Appeal|Criminal\s+Appeal|Writ|Application|Motion)\s+No\.?\s*"
    r"([A-Z]{1,4}/?\d+/?[A-Z]?\d*(?:/\d{2,4})?)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(slots=True)
class ResolutionResult:
    """Outcome of resolving a single citation row."""

    citation_id: uuid.UUID
    resolved: bool
    cited_case_id: uuid.UUID | None
    confidence: float
    reason: str


class CitationResolutionService:
    def __init__(
        self,
        db: Session,
        *,
        threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self._db = db
        self._threshold = (
            threshold if threshold is not None else settings.citation_resolution_threshold
        )

    # ---- public API --------------------------------------------------------

    def resolve_for_case(self, citing_case_id: uuid.UUID) -> list[ResolutionResult]:
        """Resolve every unresolved citation outgoing from a single case."""
        rows = self._db.scalars(
            select(CaseCitation).where(
                CaseCitation.citing_case_id == citing_case_id,
                CaseCitation.cited_case_id.is_(None),
            )
        ).all()
        results = [self._resolve_one(c) for c in rows]
        self._db.flush()
        logger.info(
            "citation.resolved.batch",
            citing_case_id=str(citing_case_id),
            considered=len(results),
            linked=sum(1 for r in results if r.resolved),
        )
        return results

    def resolve_all_unlinked(self, limit: int = 500) -> list[ResolutionResult]:
        """Sweep up to ``limit`` unresolved edges across the corpus.

        Intended for a periodic batch job (or a one-shot script). Bounded
        so the sweep stays predictable even on large corpora.
        """
        rows = self._db.scalars(
            select(CaseCitation)
            .where(CaseCitation.cited_case_id.is_(None))
            .limit(limit)
        ).all()
        results = [self._resolve_one(c) for c in rows]
        self._db.flush()
        return results

    # ---- internals ---------------------------------------------------------

    def _resolve_one(self, edge: CaseCitation) -> ResolutionResult:
        raw = edge.cited_raw_text or ""
        title = edge.cited_case_title or ""

        # The citing case's scope determines which cases are eligible targets.
        # Global cases may only link to global cases; tenant-private cases
        # may link to the same org's private corpus AND to global cases.
        citing_case = self._db.get(LegalCase, edge.citing_case_id)
        if citing_case is None:
            return ResolutionResult(
                citation_id=edge.id,
                resolved=False,
                cited_case_id=None,
                confidence=0.0,
                reason="citing case not found",
            )

        eligible_cases = self._eligible_targets(citing_case)

        best_id: uuid.UUID | None = None
        best_score = 0.0
        best_reason = "no match"

        # 1. Case number exact match.
        suit_match = _SUIT_NUMBER_RE.search(raw)
        if suit_match:
            needle = suit_match.group(0)
            for c in eligible_cases:
                if c.case_number and needle.lower() in c.case_number.lower():
                    best_id = c.id
                    best_score = 1.0
                    best_reason = f"case_number exact: {needle}"
                    break

        # 2. Reporter form.
        if best_id is None:
            rep = _REPORTER_RE.search(raw)
            if rep:
                normalized = f"[{rep.group('year')}] {rep.group('reporter').upper()} {rep.group('num')}"
                for c in eligible_cases:
                    if c.case_number and normalized.lower() in c.case_number.lower():
                        best_id = c.id
                        best_score = 0.95
                        best_reason = f"reporter exact: {normalized}"
                        break

        # 3. Title fuzzy match.
        if best_id is None and title:
            for c in eligible_cases:
                if not c.case_title:
                    continue
                score = _title_similarity(title, c.case_title)
                if score > best_score:
                    best_score = score
                    best_id = c.id
                    best_reason = f"title sim={score:.2f}"

        resolved = best_id is not None and best_score >= self._threshold
        if resolved:
            edge.cited_case_id = best_id
            edge.resolution_confidence = Decimal(f"{best_score:.3f}")
            edge.resolved_at = datetime.now(UTC)
        record_citation_resolution(resolved)

        return ResolutionResult(
            citation_id=edge.id,
            resolved=resolved,
            cited_case_id=best_id,
            confidence=best_score,
            reason=best_reason if best_id else "no candidate",
        )

    def _eligible_targets(self, citing: LegalCase) -> list[LegalCase]:
        """Return the LegalCase rows the citing case is allowed to link to.

        Global → global only. Tenant-private → its own org + global.
        Self-citation is excluded.
        """
        if citing.visibility_scope == VisibilityScope.GLOBAL.value:
            stmt = select(LegalCase).where(
                LegalCase.visibility_scope == VisibilityScope.GLOBAL.value,
                LegalCase.id != citing.id,
            )
        else:
            stmt = select(LegalCase).where(
                or_(
                    LegalCase.visibility_scope == VisibilityScope.GLOBAL.value,
                    (LegalCase.visibility_scope == VisibilityScope.TENANT_PRIVATE.value)
                    & (LegalCase.organization_id == citing.organization_id),
                ),
                LegalCase.id != citing.id,
            )
        return list(self._db.scalars(stmt).all())


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GraphNode:
    case_id: uuid.UUID
    case_title: str
    case_number: str | None
    court: str | None
    confidence: float


def get_citation_graph(
    db: Session,
    case_id: uuid.UUID,
    *,
    organization_id: uuid.UUID | None,
    max_chain_depth: int = 4,
) -> dict[str, list[GraphNode]]:
    """Return ``{cites, cited_by, precedent_chain}`` for one case.

    The traversal respects tenant scope: only edges where both endpoints
    are visible to the requesting org are surfaced. Cycle-safe via a
    visited set; ``max_chain_depth`` caps recursion to keep responses small.
    """
    root = db.get(LegalCase, case_id)
    if root is None:
        return {"cites": [], "cited_by": [], "precedent_chain": []}

    if not _case_visible(root, organization_id):
        return {"cites": [], "cited_by": [], "precedent_chain": []}

    cites = _outgoing(db, root, organization_id)
    cited_by = _incoming(db, root, organization_id)
    chain = _precedent_chain(db, root, organization_id, max_chain_depth)

    return {"cites": cites, "cited_by": cited_by, "precedent_chain": chain}


def _case_visible(case: LegalCase, org_id: uuid.UUID | None) -> bool:
    if case.visibility_scope == VisibilityScope.GLOBAL.value:
        return True
    return (
        org_id is not None
        and case.organization_id is not None
        and case.organization_id == org_id
    )


def _outgoing(
    db: Session, case: LegalCase, org_id: uuid.UUID | None
) -> list[GraphNode]:
    rows = db.scalars(
        select(CaseCitation).where(
            CaseCitation.citing_case_id == case.id,
            CaseCitation.cited_case_id.is_not(None),
        )
    ).all()
    return list(_to_nodes(db, (r.cited_case_id for r in rows), org_id, edges=rows, role="cited"))


def _incoming(
    db: Session, case: LegalCase, org_id: uuid.UUID | None
) -> list[GraphNode]:
    rows = db.scalars(
        select(CaseCitation).where(CaseCitation.cited_case_id == case.id)
    ).all()
    return list(_to_nodes(db, (r.citing_case_id for r in rows), org_id, edges=rows, role="citing"))


def _to_nodes(
    db: Session,
    case_ids: Iterable[uuid.UUID | None],
    org_id: uuid.UUID | None,
    *,
    edges: list[CaseCitation],
    role: str,
) -> Iterable[GraphNode]:
    seen: set[uuid.UUID] = set()
    id_to_conf: dict[uuid.UUID, float] = {}
    for e in edges:
        if role == "cited" and e.cited_case_id is not None:
            id_to_conf[e.cited_case_id] = float(e.resolution_confidence or 0)
        elif role == "citing":
            id_to_conf[e.citing_case_id] = float(e.resolution_confidence or 0)

    for cid in case_ids:
        if cid is None or cid in seen:
            continue
        seen.add(cid)
        case = db.get(LegalCase, cid)
        if case is None or not _case_visible(case, org_id):
            continue
        yield GraphNode(
            case_id=case.id,
            case_title=case.case_title,
            case_number=case.case_number,
            court=case.court,
            confidence=id_to_conf.get(cid, 1.0),
        )


def _precedent_chain(
    db: Session,
    root: LegalCase,
    org_id: uuid.UUID | None,
    max_depth: int,
) -> list[GraphNode]:
    """Greedy chain: follow the highest-confidence outgoing edge from each
    node until we hit a leaf, a cycle, or ``max_depth`` hops."""
    chain: list[GraphNode] = []
    visited: set[uuid.UUID] = {root.id}
    current = root
    for _ in range(max_depth):
        next_edge = db.scalar(
            select(CaseCitation)
            .where(
                CaseCitation.citing_case_id == current.id,
                CaseCitation.cited_case_id.is_not(None),
                CaseCitation.cited_case_id.notin_(list(visited)),
            )
            .order_by(CaseCitation.resolution_confidence.desc().nulls_last())
        )
        if next_edge is None or next_edge.cited_case_id is None:
            break
        nxt = db.get(LegalCase, next_edge.cited_case_id)
        if nxt is None or not _case_visible(nxt, org_id):
            break
        visited.add(nxt.id)
        chain.append(
            GraphNode(
                case_id=nxt.id,
                case_title=nxt.case_title,
                case_number=nxt.case_number,
                court=nxt.court,
                confidence=float(next_edge.resolution_confidence or 0),
            )
        )
        current = nxt
    return chain


# ---------------------------------------------------------------------------
# Similarity helper
# ---------------------------------------------------------------------------


def _title_similarity(a: str, b: str) -> float:
    """Cheap token-overlap score in [0, 1].

    Tokens are lowercased alphanumerics. We use Jaccard-ish:
    ``|tokens(a) ∩ tokens(b)| / |tokens(a) ∪ tokens(b)|``. This catches
    "Asare v Attorney-General" vs "ASARE V THE ATTORNEY GENERAL" without
    needing an external fuzzy lib.
    """
    ta = {t.lower() for t in _TOKEN_RE.findall(a) if t}
    tb = {t.lower() for t in _TOKEN_RE.findall(b) if t}
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union)
