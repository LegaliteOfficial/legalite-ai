"""Legal authority ranking — controls how the prompt presents passages.

Ghana law follows a clear hierarchy of authority. Inside the RAG pipeline
we encode that hierarchy as a numeric rank so the prompt can be built in
tiers (highest-authority first) and so the answer-generation contract can
state the rule plainly: a tenant memo never overrides a statute.

The numeric ranks are arbitrary; only their ORDER matters. We keep gaps
between tiers so future categories (e.g. ``treaty``) can be inserted
without renumbering everything else.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from app.db.enums import DocumentOrigin, VisibilityScope
from app.rag.retriever import RetrievedPassage

SourceScope = Literal["global", "tenant_private"]


# Authority tier name → numeric rank. Higher = more authoritative.
# Constitution is supreme; tenant strategy docs sit at the floor.
_TIER_RANK: dict[str, int] = {
    "constitution": 100,
    "act": 90,
    "regulation": 80,
    "public_case_law": 70,  # case_law / precedent / judgement under GLOBAL
    "tenant_sample_case": 50,
    "tenant_legal_opinion": 45,
    "tenant_internal_memo": 40,
    "tenant_private_reference": 35,
    "tenant_template": 20,
    "tenant_strategy_document": 10,
    "unknown": 60,  # falls between regulation and case_law as a neutral default
}

# Human-readable label rendered in the prompt for each tier.
TIER_LABEL: dict[str, str] = {
    "constitution": "Constitution",
    "act": "Acts (statutes)",
    "regulation": "Regulations / Legislative Instruments",
    "public_case_law": "Public case law",
    "tenant_sample_case": "Your firm's sample cases",
    "tenant_legal_opinion": "Your firm's legal opinions",
    "tenant_internal_memo": "Your firm's internal memos",
    "tenant_private_reference": "Your firm's private references",
    "tenant_template": "Your firm's templates",
    "tenant_strategy_document": "Your firm's strategy documents",
    "unknown": "Other supporting materials",
}

# Ordered list of tiers (highest → lowest). Drives prompt section order.
TIER_ORDER: tuple[str, ...] = (
    "constitution",
    "act",
    "regulation",
    "public_case_law",
    "unknown",
    "tenant_sample_case",
    "tenant_legal_opinion",
    "tenant_internal_memo",
    "tenant_private_reference",
    "tenant_template",
    "tenant_strategy_document",
)


@dataclass(slots=True, frozen=True)
class AuthorityClassification:
    """Stable identity for a passage's place in the hierarchy."""

    tier: str
    rank: int
    source_scope: SourceScope

    @property
    def label(self) -> str:
        return TIER_LABEL.get(self.tier, "Other supporting materials")

    @property
    def is_global(self) -> bool:
        return self.source_scope == "global"


def classify(passage: RetrievedPassage) -> AuthorityClassification:
    """Map a passage to its authority tier deterministically.

    Resolution order: explicit ``document_origin`` (most precise) → fall back
    to ``law_type`` (auto-detected by the chunker) → fall back to ``unknown``.
    ``visibility_scope`` then disambiguates global vs tenant case_law.
    """
    scope: SourceScope = (
        "tenant_private"
        if passage.visibility_scope == VisibilityScope.TENANT_PRIVATE.value
        else "global"
    )

    tier = _tier_from_origin(passage.document_origin, scope)
    if tier is None:
        tier = _tier_from_law_type(passage.law_type, scope)
    if tier is None:
        tier = "unknown"

    return AuthorityClassification(
        tier=tier,
        rank=_TIER_RANK[tier],
        source_scope=scope,
    )


def annotate(passages: list[RetrievedPassage]) -> list[RetrievedPassage]:
    """Populate ``legal_authority_rank`` + ``source_scope`` on each passage.

    Mutates the passages in place and returns the same list for chaining.
    Idempotent — re-running on an already-annotated list is a no-op.
    """
    for p in passages:
        cls = classify(p)
        p.legal_authority_rank = cls.rank
        p.source_scope = cls.source_scope
        # The tier label is useful for both the prompt and the response
        # payload — stash it on the passage to avoid re-classifying later.
        p.authority_tier = cls.tier
    return passages


def partition_by_tier(
    passages: list[RetrievedPassage],
) -> dict[str, list[RetrievedPassage]]:
    """Group passages by tier, preserving the inbound order within each tier.

    The inbound order is the post-rerank order (rerank_score desc), so each
    tier list is already sorted by relevance within its authority bucket.
    """
    buckets: dict[str, list[RetrievedPassage]] = defaultdict(list)
    for p in passages:
        cls = classify(p)
        buckets[cls.tier].append(p)
    return dict(buckets)


def ordered_tiers(
    buckets: dict[str, list[RetrievedPassage]],
) -> list[tuple[str, list[RetrievedPassage]]]:
    """Return ``(tier, passages)`` pairs in the canonical authority order.

    Empty tiers are dropped so the prompt doesn't include blank sections.
    """
    return [(tier, buckets[tier]) for tier in TIER_ORDER if buckets.get(tier)]


# ---------------------------------------------------------------------------
# Internals — mapping origin / law_type → tier
# ---------------------------------------------------------------------------


def _tier_from_origin(origin: str | None, scope: SourceScope) -> str | None:
    if not origin:
        return None
    try:
        o = DocumentOrigin(origin)
    except ValueError:
        return None

    # Global side — origin maps straight to its tier.
    if scope == "global":
        if o is DocumentOrigin.CONSTITUTION:
            return "constitution"
        if o is DocumentOrigin.ACT:
            return "act"
        if o is DocumentOrigin.REGULATION:
            return "regulation"
        if o in (DocumentOrigin.CASE_LAW, DocumentOrigin.PRECEDENT, DocumentOrigin.JUDGEMENT):
            return "public_case_law"
        # Tenant-private origins should not appear under GLOBAL scope, but if
        # ingestion ever lets one slip, treat it as unknown rather than blindly
        # promoting it.
        return "unknown"

    # Tenant-private side.
    if o is DocumentOrigin.SAMPLE_CASE:
        return "tenant_sample_case"
    if o is DocumentOrigin.LEGAL_OPINION:
        return "tenant_legal_opinion"
    if o is DocumentOrigin.INTERNAL_MEMO:
        return "tenant_internal_memo"
    if o is DocumentOrigin.PRIVATE_REFERENCE:
        return "tenant_private_reference"
    if o is DocumentOrigin.TEMPLATE:
        return "tenant_template"
    if o is DocumentOrigin.STRATEGY_DOCUMENT:
        return "tenant_strategy_document"
    # Global-only origin somehow ended up under tenant scope — same defensive
    # downgrade as above.
    return "unknown"


def _tier_from_law_type(law_type: str | None, scope: SourceScope) -> str | None:
    """Older chunks have no document_origin yet — fall back to law_type."""
    if not law_type:
        return None
    lt = law_type.lower()
    if scope == "tenant_private":
        # Tenant chunks ingested before document_origin existed — treat as
        # private reference. They're still subordinated below all global law.
        return "tenant_private_reference"
    if lt == "constitution":
        return "constitution"
    if lt == "act":
        return "act"
    if lt == "regulation":
        return "regulation"
    if lt in ("case_law", "precedent", "judgement"):
        return "public_case_law"
    return None
