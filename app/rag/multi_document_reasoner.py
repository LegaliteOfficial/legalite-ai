"""Cross-document legal reasoning.

A question like "Can a pregnant employee be dismissed?" should pull from:
  - the Constitution (fundamental rights),
  - the Labour Act (employment / maternity protections),
  - relevant regulations,
  - case law (if any has been ingested).

We do this by:
  1. Running the hybrid retriever once per `law_type` bucket so each
     category gets fair representation (a Constitution clause shouldn't be
     pushed out of the top-k by ten near-duplicate Labour Act sections).
  2. Concatenating the per-bucket results into a single candidate pool.
  3. Reranking the pool together to produce one cross-document ordering.

The class returns the final reranked passages; the caller (QAService) then
runs `AnswerGenerator` once on the merged context, which is where the actual
synthesis happens via the prompt's "surface conflicts" rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.tenant_context import TenantContext
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import BaseReranker
from app.rag.retriever import RetrievedPassage

logger = get_logger(__name__)

# Order matters: when budgets are tight we want constitutional/statutory
# foundations before case-law commentary.
DEFAULT_LAW_BUCKETS: tuple[str | None, ...] = (
    "constitution",
    "act",
    "regulation",
    "case_law",
    None,  # ungrouped — picks up anything without a detected law_type
)


@dataclass(slots=True)
class CrossDocResult:
    passages: list[RetrievedPassage]
    bucket_counts: dict[str, int]
    # Phase A diagnostics — what the query understanding layer actually
    # ran. Empty list means the rewriter was disabled or returned nothing.
    query_variants: list[str] = field(default_factory=list)


class MultiDocumentReasoner:
    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        reranker: BaseReranker,
        buckets: tuple[str | None, ...] | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._buckets = buckets or DEFAULT_LAW_BUCKETS
        # ``rewriter`` is optional. When None, gather() runs single-query
        # retrieval (the pre-Phase-A behavior). Callers that want to
        # opt-out of rewriting can pass ``rewriter=None`` explicitly.
        self._rewriter = rewriter

    def gather(
        self, question: str, *, tenant: TenantContext | None = None
    ) -> CrossDocResult:
        settings = get_settings()
        per_bucket = max(2, settings.hybrid_topk_fused // max(1, len(self._buckets)))

        merged: list[RetrievedPassage] = []
        bucket_counts: dict[str, int] = {}

        # Stage 1 — original question, bucketed retrieval. This is the
        # path we trust most: per-law-type retrieval ensures every
        # authority tier (constitution, act, regulation, case_law) gets
        # fair representation in the candidate pool.
        for law_type in self._buckets:
            result = self._retriever.retrieve(
                question,
                law_type=law_type,
                top_k_final=per_bucket,
                tenant=tenant,
            )
            label = law_type or "unspecified"
            bucket_counts[label] = result.n_fused
            merged.extend(result.passages)

        # Stage 2 — semantic variants, single (non-bucketed) retrieval
        # each. We don't re-bucket variants because the original already
        # spread the candidate pool across tiers; variants are about
        # recall on different phrasings of the SAME intent, not
        # different authority tiers. Cost: 1 Qdrant search per variant
        # instead of 5.
        variants: list[str] = []
        if self._rewriter is not None:
            variants = self._rewriter.rewrite(question)
        for v_idx, variant in enumerate(variants):
            result = self._retriever.retrieve(
                variant,
                law_type=None,
                top_k_final=per_bucket,
                tenant=tenant,
            )
            bucket_counts[f"rewrite_{v_idx}"] = result.n_fused
            merged.extend(result.passages)

        if not merged:
            return CrossDocResult(
                passages=[],
                bucket_counts=bucket_counts,
                query_variants=variants,
            )

        # Dedup by vector_id. The variant retrievals will frequently
        # surface chunks the bucketed pass already found — that's fine,
        # we just collapse them to one passage and keep the highest score
        # observed across all retrievals.
        best_by_key: dict[str, RetrievedPassage] = {}
        for p in merged:
            key = p.vector_id or f"text:{hash(p.content)}"
            existing = best_by_key.get(key)
            if existing is None or (p.score or 0.0) > (existing.score or 0.0):
                best_by_key[key] = p
        deduped = list(best_by_key.values())

        # Rerank the unified pool against the ORIGINAL question — the
        # variants were a recall tool, not the user's actual intent. The
        # cross-encoder should score relevance to what the user typed.
        final = self._reranker.rerank(
            question, deduped, top_n=settings.rag_topk_final
        )
        logger.info(
            "multidoc.gathered",
            buckets=bucket_counts,
            merged=len(merged),
            deduped=len(deduped),
            final=len(final),
            variants=len(variants),
        )
        return CrossDocResult(
            passages=final,
            bucket_counts=bucket_counts,
            query_variants=variants,
        )
