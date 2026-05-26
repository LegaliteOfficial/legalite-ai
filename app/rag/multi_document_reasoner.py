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

from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.tenant_context import TenantContext
from app.rag.hybrid_retriever import HybridRetriever
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


class MultiDocumentReasoner:
    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        reranker: BaseReranker,
        buckets: tuple[str | None, ...] | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._buckets = buckets or DEFAULT_LAW_BUCKETS

    def gather(
        self, question: str, *, tenant: TenantContext | None = None
    ) -> CrossDocResult:
        settings = get_settings()
        per_bucket = max(2, settings.hybrid_topk_fused // max(1, len(self._buckets)))

        merged: list[RetrievedPassage] = []
        bucket_counts: dict[str, int] = {}

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

        if not merged:
            return CrossDocResult(passages=[], bucket_counts=bucket_counts)

        # Dedup by vector_id (same chunk may appear in `None` bucket + typed bucket).
        seen: set[str] = set()
        deduped: list[RetrievedPassage] = []
        for p in merged:
            key = p.vector_id or f"text:{hash(p.content)}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)

        final = self._reranker.rerank(
            question, deduped, top_n=settings.rag_topk_final
        )
        logger.info(
            "multidoc.gathered",
            buckets=bucket_counts,
            merged=len(merged),
            deduped=len(deduped),
            final=len(final),
        )
        return CrossDocResult(passages=final, bucket_counts=bucket_counts)
