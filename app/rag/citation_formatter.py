"""Turn retrieved passages into Citation + SourceUsed payloads.

v2 changes:
- Deduplicate by structural identity (law + article + section + clause + page).
- Rank citations by best relevance among duplicates (max of rerank or vector score).
- Preserve legacy fields (`law`, `article`, `section`, `page`) so v1 clients
  continue to read the same response.
- Emit `sources_used` (richer per-passage detail with excerpts).
"""

from __future__ import annotations

from app.rag.retriever import RetrievedPassage
from app.schemas.qa import Citation, SourceUsed

_EXCERPT_MAX_CHARS = 320


def _passage_relevance(p: RetrievedPassage) -> float:
    """Best available relevance signal. Rerank > raw retrieval score."""
    if p.rerank_score is not None:
        return float(p.rerank_score)
    return float(p.score)


def _citation_key(p: RetrievedPassage) -> tuple[str, str, str, str, str]:
    return (
        (p.law or "").strip().lower(),
        (p.article or "").strip(),
        (p.section or "").strip(),
        (p.clause or "").strip(),
        _format_page(p.page_start, p.page_end),
    )


def format_citations(passages: list[RetrievedPassage]) -> list[Citation]:
    """Dedup + sort by relevance descending."""
    best_by_key: dict[tuple[str, str, str, str, str], RetrievedPassage] = {}
    score_by_key: dict[tuple[str, str, str, str, str], float] = {}

    for p in passages:
        key = _citation_key(p)
        rel = _passage_relevance(p)
        if key not in score_by_key or rel > score_by_key[key]:
            score_by_key[key] = rel
            best_by_key[key] = p

    ordered_keys = sorted(score_by_key, key=lambda k: score_by_key[k], reverse=True)
    return [
        _passage_to_citation(best_by_key[k], score_by_key[k]) for k in ordered_keys
    ]


def format_sources(passages: list[RetrievedPassage]) -> list[SourceUsed]:
    return [_passage_to_source(p) for p in passages]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _passage_to_citation(p: RetrievedPassage, relevance: float) -> Citation:
    page = _format_page(p.page_start, p.page_end)
    return Citation(
        law=p.law or "",
        article=p.article or "",
        section=p.section or "",
        page=page,
        law_name=p.law or "",
        law_type=p.law_type or "",
        act_number=p.act_number or "",
        chapter=p.chapter or "",
        part=p.part or "",
        clause=p.clause or "",
        subsection=p.subsection or "",
        page_number=p.page_start,
        relevance=round(relevance, 4),
        source_scope=p.source_scope or "global",  # type: ignore[arg-type]
        document_origin=p.document_origin,
    )


def _passage_to_source(p: RetrievedPassage) -> SourceUsed:
    excerpt = p.content.strip().replace("\n", " ")
    if len(excerpt) > _EXCERPT_MAX_CHARS:
        excerpt = excerpt[:_EXCERPT_MAX_CHARS].rsplit(" ", 1)[0] + "…"
    return SourceUsed(
        law_name=p.law or "",
        law_type=p.law_type or "",
        section=p.section or "",
        article=p.article or "",
        page=_format_page(p.page_start, p.page_end),
        excerpt=excerpt,
        score=round(_passage_relevance(p), 4),
        source_scope=p.source_scope or "global",  # type: ignore[arg-type]
        document_origin=p.document_origin,
    )


def _format_page(start: int | None, end: int | None) -> str:
    if start is None:
        return ""
    if end is None or end == start:
        return str(start)
    return f"{start}-{end}"
