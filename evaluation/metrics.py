"""Metrics for the legal RAG evaluation harness.

Each metric takes a list of `EvalRun` records (one per dataset entry) and
returns a single float, or ``None`` when the metric has nothing to measure
(no eligible cases) — ``None`` means "not applicable", never conflate it
with ``0.0`` ("measured and the result was zero"). The runner renders
``None`` as ``N/A``.

Phase 2 change: metrics must never blend verified and unverified cases into
one number that LOOKS authoritative. ``summarize()`` computes the
exploratory view over every case (labelled as such); ``summarize_verified()``
computes the same metrics restricted to ``EvalCase.verified`` rows and is
the only one that should ever be called "the baseline". See
evaluation/datasets/README.md for what "verified" means and why this
separation exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.qa import AskResponse


@dataclass(slots=True)
class EvalCase:
    id: str
    question: str
    # Legacy single-citation fields — kept for backward compatibility with
    # every pre-Phase-2 case. Superseded by `expected_citations` for cases
    # that need more than one expected source (see expected_citation_list).
    expected_law: str | None = None
    expected_section: str | None = None
    expected_article: str | None = None
    expected_law_type: str | None = None
    must_refuse: bool = False
    # Phase 1 additions.
    category: str | None = None  # see evaluation/datasets/README.md for the list
    difficulty: str | None = None  # easy | medium | hard
    verified: bool = False

    # Phase 2 additions — all optional and additive; every existing row
    # loads unchanged via EvalCase(**row).
    #
    # A list of {"law": ..., "section": ..., "article": ...} dicts, for
    # cases needing more than one expected citation (multi-document
    # questions). When absent, callers fall back to the single
    # expected_law/section/article fields via expected_citation_list().
    expected_citations: list[dict[str, str | None]] = field(default_factory=list)
    # Document/law TITLES (matched against Citation.law_name /
    # SourceUsed.law_name — the corpus stores a document's title as its
    # `law` field, see app/vector_store/qdrant_client.py's VectorPayload)
    # that should be surfaced SOMEWHERE in the response, even if not
    # specifically cited with a matching section. Weaker signal than
    # expected_citations — "did it pull from this source at all".
    expected_documents: list[str] = field(default_factory=list)
    # Substrings expected to appear in a retrieved passage's excerpt
    # (SourceUsed.excerpt). Approximate by nature — excerpts can be
    # truncated — treat a miss as "check manually", not proof of failure.
    expected_passages: list[str] = field(default_factory=list)
    # For the human reviewer's own reference during verification, and for
    # a person reading the report later. Deliberately NOT compared against
    # the generated answer by any metric: automated text-similarity
    # scoring needs either fuzzy string matching (brittle, produces false
    # confidence) or an LLM judge (explicitly excluded — an LLM cannot be
    # the source of legal ground truth). See evaluation/datasets/README.md.
    expected_answer: str | None = None
    # Free-text annotator remark — e.g. "drop this case if the corpus has
    # no dated material" or "verified against legal_documents id=...".
    # Never read by any metric; for humans only.
    notes: str | None = None


@dataclass(slots=True)
class EvalRun:
    case: EvalCase
    response: AskResponse


# ---- shared helpers ---------------------------------------------------------


_ALNUM_RE = re.compile(r"[a-z0-9]")


def law_name_matches(expected: str, actual: str) -> bool:
    """Case-insensitive substring match, but bounded so a numeric suffix
    can't create a false positive — 'Act 651' must NOT match 'Act 6510'
    (a different act) even though plain `in` says it does, while 'Labour
    Act' must still match 'Labour Act, 2003' (the deliberate, documented
    behavior: real citations carry extra qualifying text — a year, an act
    number — beyond the bare law name a reviewer writes).

    A match is accepted only when the character immediately before and
    after the expected substring (if any) is NOT itself alphanumeric —
    i.e. the expected text ends on a real word/token boundary in the
    actual string.
    """
    expected = expected.strip().lower()
    actual = actual.lower()
    if not expected:
        return False
    start = 0
    while True:
        idx = actual.find(expected, start)
        if idx == -1:
            return False
        before_ok = idx == 0 or not _ALNUM_RE.match(actual[idx - 1])
        after_idx = idx + len(expected)
        after_ok = after_idx >= len(actual) or not _ALNUM_RE.match(actual[after_idx])
        if before_ok and after_ok:
            return True
        start = idx + 1


def expected_citation_list(case: EvalCase) -> list[dict[str, str | None]]:
    """Normalize a case's expected citation(s) to a list of
    {law, section, article} dicts, whichever way they were specified."""
    if case.expected_citations:
        return case.expected_citations
    if case.expected_law:
        return [
            {
                "law": case.expected_law,
                "section": case.expected_section,
                "article": case.expected_article,
            }
        ]
    return []


# ---- metrics ---------------------------------------------------------------
#
# citation_accuracy@3 and recall@5 both key off `AskResponse.citations`
# (Citation.law_name / .section / .article — human-readable strings, not
# document_id/chunk_id). That list is built from the passages the LLM
# actually cited in its answer (see app/rag/citation_formatter.py), which
# is downstream of retrieval, reranking, AND the LLM's own choice of what
# to cite — the evaluator has no access to the raw pre-generation
# candidate set at all (QAService.ask() returns only the final
# AskResponse). So "recall@5" here means "did the expected law end up
# among the top-5 citations the ANSWER surfaced", not classic IR recall
# against the retriever's candidate pool. Precise, not a bug — but easy to
# over-read as a pure retrieval metric, so it's spelled out here and in
# evaluation/datasets/README.md.
#
# Neither metric checks whether the citation's passage text actually
# supports the generated answer (no semantic/entailment check) — only
# that a citation with a matching law name (and, for citation_accuracy,
# matching section/article) exists in the response. document_recall and
# passage_support_rate below narrow this further: the former checks
# whether an expected SOURCE was surfaced at all (via law_name, in either
# citations or the broader sources_used list); the latter is the only
# metric that inspects passage TEXT (SourceUsed.excerpt), and only via
# substring match — there is no "correct passage" check today beyond that.


def recall_at_k(runs: list[EvalRun], k: int = 5) -> float | None:
    """Per-case fraction of expected citations whose law appears in the
    top-k citations, averaged across eligible cases. None if no case has
    an expected citation set (nothing to measure)."""
    eligible = [r for r in runs if expected_citation_list(r.case) and not r.case.must_refuse]
    if not eligible:
        return None
    total = 0.0
    for r in eligible:
        expected = expected_citation_list(r.case)
        cited = [c.law_name for c in r.response.citations[:k] if c.law_name]
        found = sum(
            1
            for e in expected
            if (law := (e.get("law") or "").strip())
            and any(law_name_matches(law, c) for c in cited)
        )
        total += found / len(expected)
    return total / len(eligible)


def citation_accuracy(runs: list[EvalRun]) -> float | None:
    """Per-case fraction of expected citations matched by (law AND
    section/article, when specified) against a top-3 citation."""
    eligible = [r for r in runs if expected_citation_list(r.case) and not r.case.must_refuse]
    if not eligible:
        return None
    total = 0.0
    for r in eligible:
        expected = expected_citation_list(r.case)
        found = 0
        for e in expected:
            expected_law = (e.get("law") or "").strip()
            expected_sec = (e.get("section") or "").strip()
            expected_art = (e.get("article") or "").strip()
            for c in r.response.citations[:3]:
                law_match = bool(expected_law) and law_name_matches(expected_law, c.law_name or "")
                sec_match = not expected_sec or expected_sec == (c.section or "")
                art_match = not expected_art or expected_art == (c.article or "")
                if law_match and sec_match and art_match:
                    found += 1
                    break
        total += found / len(expected)
    return total / len(eligible)


def document_recall(runs: list[EvalRun]) -> float | None:
    """Per-case fraction of `expected_documents` (by title) surfaced
    anywhere in citations OR sources_used — a source being surfaced at
    all, independent of whether it was cited with a matching section.
    None if no case sets expected_documents."""
    eligible = [r for r in runs if r.case.expected_documents]
    if not eligible:
        return None
    total = 0.0
    for r in eligible:
        surfaced = {c.law_name.strip() for c in r.response.citations if c.law_name}
        surfaced |= {s.law_name.strip() for s in r.response.sources_used if s.law_name}
        expected = [d.strip() for d in r.case.expected_documents if d.strip()]
        found = sum(1 for d in expected if any(law_name_matches(d, s) for s in surfaced))
        total += found / len(expected)
    return total / len(eligible)


def passage_support_rate(runs: list[EvalRun]) -> float | None:
    """Per-case fraction of `expected_passages` substrings found in any
    retrieved passage's excerpt. Approximate — SourceUsed.excerpt can be
    truncated, so a miss here is a prompt to check manually, not proof of
    failure. None if no case sets expected_passages."""
    eligible = [r for r in runs if r.case.expected_passages]
    if not eligible:
        return None
    total = 0.0
    for r in eligible:
        excerpts = [s.excerpt.lower() for s in r.response.sources_used if s.excerpt]
        expected = [p.strip().lower() for p in r.case.expected_passages if p.strip()]
        found = sum(1 for p in expected if any(p in e for e in excerpts))
        total += found / len(expected)
    return total / len(eligible)


def refusal_correctness(runs: list[EvalRun]) -> float | None:
    """How well refusals align with `must_refuse`, over every case (this
    is the one metric that always applies — every case has a must_refuse
    value, default False). None only when there are zero cases at all."""
    if not runs:
        return None
    hits = 0
    for r in runs:
        is_low = r.response.confidence == "low"
        if r.case.must_refuse and is_low or not r.case.must_refuse and not is_low:
            hits += 1
    return hits / len(runs)


def hallucination_rate(runs: list[EvalRun]) -> float | None:
    """Heuristic: cases where confidence != 'low' but no citations were
    produced — the same floor computed live in production (see
    app.core.telemetry.record_hallucination_floor_if_ungrounded). A real
    hallucination (a wrong claim behind a real-looking but incorrect
    citation) is invisible to this check by construction; catching that
    needs the verified-citation comparison above, not this heuristic.
    None only when there are zero cases at all."""
    if not runs:
        return None
    bad = sum(1 for r in runs if r.response.confidence != "low" and not r.response.citations)
    return bad / len(runs)


# ---- aggregation -----------------------------------------------------------

# Shared display order — the runner prints this same tuple in three places
# (exploratory, verified baseline, per-category), and every metric name
# used here must match a key _compute_metrics() actually returns.
METRIC_ORDER = (
    "recall@5",
    "citation_accuracy@3",
    "document_recall",
    "passage_support_rate",
    "refusal_correctness",
    "hallucination_rate",
)


def _compute_metrics(runs: list[EvalRun]) -> dict[str, float | None]:
    raw: dict[str, float | None] = {
        "recall@5": recall_at_k(runs, k=5),
        "citation_accuracy@3": citation_accuracy(runs),
        "document_recall": document_recall(runs),
        "passage_support_rate": passage_support_rate(runs),
        "refusal_correctness": refusal_correctness(runs),
        "hallucination_rate": hallucination_rate(runs),
    }
    return {name: (round(v, 3) if v is not None else None) for name, v in raw.items()}


def summarize(runs: list[EvalRun]) -> dict[str, object]:
    """Exploratory metrics over ALL cases (verified + unverified).

    NOT a trustworthy quality signal on its own — an unverified case's
    expected_* is a guess, and this blends guesses in with confirmed
    ground truth. Always read alongside summarize_verified(); never quote
    this dict's metric values as "the baseline".
    """
    verified_count = sum(1 for r in runs if r.case.verified)
    return {
        **_compute_metrics(runs),
        "n_cases": len(runs),
        "n_verified_cases": verified_count,
        "n_unverified_cases": len(runs) - verified_count,
        "verified_fraction": round(verified_count / len(runs), 3) if runs else 0.0,
    }


def summarize_verified(runs: list[EvalRun]) -> dict[str, object] | None:
    """The trustworthy baseline — metrics computed ONLY over verified
    cases. Returns None (never a zero-filled dict) when there are no
    verified cases, so callers report "blocked", not a fabricated score.
    """
    verified_runs = [r for r in runs if r.case.verified]
    if not verified_runs:
        return None
    return {**_compute_metrics(verified_runs), "n_cases": len(verified_runs)}


def summarize_by_category(runs: list[EvalRun]) -> dict[str, dict[str, object]]:
    """Same metrics as :func:`summarize` (exploratory, all cases),
    grouped by ``EvalCase.category``. Cases with no category fall under
    ``"uncategorized"``.
    """
    by_category: dict[str, list[EvalRun]] = {}
    for r in runs:
        by_category.setdefault(r.case.category or "uncategorized", []).append(r)
    return {cat: summarize(cat_runs) for cat, cat_runs in by_category.items()}
