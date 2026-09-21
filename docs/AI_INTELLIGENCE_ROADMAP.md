# LegaLite AI Intelligence Roadmap

_As of 2026-09-20._

A RAG system's intelligence comes from four compounding levers: **corpus**
(what it can find), **retrieval** (how well it finds it), **reasoning** (what
it does with what it found), and **evaluation** (whether any of the above is
actually improving). LegaLite's retrieval plumbing is architecturally solid —
hybrid vector + keyword search, reranking, tenant isolation, citation
grounding are all built. The two levers still mostly unused are corpus
density and a real evaluation loop, which is why they lead this roadmap.

The recent work — self-hosted Qdrant and the `/legal-docs-upload` bulk tool —
removed the operational bottleneck on corpus growth. That makes this the
right moment to also fix how that growth gets measured, or volume risks
adding noise instead of intelligence.

## Corpus & data pipeline

The bulk-upload page removes the ingestion bottleneck, but volume alone
isn't the lever — structure, coverage, and freshness are.

- **Structure-aware chunking.** `RAG_CHUNK_SIZE=900` / `RAG_CHUNK_OVERLAP=120`
  splits on raw character count, even though every chunk's payload already
  carries `article`, `section`, `clause`, `subsection`. Chunking along those
  structural boundaries instead would stop a chunk from splitting mid-clause
  — exactly the failure seen firsthand during testing, where a truncated
  chunk ("over the intern[al]") led the model to guess "the internet."
- **Metadata enrichment consistency.** `law_type`, `act_number`,
  `effective_year`, `legal_topic` directly power `CaseSimilarityEngine`'s
  topic weighting and filtered retrieval. Worth auditing how consistently
  these are populated at ingest time — sparse metadata quietly degrades both
  without ever showing up as an obvious bug.
- **Superseded-law risk.** There's no visible field marking an amended Act's
  prior text as superseded. A corpus holding both old and new versions of the
  same law risks citing repealed provisions as current. An `effective_until`
  / `superseded_by` field is worth adding before the corpus scales past
  manual review.
- **Coverage strategy, not just coverage volume.** A `GROUP BY doc_type` on
  `legal_documents` today would show which practice areas are thin. Once
  evaluation (below) exists, prioritize ingestion against real query
  failures rather than uploading opportunistically.

## Retrieval quality

- **Embedding model.** `text-embedding-3-small` (1536-dim) is the current
  default. `text-embedding-3-large` captures more semantic nuance — material
  for legal language, where near-synonyms ("shall" vs "may") carry real
  distinctions. Trade-off: re-embedding the whole corpus and roughly 5x the
  per-call cost, so this should be A/B-tested against the golden set
  (Evaluation, below) before committing.
- **Hybrid fusion tuning.** Reciprocal Rank Fusion blends Qdrant vector
  search with Postgres full-text search. Legal research is unusually
  keyword-heavy — statute numbers, exact case names, section citations — so
  `HYBRID_TOPK_VECTOR`, `HYBRID_TOPK_KEYWORD`, and `RRF_K` deserve tuning
  against real query logs rather than left at their defaults.
- **Confirm the reranker is live.** `RERANK_PROVIDER` is configurable to
  `cohere` or `none`. A cross-encoder rerank pass is typically the single
  highest-ROI retrieval improvement after the embedding model itself — worth
  confirming it's actually enabled in production, not just wired in.
- **Query understanding.** `QUERY_PLANNER_LLM_FALLBACK_ENABLED=false` by
  default — heuristic-only query planning. Multi-part questions ("what's the
  penalty under the Act, and has any case narrowed it?") likely need
  decomposition into sub-questions before retrieval. Enabling the LLM
  fallback, or adding query expansion (HyDE — embedding a hypothetical
  answer instead of the raw question), targets exactly that gap.

## Reasoning & generation quality

- **Model choice by query difficulty.** Config defaults to `gpt-4o-mini`;
  production currently runs `gpt-4o`. Mini is fine for single-citation
  lookups; genuinely multi-hop reasoning — reconciling a statute with a case
  that reinterprets it — benefits from a stronger reasoning-tier model.
  Worth split-testing on hard queries specifically rather than switching all
  traffic, since cost scales with it.
- **Use citation chains proactively.** `multi_document_reasoner.py` and
  citation-chain resolution (`CITATION_RESOLUTION_THRESHOLD`,
  `CITATION_RESOLUTION_CHAIN_DEPTH=4`) already exist. The deeper lever is
  using them before being asked: when a statute chunk is retrieved, also
  pull cases already tracked in `case_citations` that cite it, instead of
  waiting for a follow-up question.
- **Push toward explicit legal reasoning structure.** The answer schema
  already carries `reasoning_summary` and a `structured_answer` (direct
  answer, applicable law, legal reasoning). Tightening this toward an
  explicit issue → rule → application → conclusion structure — the standard
  form of legal reasoning — would make answers both more rigorous and
  easier to audit.
- **Calibrate the confidence threshold.** `RAG_MIN_CONFIDENCE_SCORE=0.35`
  gates refusal today, almost certainly as a guess rather than a measured
  value. It should be tuned against the golden set below — too low risks
  confident hallucination, too high produces an unhelpfully cautious
  assistant.

## Evaluation & feedback loops

This is the highest-leverage gap: without it, every change proposed above
is a guess.

- **The current eval isn't authoritative.** `evaluation/tenant_isolation/runner.py`
  simulates Qdrant filtering in Python rather than exercising the live
  service — fine for isolation logic, but it says nothing about answer
  quality.
- **Build a golden set.** A few hundred real legal questions with
  expert-verified correct citations and answers, run automatically against
  the live pipeline whenever the corpus, prompts, or model change. This is
  what turns retrieval precision, answer correctness, and confidence
  calibration from opinions into numbers that move over time.
- **Fix dead telemetry first.** `hallucination_floor`, `postgres_failures`,
  `rerank_latency`, and `embedding_latency` currently have zero call sites,
  and `qdrant_failures{operation}` only increments for search. There's no
  visibility today into where retrieval is actually failing in production —
  wiring these up is a prerequisite to prioritizing anything else here.
- **Mine user feedback.** A feedback endpoint already exists client-side.
  Surfacing low-rated or low-confidence answers by recurring pattern — a
  practice area with no good matches, a chunk too short to be useful — turns
  real usage into a prioritized ingestion and tuning backlog instead of
  guesswork.

## Trust & guardrails as intelligence

In a legal-advice context, a wrong answer stated confidently is worse than a
refusal — so guardrail quality is intelligence here, not a tax on it.

- **Add a self-verification pass.** Citation grounding already exists
  (every claim ties to a `document_id`/`chunk_id`). The next step is having
  the model — or a second, cheaper model — check each claim against its
  cited passage before returning, catching cases where retrieval found the
  right document but the answer still overstated what it says.
- **Measure refusal calibration, don't assume it.** Track false-refusal rate
  (a good answer existed but was refused) alongside hallucination rate. Both
  are only visible once the golden set above exists.
- **Keep tenant isolation as a standing gate.** `build_tenant_filter` already
  prevents one firm's private documents from leaking into another firm's
  answer — a severe failure if it ever broke. The integration suite covering
  this against live Qdrant should run on every retrieval-path change, not
  just once.

## Prioritized roadmap

Ordered by expected impact per unit of effort. Evaluation leads because
everything after it becomes measurable instead of guessed.

| Priority | Action | Lever | Effort |
| --- | --- | --- | --- |
| 1 | Wire up dead telemetry (rerank/embedding latency, hallucination floor) | Evaluation | Low |
| 2 | Build a 100–200 question golden set with expert-verified answers | Evaluation | Medium |
| 3 | Confirm the reranker is enabled in production | Retrieval | Low |
| 4 | Structure-aware chunking on article/section/clause boundaries | Corpus | Medium |
| 5 | Tune RRF/topk fusion weights and confidence threshold against the golden set | Retrieval, Reasoning | Medium |
| 6 | Add a superseded-law field; audit for stale duplicate law versions | Corpus | Medium |
| 7 | A/B `text-embedding-3-large` before committing to a full re-embed | Retrieval | Medium–High |
| 8 | Proactive citation-chain surfacing alongside retrieved statutes | Reasoning | Medium |
| 9 | Answer self-verification pass against cited passages | Trust | Medium–High |
| 10 | Coverage-gap-driven ingestion, fed by eval failures and the bulk-upload tool | Corpus | Ongoing |
| 11 | Split-test a stronger reasoning model on hard multi-hop queries only | Reasoning | Medium |

The first four are the near-term set: cheap, and every one of them either
unlocks measurement or fixes a failure mode already seen directly (the
truncated-chunk hallucination, an unconfirmed reranker).
