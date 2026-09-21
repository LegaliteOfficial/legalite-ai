"""Evaluation runner.

Usage:
    python -m evaluation.runner --dataset evaluation/datasets/sample.json

Loads a JSON dataset, runs each question through QAService against the live
corpus (real Qdrant + Postgres + OpenAI — this is not a simulation), and
prints:

1. Case counts — total / verified / unverified.
2. Exploratory metrics over ALL cases (labelled as such — not trustworthy
   on its own, since unverified expected_* values are guesses).
3. The VERIFIED baseline — the same metrics restricted to verified cases
   only. This is the only section that should ever be called "the
   baseline". Printed as BLOCKED, not a fabricated score, when there are
   zero verified cases.
4. A per-category breakdown (exploratory, all cases).
5. The production configuration this run actually exercised, so a later
   run is comparable.

Also writes a machine-readable JSON report to evaluation/results/ (gitignored
— see evaluation/datasets/README.md before committing a specific one on
purpose). Contains metric numbers and configuration, not raw question/answer
content, so it's safe to keep around locally even though it's not tracked.

Requires: OPENAI_API_KEY, a reachable Qdrant (QDRANT_URL/QDRANT_API_KEY), and
a reachable Postgres (DATABASE_URL) with the corpus you want to evaluate
against already ingested. Expected runtime: a few seconds per case (one live
LLM call each).

See evaluation/datasets/README.md for the case schema and the legal-review
workflow that turns a case from a placeholder into ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.rag.reranker import CohereReranker, build_reranker
from app.services.qa_service import QAService
from evaluation.metrics import (
    METRIC_ORDER,
    EvalCase,
    EvalRun,
    summarize,
    summarize_by_category,
    summarize_verified,
)


def load_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text())
    return [EvalCase(**row) for row in data]


def current_configuration() -> dict[str, object]:
    """Snapshot of the production configuration this run actually
    exercised — without this, a later run's numbers aren't comparable to
    this one. Values only, no secrets (API keys are never read here)."""
    settings = get_settings()
    reranker = build_reranker()
    return {
        "embedding_model": settings.openai_embedding_model,
        "generation_model": settings.openai_llm_model,
        "reranker_configured_provider": settings.rerank_provider,
        "reranker_active_implementation": type(reranker).__name__,
        "reranker_enabled": isinstance(reranker, CohereReranker),
        "rrf_k": settings.rrf_k,
        "hybrid_topk_vector": settings.hybrid_topk_vector,
        "hybrid_topk_keyword": settings.hybrid_topk_keyword,
        "hybrid_topk_fused": settings.hybrid_topk_fused,
        "rag_topk_final": settings.rag_topk_final,
        "rag_max_chunk_chars": settings.rag_max_chunk_chars,
        "rag_chunk_overlap_chars": settings.rag_chunk_overlap_chars,
        "rag_catastrophic_cosine_threshold": settings.rag_catastrophic_cosine_threshold,
        "rag_catastrophic_rerank_threshold": settings.rag_catastrophic_rerank_threshold,
        "rag_confidence_high_threshold": settings.rag_confidence_high_threshold,
        "rag_confidence_medium_threshold": settings.rag_confidence_medium_threshold,
        "rag_citation_consistency_high_threshold": settings.rag_citation_consistency_high_threshold,
    }


def _fmt(v: object) -> str:
    return "N/A" if v is None else str(v)


def run(dataset_path: Path) -> dict[str, object]:
    cases = load_cases(dataset_path)
    runs: list[EvalRun] = []
    db = SessionLocal()
    try:
        service = QAService(db)
        for case in cases:
            response = service.ask(case.question)
            runs.append(EvalRun(case=case, response=response))
            print(
                f"[{case.id}] verified={case.verified} confidence={response.confidence} "
                f"citations={len(response.citations)}"
            )
    finally:
        db.close()

    verified_n = sum(1 for c in cases if c.verified)
    print("\n=== Case counts ===")
    print(f"  Total cases:       {len(cases)}")
    print(f"  Verified cases:    {verified_n}")
    print(f"  Unverified cases:  {len(cases) - verified_n}")
    print(f"  Verified fraction: {(verified_n / len(cases) * 100) if cases else 0:.1f}%")

    exploratory = summarize(runs)
    print("\n=== ALL CASES (exploratory only — includes unverified guesses) ===")
    for name in METRIC_ORDER:
        print(f"  {name:24s} {_fmt(exploratory[name])}")

    baseline = summarize_verified(runs)
    print("\n=== VERIFIED baseline (trustworthy) ===")
    if baseline is None:
        print("  BLOCKED — zero verified cases. See evaluation/datasets/README.md")
        print("  for the legal-review workflow; a baseline requires at least some")
        print("  cases with verified: true before these numbers mean anything.")
    else:
        for name in METRIC_ORDER:
            print(f"  {name:24s} {_fmt(baseline[name])}")
        print(f"  (n={baseline['n_cases']} verified cases)")

    by_category = summarize_by_category(runs)
    print("\n=== By category (exploratory, all cases) ===")
    for cat, cat_report in sorted(by_category.items()):
        n = cat_report["n_cases"]
        nv = cat_report["n_verified_cases"]
        print(f"  {cat} (n={n}, verified={nv})")
        for name in METRIC_ORDER:
            print(f"    {name:24s} {_fmt(cat_report[name])}")

    config = current_configuration()
    print("\n=== Configuration exercised by this run ===")
    for k, v in config.items():
        print(f"  {k:38s} {v}")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_path),
        "case_counts": {
            "total": len(cases),
            "verified": verified_n,
            "unverified": len(cases) - verified_n,
            "verified_fraction": round(verified_n / len(cases), 3) if cases else 0.0,
        },
        "all_cases_exploratory": exploratory,
        "verified_baseline": baseline,  # null in JSON when blocked — not a fabricated score
        "by_category": by_category,
        "configuration": config,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/sample.json"),
        help="Path to the eval dataset JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Where to write the machine-readable JSON report (gitignored by default)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the report but don't write a JSON file",
    )
    args = parser.parse_args(argv)
    report = run(args.dataset)

    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = args.output_dir / f"{args.dataset.stem}_{ts}.json"
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nJSON report written to {out_path}")

    if report["case_counts"]["verified_fraction"] < 1.0:  # type: ignore[index]
        vf = report["case_counts"]["verified_fraction"]  # type: ignore[index]
        print(
            f"\nWARNING: only {vf:.0%} of cases are expert-verified. The 'ALL CASES' "
            "section above is exploratory only — read the 'VERIFIED baseline' "
            "section for anything you intend to treat as a real measurement. "
            "See evaluation/datasets/README.md."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
