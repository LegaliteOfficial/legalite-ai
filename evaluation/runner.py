"""Evaluation runner.

Usage:
    python -m evaluation.runner --dataset evaluation/datasets/sample.json

Loads a JSON dataset, runs each question through QAService against the live
corpus, and prints a metric report. Designed to plug into CI once a curated
Ghana-law dataset is in place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.db.session import SessionLocal
from app.services.qa_service import QAService
from evaluation.metrics import EvalCase, EvalRun, summarize


def load_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text())
    return [EvalCase(**row) for row in data]


def run(dataset_path: Path) -> dict[str, float]:
    cases = load_cases(dataset_path)
    runs: list[EvalRun] = []
    db = SessionLocal()
    try:
        service = QAService(db)
        for case in cases:
            response = service.ask(case.question)
            runs.append(EvalRun(case=case, response=response))
            print(
                f"[{case.id}] confidence={response.confidence} "
                f"citations={len(response.citations)}"
            )
    finally:
        db.close()

    report = summarize(runs)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/sample.json"),
        help="Path to the eval dataset JSON file",
    )
    args = parser.parse_args(argv)
    report = run(args.dataset)
    print("\n=== Evaluation report ===")
    for k, v in report.items():
        print(f"  {k:24s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
