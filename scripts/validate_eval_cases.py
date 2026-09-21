"""Corpus/evaluation alignment check — a validation tool, not a verifier.

For every case in an evaluation dataset, checks whether its
`expected_law` / `expected_citations` / `expected_documents` /
`expected_passages` actually correspond to something in the real corpus
(Postgres). This answers "is this case even checkable" before a legal
reviewer spends time on it — it does NOT answer "is the expected answer
legally correct", and it must never be used to set `verified: true`
automatically. Only a human with legal-domain knowledge does that (see
evaluation/datasets/README.md).

Usage::

    docker compose exec app python -m scripts.validate_eval_cases \\
        --dataset evaluation/datasets/sample.json

Read-only. Exit code is always 0 — this is a diagnostic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import text

from app.db.session import SessionLocal
from evaluation.metrics import EvalCase, expected_citation_list
from evaluation.runner import load_cases


def _document_exists(db, law_name: str) -> bool:
    if not law_name.strip():
        return False
    row = db.execute(
        text("SELECT 1 FROM legal_documents WHERE title ILIKE :pattern LIMIT 1"),
        {"pattern": f"%{law_name.strip()}%"},
    ).fetchone()
    return row is not None


def _passage_exists(db, substring: str) -> bool:
    if not substring.strip():
        return False
    row = db.execute(
        text("SELECT 1 FROM legal_chunks WHERE content ILIKE :pattern LIMIT 1"),
        {"pattern": f"%{substring.strip()}%"},
    ).fetchone()
    return row is not None


def validate_case(db, case: EvalCase) -> list[str]:
    """Returns a list of "✓/✗ <check>" lines for one case. Empty list
    means the case has nothing checkable yet (a pure template)."""
    lines: list[str] = []

    citations = expected_citation_list(case)
    if citations:
        for c in citations:
            law = (c.get("law") or "").strip()
            if not law:
                continue
            ok = _document_exists(db, law)
            mark = "✓" if ok else "✗"
            lines.append(f"{mark} expected citation document exists: {law!r}")

    for doc_title in case.expected_documents:
        ok = _document_exists(db, doc_title)
        mark = "✓" if ok else "✗"
        lines.append(f"{mark} expected document exists: {doc_title!r}")

    for passage in case.expected_passages:
        ok = _passage_exists(db, passage)
        mark = "✓" if ok else "✗"
        lines.append(f"{mark} expected passage exists: {passage[:60]!r}")

    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/sample.json"),
    )
    args = parser.parse_args(argv)

    cases = load_cases(args.dataset)
    db = SessionLocal()
    checkable = 0
    fully_ok = 0
    try:
        for case in cases:
            lines = validate_case(db, case)
            print(f"\n{case.id} (verified={case.verified})")
            if not lines:
                print("  (template — nothing to check yet)")
                continue
            checkable += 1
            if all(line.startswith("✓") for line in lines):
                fully_ok += 1
            for line in lines:
                print(f"  {line}")
    finally:
        db.close()

    print(f"\n{'-' * 50}")
    print(f"Cases with checkable fields: {checkable}/{len(cases)}")
    print(f"Cases where every checkable field matched: {fully_ok}/{checkable if checkable else 0}")
    print(
        "\nThis tool confirms a document/passage EXISTS in the corpus — it says"
        "\nnothing about whether the expected answer is legally correct. Setting"
        "\nverified: true is always a human legal-review decision, never automatic."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
