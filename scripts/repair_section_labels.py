"""Fix section / article labels on chunks already in the library.

Replays the chunker's section guard (``app/ingestion/section_repair.py``)
over each document's stored chunks and corrects the labels in Postgres
(``legal_chunks``) and in the Qdrant payload, which is what retrieval and
citations read. Chunk text, boundaries and vectors are not touched.

Usage::

    python -m scripts.repair_section_labels                 # dry run: report only
    python -m scripts.repair_section_labels --show 20       # dry run with examples
    python -m scripts.repair_section_labels --apply         # write the changes
    python -m scripts.repair_section_labels --document <uuid> [--apply]

The dry run never writes. ``--apply`` handles each document separately:
Qdrant is updated first and Postgres committed after, so an interrupted
run can simply be re-run (already-correct chunks produce no change).
Chunks whose point is missing from Qdrant are fixed in Postgres only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

from sqlalchemy import text

from app.db.session import SessionLocal
from app.ingestion.section_repair import LABELS, StoredChunk, relabel
from app.vector_store.qdrant_client import get_vector_store


def _documents(db, only: str | None) -> list[dict]:  # type: ignore[no-untyped-def]
    sql = "SELECT id, title, doc_type, law_type, page_count FROM legal_documents"
    params: dict[str, str] = {}
    if only:
        sql += " WHERE id = :id"
        params["id"] = only
    return [dict(r._mapping) for r in db.execute(text(sql + " ORDER BY title"), params)]


def _chunks(db, document_id) -> list[StoredChunk]:  # type: ignore[no-untyped-def]
    rows = db.execute(
        text(
            "SELECT id, vector_id, page_start, start_char, content, "
            "article, section, clause, subsection "
            "FROM legal_chunks WHERE document_id = :d"
        ),
        {"d": document_id},
    )
    return [StoredChunk(**dict(r._mapping)) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the corrections (default: dry run)"
    )
    parser.add_argument("--document", help="only this legal_documents.id")
    parser.add_argument("--show", type=int, default=0, help="print up to N example changes")
    parser.add_argument("--json", action="store_true", help="machine-readable summary")
    args = parser.parse_args()

    db = SessionLocal()
    store = get_vector_store() if args.apply else None
    summary: dict[str, object] = {"mode": "apply" if args.apply else "dry-run"}
    per_doc: list[dict[str, object]] = []
    field_counts: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    total_chunks = 0
    qdrant_updated = 0
    missing_points = 0
    try:
        for doc in _documents(db, args.document):
            chunks = _chunks(db, doc["id"])
            total_chunks += len(chunks)
            changes = relabel(
                chunks,
                doc_type=doc["doc_type"],
                law_type=doc["law_type"],
                total_pages=doc["page_count"] or None,
                title=doc["title"],
            )
            if not changes:
                continue
            for ch in changes:
                field_counts.update(ch.new.keys())
                if len(examples) < args.show:
                    examples.append(
                        {
                            "document": doc["title"],
                            "page": ch.chunk.page_start,
                            "starts": ch.chunk.content.strip()[:70],
                            "change": {k: list(v) for k, v in ch.diff().items()},
                        }
                    )
            from app.ingestion.legal_chunker import _doc_mode

            per_doc.append(
                {
                    "document": doc["title"],
                    "doc_type": doc["doc_type"],
                    "read_as": _doc_mode(doc["doc_type"], doc["law_type"], doc["title"]),
                    "chunks_changed": len(changes),
                }
            )
            if not args.apply:
                continue
            for ch in changes:
                # Column names come from the fixed LABELS tuple, never input.
                sets = ", ".join(f"{k} = :{k}" for k in ch.new if k in LABELS)
                sql = f"UPDATE legal_chunks SET {sets} WHERE id = :id"  # noqa: S608
                db.execute(text(sql), {**ch.new, "id": ch.chunk.id})
            # Postgres commits only after Qdrant succeeds: a failed run then
            # leaves Postgres unchanged, and a re-run redoes both.
            assert store is not None
            # Only points that exist: Qdrant rejects a payload update for a
            # missing id, which would stop the run after Postgres committed.
            wanted = [str(ch.chunk.vector_id) for ch in changes if ch.chunk.vector_id]
            present: set[str] = set()
            for start in range(0, len(wanted), 500):
                present.update(
                    str(p.id)
                    for p in store._client.retrieve(
                        collection_name=store._collection,
                        ids=wanted[start : start + 500],
                        with_payload=False,
                        with_vectors=False,
                    )
                )
            missing_points += len(set(wanted) - present)
            # Group points by identical new payload to keep Qdrant calls few.
            groups: dict[str, list[str]] = defaultdict(list)
            for ch in changes:
                if ch.chunk.vector_id and str(ch.chunk.vector_id) in present:
                    groups[json.dumps(ch.new, sort_keys=True)].append(str(ch.chunk.vector_id))
                    qdrant_updated += 1
            for payload_json, ids in groups.items():
                for start in range(0, len(ids), 256):
                    store._client.set_payload(
                        collection_name=store._collection,
                        payload=json.loads(payload_json),
                        points=ids[start : start + 256],
                        wait=True,
                    )
            db.commit()
    finally:
        db.close()

    summary.update(
        chunks_scanned=total_chunks,
        documents_changed=len(per_doc),
        chunks_changed=sum(int(d["chunks_changed"]) for d in per_doc),  # type: ignore[call-overload]
        fields_changed={k: field_counts.get(k, 0) for k in LABELS},
        documents=per_doc,
        examples=examples,
    )
    if args.apply:
        summary.update(qdrant_points_updated=qdrant_updated, qdrant_points_missing=missing_points)
    if args.json:
        print(json.dumps(summary, indent=1, default=str))
        return 0
    print(f"{summary['mode']}: {summary['chunks_changed']} of {total_chunks} chunks in "
          f"{summary['documents_changed']} documents need new labels")
    print("fields:", summary["fields_changed"])
    if args.apply:
        print(f"qdrant: {qdrant_updated} points updated, {missing_points} not in the collection")
    for d in sorted(per_doc, key=lambda d: -int(d["chunks_changed"]))[:25]:  # type: ignore[call-overload]
        print(f"  {d['chunks_changed']:>6}  [{d['doc_type']} -> {d['read_as']}] {d['document']}")
    for e in examples:
        print(json.dumps(e, ensure_ascii=False, default=str))
    if not args.apply and per_doc:
        print("\nDry run only. Re-run with --apply to write these changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
