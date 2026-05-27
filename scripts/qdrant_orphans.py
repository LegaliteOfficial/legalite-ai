"""Diagnose Qdrant points whose ``document_id`` no longer matches Postgres.

Common case: you migrated Postgres providers (e.g. Neon → Supabase) but
left the Qdrant collection intact. Every point's payload still references
a ``document_id`` that doesn't exist in the new database, so retrieval
surfaces vectors with no chunk metadata to back them up.

Usage::

    docker compose exec app python -m scripts.qdrant_orphans          # default — read-only summary
    docker compose exec app python -m scripts.qdrant_orphans --json   # machine-readable

Exit code is always 0 — this script never deletes anything. Pair it with
``scripts/qdrant_purge.py`` to act on the diagnosis.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import defaultdict

from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.vector_store.qdrant_client import get_vector_store


def _existing_document_ids(db) -> set[uuid.UUID]:
    rows = db.execute(text("SELECT id FROM legal_documents")).all()
    return {r[0] for r in rows}


def _walk_collection(store) -> tuple[int, dict[str, int]]:
    """Scroll every point in the collection, returning (total, by_document)."""
    by_document: dict[str, int] = defaultdict(int)
    total = 0
    next_offset = None
    while True:
        result, next_offset = store._client.scroll(
            collection_name=store._collection,
            limit=512,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not result:
            break
        for point in result:
            total += 1
            doc_id = (point.payload or {}).get("document_id")
            if isinstance(doc_id, str):
                by_document[doc_id] += 1
        if next_offset is None:
            break
    return total, dict(by_document)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    settings = get_settings()
    db = SessionLocal()
    try:
        live_doc_ids = _existing_document_ids(db)
    finally:
        db.close()

    store = get_vector_store()
    total_points, by_document = _walk_collection(store)

    orphan_doc_ids: list[str] = []
    orphan_point_count = 0
    valid_point_count = 0

    for doc_id_str, count in by_document.items():
        try:
            doc_uuid = uuid.UUID(doc_id_str)
        except ValueError:
            orphan_doc_ids.append(doc_id_str)
            orphan_point_count += count
            continue
        if doc_uuid in live_doc_ids:
            valid_point_count += count
        else:
            orphan_doc_ids.append(doc_id_str)
            orphan_point_count += count

    summary = {
        "qdrant_collection": settings.qdrant_collection,
        "total_points": total_points,
        "distinct_document_ids_in_qdrant": len(by_document),
        "live_documents_in_postgres": len(live_doc_ids),
        "orphan_document_ids": len(orphan_doc_ids),
        "orphan_point_count": orphan_point_count,
        "valid_point_count": valid_point_count,
        "orphan_ratio": (
            orphan_point_count / total_points if total_points else 0.0
        ),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print("Qdrant ↔ Postgres reconciliation summary")
    print("-" * 50)
    print(f"  collection                       : {summary['qdrant_collection']}")
    print(f"  total points in Qdrant           : {summary['total_points']}")
    print(f"  distinct document_ids in Qdrant  : {summary['distinct_document_ids_in_qdrant']}")
    print(f"  documents present in Postgres    : {summary['live_documents_in_postgres']}")
    print(f"  ORPHAN document_ids              : {summary['orphan_document_ids']}")
    print(f"  orphan point count               : {summary['orphan_point_count']}")
    print(f"  valid point count                : {summary['valid_point_count']}")
    print(f"  orphan ratio                     : {summary['orphan_ratio']:.1%}")
    if summary["orphan_document_ids"]:
        print()
        print("To delete the orphaned points, run:")
        print("  python -m scripts.qdrant_purge --confirm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
