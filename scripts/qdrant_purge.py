"""Delete Qdrant points whose ``document_id`` is not in Postgres.

DESTRUCTIVE. Always run ``scripts/qdrant_orphans.py`` first to inspect
what will be deleted. Requires ``--confirm`` to actually run; without it
this script does a dry-run.

Usage::

    docker compose exec app python -m scripts.qdrant_purge            # dry-run
    docker compose exec app python -m scripts.qdrant_purge --confirm  # actually delete

Optional ``--recreate`` drops the entire collection and rebuilds it
empty (with the correct payload indexes). Use this when orphans
outnumber live points 10:1 and you'd rather start fresh — much faster
than per-document deletes.

This script is idempotent: re-running after a successful purge finds
nothing to delete and exits cleanly.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections import defaultdict

from qdrant_client.http import models as qm
from sqlalchemy import text

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.vector_store.qdrant_client import get_vector_store

logger = get_logger(__name__)


def _existing_document_ids(db) -> set[uuid.UUID]:
    rows = db.execute(text("SELECT id FROM legal_documents")).all()
    return {r[0] for r in rows}


def _orphan_document_ids(store, live: set[uuid.UUID]) -> list[str]:
    """Scroll the collection, return the document_id strings not in `live`."""
    seen: dict[str, int] = defaultdict(int)
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
            doc_id = (point.payload or {}).get("document_id")
            if isinstance(doc_id, str):
                seen[doc_id] += 1
        if next_offset is None:
            break

    orphans: list[str] = []
    for doc_id_str, _ in seen.items():
        try:
            if uuid.UUID(doc_id_str) not in live:
                orphans.append(doc_id_str)
        except ValueError:
            orphans.append(doc_id_str)  # malformed → also orphan
    return orphans


def _delete_by_document(store, document_id: str) -> None:
    store._client.delete(
        collection_name=store._collection,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="document_id",
                        match=qm.MatchValue(value=document_id),
                    )
                ]
            )
        ),
        wait=True,
    )


def _recreate_collection(store) -> None:
    """Drop and recreate. Faster than per-doc deletes when orphans dominate.

    Reuses :func:`ensure_collection` so the new collection comes with the
    payload keyword indexes the tenant filter requires (the fix from the
    integration-test bug we caught earlier).
    """
    try:
        store._client.delete_collection(collection_name=store._collection)
    except Exception:
        logger.exception("qdrant.delete_collection_failed")
    store.ensure_collection()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="actually delete (default is dry-run)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop and rebuild the collection instead of per-document deletes",
    )
    args = parser.parse_args()

    settings = get_settings()
    store = get_vector_store()

    if args.recreate:
        print(f"==> Recreating collection {settings.qdrant_collection!r}")
        if not args.confirm:
            print("    DRY RUN — pass --confirm to actually drop + rebuild.")
            return 0
        _recreate_collection(store)
        print("    OK. Collection is empty; re-ingest your PDFs to repopulate.")
        return 0

    db = SessionLocal()
    try:
        live = _existing_document_ids(db)
    finally:
        db.close()
    orphans = _orphan_document_ids(store, live)

    print(f"==> Found {len(orphans)} orphan document_ids in {settings.qdrant_collection!r}")
    if not orphans:
        return 0

    for d in orphans[:10]:
        print(f"    - {d}")
    if len(orphans) > 10:
        print(f"    ... and {len(orphans) - 10} more")

    if not args.confirm:
        print()
        print("DRY RUN — pass --confirm to actually delete.")
        print("Or pass --recreate to drop the entire collection.")
        return 0

    print()
    print("==> Deleting ...")
    for d in orphans:
        _delete_by_document(store, d)
    print(f"    OK. Deleted points for {len(orphans)} document_ids.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
