"""Corpus health check — Postgres/Qdrant reconciliation, both directions.

``scripts/qdrant_orphans.py`` already answers "does Qdrant have vectors
Postgres doesn't recognize" (walks Qdrant, checks against Postgres). This
script answers the *other* direction, which surfaced as a real Phase 2.5
finding: "does Postgres believe chunks are embedded (``vector_id`` set)
when the point doesn't actually exist in Qdrant" — e.g. after a Qdrant
volume reset, a partial Cloud-to-self-hosted migration, or ingestion
against a since-decommissioned Qdrant instance while Postgres (shared
across environments) kept the row.

Usage::

    docker compose exec app python -m scripts.corpus_health          # text summary
    docker compose exec app python -m scripts.corpus_health --json   # machine-readable

Read-only. Never prints document titles or chunk content — only counts and
ids, so it's safe to run against a real corpus with tenant-private data.
Exit code is always 0; this is a diagnostic, not a gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.vector_store.qdrant_client import get_vector_store

_RETRIEVE_BATCH = 500

logger = logging.getLogger(__name__)


def _postgres_counts(db) -> dict[str, object]:
    doc_total = db.execute(text("SELECT COUNT(*) FROM legal_documents")).scalar()
    doc_by_status = dict(
        db.execute(text("SELECT status, COUNT(*) FROM legal_documents GROUP BY status")).all()
    )
    doc_by_law_type = {
        (k or "(null)"): v
        for k, v in db.execute(
            text("SELECT law_type, COUNT(*) FROM legal_documents GROUP BY law_type")
        ).all()
    }
    doc_by_visibility = dict(
        db.execute(
            text("SELECT visibility_scope, COUNT(*) FROM legal_documents GROUP BY visibility_scope")
        ).all()
    )
    doc_missing_law_type = db.execute(
        text("SELECT COUNT(*) FROM legal_documents WHERE law_type IS NULL")
    ).scalar()
    doc_missing_act_number = db.execute(
        text("SELECT COUNT(*) FROM legal_documents WHERE act_number IS NULL")
    ).scalar()
    doc_missing_effective_year = db.execute(
        text("SELECT COUNT(*) FROM legal_documents WHERE effective_year IS NULL")
    ).scalar()

    chunk_total = db.execute(text("SELECT COUNT(*) FROM legal_chunks")).scalar()
    chunk_with_vector = db.execute(
        text("SELECT COUNT(*) FROM legal_chunks WHERE vector_id IS NOT NULL")
    ).scalar()
    chunk_without_vector = chunk_total - chunk_with_vector

    # Tenant distribution — counts only, never titles/content. NULL
    # organization_id means GLOBAL, not "unknown".
    tenant_distribution = dict(
        db.execute(
            text(
                "SELECT COALESCE(organization_id::text, '(global)'), COUNT(*) "
                "FROM legal_documents GROUP BY organization_id"
            )
        ).all()
    )

    return {
        "documents_total": doc_total,
        "documents_by_status": doc_by_status,
        "documents_by_law_type": doc_by_law_type,
        "documents_by_visibility_scope": doc_by_visibility,
        "documents_missing_law_type": doc_missing_law_type,
        "documents_missing_act_number": doc_missing_act_number,
        "documents_missing_effective_year": doc_missing_effective_year,
        "documents_by_tenant": tenant_distribution,
        "chunks_total": chunk_total,
        "chunks_with_vector_id": chunk_with_vector,
        "chunks_without_vector_id": chunk_without_vector,
    }


def _chunk_vector_ids(db) -> list[str]:
    rows = db.execute(
        text("SELECT vector_id FROM legal_chunks WHERE vector_id IS NOT NULL")
    ).all()
    return [r[0] for r in rows]


def _count_existing_in_qdrant(store, vector_ids: list[str]) -> int:
    """How many of these Postgres-referenced vector_ids actually exist as
    Qdrant points, checked via direct ID lookup (not a full collection
    scroll — cheap even for a large corpus)."""
    found = 0
    for i in range(0, len(vector_ids), _RETRIEVE_BATCH):
        batch = vector_ids[i : i + _RETRIEVE_BATCH]
        try:
            points = store._client.retrieve(
                collection_name=store._collection,
                ids=batch,
                with_payload=False,
                with_vectors=False,
            )
        except Exception:
            logger.exception(
                "corpus_health.qdrant_retrieve_batch_failed", extra={"batch_size": len(batch)}
            )
            continue
        found += len(points)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    settings = get_settings()
    db = SessionLocal()
    try:
        pg = _postgres_counts(db)
        vector_ids = _chunk_vector_ids(db)
    finally:
        db.close()

    store = get_vector_store()
    try:
        qdrant_point_count = store._client.get_collection(store._collection).points_count
    except Exception:  # noqa: BLE001
        qdrant_point_count = None

    existing_in_qdrant = _count_existing_in_qdrant(store, vector_ids)
    missing_in_qdrant = len(vector_ids) - existing_in_qdrant

    if pg["documents_total"] == 0:
        corpus_state = "EMPTY"
    elif missing_in_qdrant == 0:
        corpus_state = "POPULATED"
    elif existing_in_qdrant == 0:
        corpus_state = "TEST-ONLY (Postgres has documents, none are actually retrievable)"
    else:
        corpus_state = "PARTIALLY POPULATED (some chunks reference vectors that no longer exist)"

    summary = {
        "postgres": pg,
        "qdrant_collection": settings.qdrant_collection,
        "qdrant_reported_point_count": qdrant_point_count,
        "chunks_referencing_a_vector_id": len(vector_ids),
        "of_those_actually_present_in_qdrant": existing_in_qdrant,
        "of_those_missing_from_qdrant": missing_in_qdrant,
        "corpus_state": corpus_state,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print("Corpus health")
    print("-" * 60)
    print(f"  Documents (total)              : {pg['documents_total']}")
    print(f"  Documents by status             : {pg['documents_by_status']}")
    print(f"  Documents by visibility_scope   : {pg['documents_by_visibility_scope']}")
    print(f"  Documents missing law_type      : {pg['documents_missing_law_type']}")
    print(f"  Documents missing act_number    : {pg['documents_missing_act_number']}")
    print(f"  Documents missing effective_year: {pg['documents_missing_effective_year']}")
    print(f"  Documents by tenant (counts only): {pg['documents_by_tenant']}")
    print()
    print(f"  Chunks (total)                  : {pg['chunks_total']}")
    print(f"  Chunks with vector_id set       : {pg['chunks_with_vector_id']}")
    print(f"  Chunks with NO vector_id        : {pg['chunks_without_vector_id']}")
    print()
    print(f"  Qdrant collection               : {summary['qdrant_collection']}")
    print(f"  Qdrant reported point_count     : {summary['qdrant_reported_point_count']}")
    print(f"  Chunks referencing a vector_id  : {summary['chunks_referencing_a_vector_id']}")
    print(f"  ...actually present in Qdrant   : {summary['of_those_actually_present_in_qdrant']}")
    print(f"  ...MISSING from Qdrant          : {summary['of_those_missing_from_qdrant']}")
    print()
    print(f"  CORPUS STATE: {corpus_state}")
    if missing_in_qdrant > 0:
        print()
        print(f"  {missing_in_qdrant} chunks have a vector_id Postgres trusts but that")
        print("  point doesn't exist in this Qdrant instance — those documents are")
        print("  NOT actually searchable through /ask right now, despite showing")
        print("  status='ingested'. Common cause: ingested against a different")
        print("  Qdrant instance (e.g. the old Qdrant Cloud cluster, or a since-reset")
        print("  local volume) than the one currently configured. Re-run ingestion")
        print("  for those documents, or complete the Cloud snapshot migration —")
        print("  see docs/QDRANT_SELF_HOSTED.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
