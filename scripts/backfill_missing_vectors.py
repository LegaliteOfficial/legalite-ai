"""Re-create Qdrant points that Postgres references but the collection lacks.

After moving Qdrant (e.g. restoring an older snapshot into a new instance),
some ``legal_chunks`` rows can point at a ``vector_id`` the collection no
longer holds. Those chunks are invisible to vector search. This re-embeds
just those chunks from the text stored in Postgres and writes each point
under its existing ``vector_id``, so the Postgres row stays linked and
nothing else changes. Payloads are built from the Postgres row and its
document, the same fields ingestion writes.

Usage::

    python -m scripts.backfill_missing_vectors              # dry run: report only
    python -m scripts.backfill_missing_vectors --apply      # embed + upsert
    python -m scripts.backfill_missing_vectors --document <uuid> [--apply]

The dry run never writes and never calls the embedding API. ``--apply``
works one document at a time; re-running it skips points that now exist.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from qdrant_client.http import models as qm
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.vector_store.qdrant_client import VectorPayload, get_vector_store

_SQL = """
SELECT c.id, c.document_id, c.vector_id, c.title, c.chapter, c.part, c.article,
       c.section, c.clause, c.subsection, c.page_start, c.page_end, c.content,
       c.law_type, c.act_number, c.effective_year, c.legal_topic,
       c.organization_id, c.visibility_scope, c.document_origin,
       c.start_char, c.end_char,
       d.title AS doc_title, d.doc_type, d.confidentiality_level
FROM legal_chunks c JOIN legal_documents d ON d.id = c.document_id
WHERE c.vector_id IS NOT NULL
"""


def _str(v: object) -> str | None:
    return None if v is None else str(v)


def _payload(r) -> VectorPayload:  # type: ignore[no-untyped-def]
    return VectorPayload(
        document_id=str(r.document_id),
        chunk_id=str(r.id),
        law=r.doc_title,
        doc_type=r.doc_type,
        title=r.title,
        chapter=r.chapter,
        part=r.part,
        article=r.article,
        section=r.section,
        clause=r.clause,
        subsection=r.subsection,
        page_start=r.page_start,
        page_end=r.page_end,
        content=r.content,
        law_type=r.law_type,
        act_number=r.act_number,
        effective_year=r.effective_year,
        legal_topic=r.legal_topic,
        organization_id=_str(r.organization_id),
        visibility_scope=_str(r.visibility_scope) or "global",
        document_origin=_str(r.document_origin),
        confidentiality_level=_str(r.confidentiality_level),
        start_char=r.start_char,
        end_char=r.end_char,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--apply", action="store_true", help="embed and upsert (default: dry run)")
    parser.add_argument("--document", help="only this legal_documents.id")
    args = parser.parse_args()

    settings = get_settings()
    store = get_vector_store()
    db = SessionLocal()
    sql = _SQL + (" AND c.document_id = :d" if args.document else "")
    rows = db.execute(text(sql), {"d": args.document} if args.document else {}).all()
    db.close()

    present: set[str] = set()
    ids = [str(r.vector_id) for r in rows]
    for start in range(0, len(ids), 500):
        present.update(
            str(p.id)
            for p in store._client.retrieve(
                collection_name=store._collection,
                ids=ids[start : start + 500],
                with_payload=False,
                with_vectors=False,
            )
        )
    missing = [r for r in rows if str(r.vector_id) not in present]
    by_doc: dict[str, list] = {}  # type: ignore[type-arg]
    for r in missing:
        by_doc.setdefault(r.doc_title or str(r.document_id), []).append(r)
    chars = sum(len(r.content or "") for r in missing)

    print(
        f"{len(missing)} of {len(rows)} chunks have no point in '{store._collection}', "
        f"in {len(by_doc)} documents (~{chars // 4:,} tokens to embed "
        f"with {settings.openai_embedding_model})"
    )
    for title, rs in sorted(by_doc.items(), key=lambda kv: -len(kv[1]))[:30]:
        print(f"  {len(rs):>6}  {title}")
    if not args.apply:
        if missing:
            print("\nDry run only. Re-run with --apply to embed and upsert these chunks.")
        return 0

    from app.ingestion.embedding_service import EmbeddingService

    embedder = EmbeddingService()
    done: Counter[str] = Counter()
    for title, rs in by_doc.items():
        rs = [r for r in rs if (r.content or "").strip()]
        if not rs:
            continue
        vectors = embedder.embed_documents([r.content for r in rs])
        points = [
            qm.PointStruct(id=str(r.vector_id), vector=v, payload=_payload(r).as_dict())
            for r, v in zip(rs, vectors, strict=True)
        ]
        for start in range(0, len(points), 256):
            store._client.upsert(
                collection_name=store._collection, points=points[start : start + 256], wait=True
            )
        done[title] = len(points)
        print(f"  restored {len(points):>5}  {title}", flush=True)
    print(f"apply: restored {sum(done.values())} points in {len(done)} documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
