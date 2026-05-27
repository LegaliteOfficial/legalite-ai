"""Re-ingest every PDF in a directory through the live ingestion pipeline.

The typical use case: you migrated Postgres providers and have a local
folder of PDFs that the new database doesn't know about. Run this from
a machine that can reach the new Postgres + Qdrant (Railway shell,
laptop with the right env vars set, etc.).

Behavior:
- Scans ``--source-dir`` recursively for ``*.pdf``.
- Skips files whose **content size + filename** matches an existing
  ``legal_documents.source_filename`` + ``page_count == 0 OR > 0`` row,
  i.e. previously-ingested files. (See ``--force`` to override.)
- Ingests through ``IngestionService.ingest_pdf`` so chunks, embeddings,
  Qdrant payloads, case-metadata extraction, and citation resolution
  all run the same as a live upload.
- Reports per-file outcome at the end.

Usage::

    docker compose exec app python -m scripts.reingest_directory \\
        --source-dir /app/data/uploads \\
        --scope global \\
        --doc-type "Re-ingested"

    # tenant-private, attribute to an org/user:
    docker compose exec app python -m scripts.reingest_directory \\
        --source-dir /tmp/firm_uploads \\
        --scope tenant_private \\
        --origin internal_memo \\
        --org-id 7f3a4b6c-...  --user-id c91e88f2-...

When the worker container is up, this still runs the inline path because
we call ``IngestionService.ingest_pdf`` directly. To go through Arq, hit
``POST /admin/upload-law`` or ``/documents/private-upload`` via curl.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text

from app.core.tenant_context import TenantContext
from app.db.enums import (
    ConfidentialityLevel,
    DocumentOrigin,
    TenantRole,
    VisibilityScope,
)
from app.db.session import SessionLocal
from app.services.ingestion_service import IngestionService


@dataclass
class _Outcome:
    path: Path
    document_id: uuid.UUID | None
    chunks: int
    skipped: bool
    error: str | None


def _already_ingested(db, filename: str) -> bool:
    row = db.execute(
        text(
            "SELECT 1 FROM legal_documents "
            "WHERE source_filename = :name AND status = 'ingested' "
            "LIMIT 1"
        ),
        {"name": filename},
    ).first()
    return row is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=["global", "tenant_private"],
        default="global",
    )
    parser.add_argument(
        "--origin",
        type=str,
        default=None,
        help="DocumentOrigin enum value (e.g. act, sample_case)",
    )
    parser.add_argument("--doc-type", type=str, default="Re-ingested")
    parser.add_argument("--org-id", type=str, default=None)
    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-ingest even if a row with the same source_filename already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be ingested without actually running",
    )
    args = parser.parse_args()

    source = args.source_dir
    if not source.exists() or not source.is_dir():
        print(f"ERROR: source-dir {source!r} is not a directory", file=sys.stderr)
        return 2

    scope = VisibilityScope(args.scope)
    origin: DocumentOrigin | None = None
    if args.origin:
        try:
            origin = DocumentOrigin(args.origin)
        except ValueError:
            print(f"ERROR: invalid --origin {args.origin!r}", file=sys.stderr)
            return 2

    tenant: TenantContext | None = None
    if scope is VisibilityScope.TENANT_PRIVATE:
        if not args.org_id or not args.user_id:
            print("ERROR: tenant_private scope requires --org-id and --user-id", file=sys.stderr)
            return 2
        tenant = TenantContext(
            organization_id=uuid.UUID(args.org_id),
            user_id=uuid.UUID(args.user_id),
            role=TenantRole.LAWYER,
        )

    pdfs = sorted(source.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found under {source}")
        return 0
    print(f"==> {len(pdfs)} PDF(s) under {source}")

    outcomes: list[_Outcome] = []
    db = SessionLocal()
    try:
        service = IngestionService(db)
        for pdf in pdfs:
            filename = pdf.name
            if not args.force and _already_ingested(db, filename):
                print(f"   SKIP (already ingested): {filename}")
                outcomes.append(_Outcome(pdf, None, 0, True, None))
                continue
            if args.dry_run:
                print(f"   WOULD INGEST: {filename}")
                outcomes.append(_Outcome(pdf, None, 0, True, "dry-run"))
                continue
            try:
                doc_id, chunks = service.ingest_pdf(
                    file_bytes=pdf.read_bytes(),
                    original_filename=filename,
                    title=filename,
                    doc_type=args.doc_type,
                    visibility_scope=scope,
                    document_origin=origin,
                    confidentiality_level=(
                        ConfidentialityLevel.PUBLIC
                        if scope is VisibilityScope.GLOBAL
                        else ConfidentialityLevel.INTERNAL
                    ),
                    tenant=tenant,
                )
                outcomes.append(_Outcome(pdf, doc_id, chunks, False, None))
                print(f"   OK ({chunks} chunks): {filename}  doc_id={doc_id}")
            except Exception as exc:  # noqa: BLE001
                outcomes.append(_Outcome(pdf, None, 0, False, str(exc)[:200]))
                print(f"   FAIL: {filename}  error={exc}")
    finally:
        db.close()

    ingested = sum(1 for o in outcomes if not o.skipped and o.error is None)
    skipped = sum(1 for o in outcomes if o.skipped)
    failed = sum(1 for o in outcomes if o.error and o.error != "dry-run")
    print()
    print(f"==> Summary: {ingested} ingested, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
