#!/usr/bin/env bash
# =============================================================================
# Nightly Postgres logical dump → encrypted upload to S3.
#
# Run this OUTSIDE the app container (e.g. from CI, ops host, or a cron job)
# so a compromised app can't tamper with the backup.
#
# Usage:  ./scripts/backup_postgres.sh
# Env vars:
#   DATABASE_URL            (required) — postgres connection string
#   BACKUP_S3_BUCKET        (required) — s3://bucket/path destination
#   BACKUP_GPG_RECIPIENT    (required) — GPG key id / email for encryption
#
# Tables dumped (legalite-ai-owned):
#   legal_documents, legal_chunks, legal_cases, case_parties, case_citations,
#   chat_sessions, chat_messages, ingestion_jobs
#
# legalite-backend's tables (users / organizations / …) are backed up by
# the backend's own runbook.
# =============================================================================
set -euo pipefail

: "${DATABASE_URL:?DATABASE_URL is required}"
: "${BACKUP_S3_BUCKET:?BACKUP_S3_BUCKET is required}"
: "${BACKUP_GPG_RECIPIENT:?BACKUP_GPG_RECIPIENT is required (gpg key id)}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
TABLES=(
    legal_documents
    legal_chunks
    legal_cases
    case_parties
    case_citations
    chat_sessions
    chat_messages
    ingestion_jobs
)
TABLE_FLAGS=()
for t in "${TABLES[@]}"; do
    TABLE_FLAGS+=("--table=public.$t")
done

WORK_DIR="$(mktemp -d -t legalite_pg_backup_XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT
DUMP="$WORK_DIR/legalite_ai_${TS}.sql.gz.gpg"

echo "==> pg_dump → gzip → gpg → s3"
pg_dump \
    --no-owner \
    --no-acl \
    --format=plain \
    "${TABLE_FLAGS[@]}" \
    "$DATABASE_URL" \
    | gzip \
    | gpg --batch --yes --encrypt --recipient "$BACKUP_GPG_RECIPIENT" \
    > "$DUMP"

SIZE="$(stat -f%z "$DUMP" 2>/dev/null || stat -c%s "$DUMP")"
SHA="$(shasum -a 256 "$DUMP" | awk '{print $1}')"
echo "   bytes: $SIZE  sha256: $SHA"

aws s3 cp "$DUMP" "$BACKUP_S3_BUCKET/postgres/$(basename "$DUMP")" --only-show-errors

# Append to a manifest so the restore tool knows what exists.
MANIFEST="$WORK_DIR/manifest.jsonl"
printf '{"ts":"%s","filename":"%s","size":%s,"sha256":"%s","tables":%s}\n' \
    "$TS" "$(basename "$DUMP")" "$SIZE" "$SHA" \
    "$(printf '%s\n' "${TABLES[@]}" | python3 -c 'import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))')" \
    > "$MANIFEST"
aws s3 cp "$MANIFEST" "$BACKUP_S3_BUCKET/postgres/manifest.jsonl" \
    --content-type application/x-ndjson \
    --only-show-errors

echo "==> Done."
