#!/usr/bin/env bash
# =============================================================================
# Sync the uploads volume to versioned object storage.
#
# One-way upload only — we never `--delete` on the remote, so a local rm -rf
# can't propagate. Object-storage versioning is the safety net for legitimate
# deletions.
#
# Usage:  ./scripts/backup_uploads.sh
# Env vars:
#   UPLOAD_DIR              (default: /app/data/uploads)
#   BACKUP_S3_BUCKET        (required) — s3://bucket/path destination
# =============================================================================
set -euo pipefail

UPLOAD_DIR="${UPLOAD_DIR:-/app/data/uploads}"
: "${BACKUP_S3_BUCKET:?BACKUP_S3_BUCKET is required (s3://… URI)}"

if [ ! -d "$UPLOAD_DIR" ]; then
    echo "==> upload dir $UPLOAD_DIR does not exist — nothing to back up."
    exit 0
fi

echo "==> Syncing $UPLOAD_DIR → $BACKUP_S3_BUCKET/uploads/"
aws s3 sync \
    "$UPLOAD_DIR" \
    "$BACKUP_S3_BUCKET/uploads/" \
    --exact-timestamps \
    --only-show-errors

echo "==> Done."
