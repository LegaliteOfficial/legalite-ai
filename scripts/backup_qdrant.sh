#!/usr/bin/env bash
# =============================================================================
# Take a Qdrant snapshot of the production collection and upload it to S3.
#
# Works unchanged against Qdrant Cloud or a self-hosted instance — it only
# talks to the standard snapshot REST API, authenticated with an api-key
# header. Point QDRANT_URL at whichever one is current.
#
# Usage:  ./scripts/backup_qdrant.sh
# Env vars:
#   QDRANT_URL              (required) — e.g. http://qdrant:6333 (self-hosted)
#                            or https://….cloud.qdrant.io (Qdrant Cloud)
#   QDRANT_API_KEY          (required) — must match QDRANT__SERVICE__API_KEY
#                            on the target instance
#   QDRANT_COLLECTION       (default: ghana_legal_docs)
#   BACKUP_S3_BUCKET        (required) — s3://bucket/path destination
#   BACKUP_RETENTION_DAYS   (default: 30) — local-file age before pruning
# =============================================================================
set -euo pipefail

: "${QDRANT_URL:?QDRANT_URL is required}"
: "${QDRANT_API_KEY:?QDRANT_API_KEY is required}"
: "${BACKUP_S3_BUCKET:?BACKUP_S3_BUCKET is required (s3://… URI)}"

QDRANT_COLLECTION="${QDRANT_COLLECTION:-ghana_legal_docs}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
WORK_DIR="$(mktemp -d -t legalite_qdrant_backup_XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "==> Creating Qdrant snapshot for collection=$QDRANT_COLLECTION ..."
SNAP_INFO="$(curl -sf -X POST \
    -H "api-key: $QDRANT_API_KEY" \
    "$QDRANT_URL/collections/$QDRANT_COLLECTION/snapshots")"

# The response shape is {"result":{"name":"…","creation_time":"…","size":…},"status":"ok",…}
SNAP_NAME="$(printf '%s' "$SNAP_INFO" | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["name"])')"
echo "   snapshot name: $SNAP_NAME"

echo "==> Downloading snapshot ..."
DOWNLOAD_URL="$QDRANT_URL/collections/$QDRANT_COLLECTION/snapshots/$SNAP_NAME"
OUT_FILE="$WORK_DIR/${QDRANT_COLLECTION}_${TS}.snapshot"
curl -sfL \
    -H "api-key: $QDRANT_API_KEY" \
    "$DOWNLOAD_URL" \
    --output "$OUT_FILE"
SIZE="$(stat -f%z "$OUT_FILE" 2>/dev/null || stat -c%s "$OUT_FILE")"
echo "   bytes: $SIZE"

echo "==> Uploading to $BACKUP_S3_BUCKET/qdrant/ ..."
aws s3 cp "$OUT_FILE" "$BACKUP_S3_BUCKET/qdrant/${QDRANT_COLLECTION}_${TS}.snapshot" \
    --only-show-errors

# Write a manifest line that lists snapshot, size, sha256.
SHA="$(shasum -a 256 "$OUT_FILE" | awk '{print $1}')"
MANIFEST="${WORK_DIR}/manifest.jsonl"
printf '{"ts":"%s","collection":"%s","name":"%s","size":%s,"sha256":"%s"}\n' \
    "$TS" "$QDRANT_COLLECTION" "$SNAP_NAME" "$SIZE" "$SHA" > "$MANIFEST"
aws s3 cp "$MANIFEST" "$BACKUP_S3_BUCKET/qdrant/manifest.jsonl" \
    --content-type application/x-ndjson \
    --only-show-errors

# Cleanup the remote snapshot on Qdrant Cloud (we've got a copy in S3).
echo "==> Deleting in-cluster snapshot ..."
curl -sf -X DELETE \
    -H "api-key: $QDRANT_API_KEY" \
    "$QDRANT_URL/collections/$QDRANT_COLLECTION/snapshots/$SNAP_NAME" > /dev/null

echo "==> Done. s3 path: $BACKUP_S3_BUCKET/qdrant/${QDRANT_COLLECTION}_${TS}.snapshot"
