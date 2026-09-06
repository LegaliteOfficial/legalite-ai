#!/usr/bin/env bash
# =============================================================================
# Restore a Qdrant snapshot into a collection — used both for disaster
# recovery and for the Qdrant Cloud -> self-hosted migration (see
# docs/QDRANT_SELF_HOSTED.md).
#
# DESTRUCTIVE: uploading a snapshot to an EXISTING collection name drops and
# rebuilds that collection from the snapshot. Restore into a differently
# named collection first (the "parallel restore" path) unless you are sure.
#
# Usage:
#   ./scripts/restore_qdrant.sh <snapshot-file-or-s3-uri> [target-collection] --confirm
#
# Examples:
#   # Parallel restore into a scratch collection for verification first:
#   ./scripts/restore_qdrant.sh s3://my-backups/qdrant/ghana_legal_docs_20260901.snapshot \
#       ghana_legal_docs_restore_20260901 --confirm
#
#   # In-place restore once verified (drops + rebuilds the live collection):
#   ./scripts/restore_qdrant.sh ./ghana_legal_docs_20260901.snapshot \
#       ghana_legal_docs --confirm
#
# Env vars:
#   QDRANT_URL         (required) — target instance, e.g. http://localhost:6333
#                       for self-hosted, or https://….cloud.qdrant.io for Cloud
#   QDRANT_API_KEY     (required) — must match the target instance's key
# =============================================================================
set -euo pipefail

SNAPSHOT_SRC="${1:?Usage: $0 <snapshot-file-or-s3-uri> [target-collection] --confirm}"
shift

TARGET_COLLECTION="${QDRANT_COLLECTION:-ghana_legal_docs}"
CONFIRM=""
for arg in "$@"; do
    case "$arg" in
        --confirm) CONFIRM="1" ;;
        *) TARGET_COLLECTION="$arg" ;;
    esac
done

: "${QDRANT_URL:?QDRANT_URL is required}"
: "${QDRANT_API_KEY:?QDRANT_API_KEY is required}"

if [[ -z "$CONFIRM" ]]; then
    echo "DRY RUN — no snapshot was uploaded. Re-run with --confirm to restore."
    echo "  source            : $SNAPSHOT_SRC"
    echo "  target collection : $TARGET_COLLECTION"
    echo "  target instance   : $QDRANT_URL"
    exit 0
fi

WORK_DIR="$(mktemp -d -t legalite_qdrant_restore_XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

if [[ "$SNAPSHOT_SRC" == s3://* ]]; then
    echo "==> Downloading snapshot from $SNAPSHOT_SRC ..."
    LOCAL_FILE="$WORK_DIR/$(basename "$SNAPSHOT_SRC")"
    aws s3 cp "$SNAPSHOT_SRC" "$LOCAL_FILE" --only-show-errors
else
    LOCAL_FILE="$SNAPSHOT_SRC"
fi

[[ -f "$LOCAL_FILE" ]] || { echo "Snapshot file not found: $LOCAL_FILE" >&2; exit 1; }

echo "==> Restoring '$LOCAL_FILE' into collection '$TARGET_COLLECTION' on $QDRANT_URL ..."
curl -sf -X POST \
    -H "api-key: $QDRANT_API_KEY" \
    -F "snapshot=@${LOCAL_FILE}" \
    "$QDRANT_URL/collections/$TARGET_COLLECTION/snapshots/upload?priority=snapshot"

echo
echo "==> Done. Verify with:"
echo "    curl -s -H \"api-key: \$QDRANT_API_KEY\" \"$QDRANT_URL/collections/$TARGET_COLLECTION\" | python3 -m json.tool"
