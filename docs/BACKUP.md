# Legalite AI — Backup & Restore Runbook

Three independent stateful surfaces back this service. Each needs its own
backup story because losing any one of them corrupts the system in a
different way:

| Surface | What's there | Worst-case loss | Recovery primitive |
| --- | --- | --- | --- |
| **Neon Postgres** | `legal_documents`, `legal_chunks` metadata, `legal_cases`, `case_citations`, `chat_sessions`/`chat_messages`, `ingestion_jobs`, + every legalite-backend table (users, orgs, etc.) | Tenant data, firm Q&A history, audit trail | Neon Point-in-Time Recovery (PITR) |
| **Qdrant collection** (`ghana_legal_docs`) | Every vector + payload (the content of every chunk) | All retrieval breaks; re-ingest needed | Qdrant snapshots (self-hosted, see [docs/QDRANT_SELF_HOSTED.md](QDRANT_SELF_HOSTED.md)) |
| **Uploads volume** (`/app/data/uploads`) | Original PDFs as uploaded by firms | Cannot re-ingest from source after Qdrant loss | Object-storage sync (S3 / GCS / R2) |

The three surfaces drift independently. A Postgres restore without a
matching Qdrant restore leaves the document_id → vector_id pointers
broken; a Qdrant restore without Postgres leaves vector hits with no
chunk metadata. **Always restore the most recent backup of all three to a
consistent timestamp.**

---

## 1. Neon Postgres — PITR

Neon ships continuous WAL archival on every plan. Default retention:

- **Free tier** — 24 h. Not acceptable for paying-firm data.
- **Pro / Scale** — 7 d to 30 d configurable.

**Recommended baseline before pilot firms onboard:** 14 d PITR + nightly
logical dumps for archive.

### Set PITR retention

1. Neon console → Project → Settings → Storage.
2. "History retention" → **14 days minimum** (or your contracted
   recovery point objective).
3. Cost note: PITR retention is the dominant Neon cost driver after
   compute. Document the choice in your monthly cost log.

### Nightly logical dump (offsite archive)

The PITR window is fast but lives on the same Neon storage. A monthly
ransomware / account-compromise scenario needs an **offline** copy.

```bash
# scripts/backup_postgres.sh — run from CI or a cron job (NOT the app container)
./scripts/backup_postgres.sh
```

The script dumps every legalite-ai-owned table (see "Migrations" in the
main README) plus shared `chat_sessions`/`chat_messages`, encrypts with
the operator's GPG key, and pushes to S3.

### Restore drill (must run quarterly)

1. Neon console → Branches → Create branch from timestamp.
2. Verify row counts on `legal_documents`, `legal_chunks`, `legal_cases`
   match expectations for the chosen timestamp.
3. Repoint a non-production app instance at the branch's
   `DATABASE_URL`.
4. Run `pytest -q tests/security/` + the smoke commands in the main
   README's "Test suite" section.
5. If everything passes, promote the branch to primary (or copy data
   selectively into prod).

Untested backups are worse than no backups. **Do the drill quarterly and
sign off in `docs/RESTORE_DRILLS.md`.**

---

## 2. Qdrant — collection snapshots

Qdrant exposes a snapshot API per collection. Snapshots are atomic and
include vectors + payloads + the collection schema.

### Take a snapshot

```bash
./scripts/backup_qdrant.sh
```

The script:

1. POSTs to `{QDRANT_URL}/collections/{QDRANT_COLLECTION}/snapshots`.
2. Downloads the resulting `.snapshot` file via the URL Qdrant returns.
3. Uploads it to the configured backup bucket (`BACKUP_S3_BUCKET`).
4. Records the snapshot timestamp + size into a local manifest for the
   restore tool to read.

Schedule: **every 6 h** during business hours, **daily off-hours**. Each
snapshot is the full collection — incremental snapshots aren't supported
by Qdrant as of this writing, so set retention based on disk
budget (typically 7 d hot + 30 d archive).

### Restore a snapshot

```bash
./scripts/restore_qdrant.sh <snapshot-file-or-url>
```

Two paths:

1. **In-place restore** — POST the snapshot to
   `{QDRANT_URL}/collections/{QDRANT_COLLECTION}/snapshots/upload`. The
   collection is **dropped and rebuilt** from the snapshot — destructive.
2. **Parallel restore** — restore into a new collection name (e.g.
   `ghana_legal_docs_restore_20260527`), point a non-production app
   instance at it, verify, then atomically swap names.

Use the parallel path for any restore involving customer data.

---

## 3. Uploads volume — object storage sync

The `upload_data` Docker volume holds the original PDFs. After a Qdrant
loss, these are the only way to re-derive the embeddings — without them
the firm has to re-upload every document.

### Mount strategy

Mount the upload volume against **versioned object storage**:

- AWS S3 with versioning enabled (~$0.023/GB/month, undelete via
  `aws s3api list-object-versions`).
- Cloudflare R2 with versioning (no egress fees).
- GCS with object versioning + lifecycle to move old versions to
  Archive class.

Sync direction is upload-only — never delete remotely just because a
file disappeared locally (avoid the "rm -rf bug → backup gone" failure
mode).

```bash
./scripts/backup_uploads.sh
```

Internally this is a one-way `aws s3 sync --delete=false /app/data/uploads s3://<bucket>/uploads/`,
or the equivalent for your provider. Run every 15 min as a cron task on
the same host as the app (or as a sidecar in Kubernetes).

### Restore uploads

```bash
aws s3 sync s3://<bucket>/uploads/ /app/data/uploads/
```

Then re-trigger ingestion by hitting `/admin/upload-law` for each file
that needs to land back in Qdrant (or write a one-shot script that
walks the directory and calls `IngestionService.ingest_pdf` directly).

---

## 4. Consistency window

The three surfaces above each have their own RPO (recovery point
objective). The realistic floor is dominated by the longest:

| Surface | Default cadence | RPO |
| --- | --- | --- |
| Postgres PITR | continuous WAL | ~1 min |
| Postgres nightly dump | 24 h | ≤ 24 h |
| Qdrant snapshot | 6 h | ≤ 6 h |
| Uploads sync | 15 min | ≤ 15 min |

If a hard restore lands at T-2h, Postgres goes to T-2h via PITR,
Qdrant to the latest pre-T-2h snapshot, and uploads to the latest sync
≤ T-2h. Document IDs created after the chosen Postgres timestamp will
have no matching vectors and should be marked failed:

```sql
UPDATE legal_documents
   SET status = 'failed', notes = 'orphaned by restore — re-ingest needed'
 WHERE created_at > <restore_timestamp>
   AND status = 'ingested'
   AND NOT EXISTS (
     SELECT 1 FROM legal_chunks c
     WHERE c.document_id = legal_documents.id
   );
```

This query is included verbatim in `scripts/restore_reconcile.sh`.

---

## 5. Operational checklist

Run through this before declaring a backup strategy "done":

- [ ] Neon PITR retention ≥ 14 d, alert if it drops below.
- [ ] Nightly Postgres logical dump runs via cron / scheduled CI.
- [ ] Qdrant snapshot every 6 h; manifest written to S3.
- [ ] Uploads volume syncs to versioned object storage every 15 min.
- [ ] **Quarterly restore drill** completed and signed off in
      `docs/RESTORE_DRILLS.md`.
- [ ] Pagerduty / opsgenie alerts on any backup script's non-zero exit.
- [ ] Documented `BACKUP_S3_BUCKET`, `BACKUP_GPG_RECIPIENT`,
      `QDRANT_API_KEY` in the secrets vault used by ops, NOT in git.

A backup that has never been restored is theatre. The drill is the
only thing that proves the rest works.
