# Qdrant: self-hosted (replacing Qdrant Cloud)

Legalite AI no longer depends on Qdrant Cloud. `docker-compose.yml` runs its
own `qdrant/qdrant` container on a private Docker network, with a persistent
volume and API-key authentication. The application code is unchanged —
`app/vector_store/qdrant_client.py` still talks to Qdrant through the
official `qdrant-client` SDK using `QDRANT_URL` / `QDRANT_API_KEY`; only the
target instance changed.

## Architecture

```
Application (app, worker containers)
        │  QDRANT_URL=http://qdrant:6333   (internal Docker network)
        ▼
   Qdrant container (qdrant/qdrant:v1.12.4)
        │  API-key auth (QDRANT__SERVICE__API_KEY)
        ▼
   qdrant_data (persistent Docker volume)
```

Qdrant's ports are published to `127.0.0.1` only (not `0.0.0.0`), so they
are reachable from the host machine for local debugging but not from the
network. The app and worker containers reach Qdrant over the internal
compose network by service name (`qdrant:6333`) — no host binding is needed
for that path at all.

For a production deployment behind a reverse proxy / API gateway, Qdrant
should have **no public port at all**:

```
Internet → Reverse proxy / API gateway → Backend app (this repo)
                                                │
                                     private Docker network
                                                ▼
                                             Qdrant
                                                │
                                     persistent volume
```

Only the backend container should ever hold `QDRANT_API_KEY` or reach the
Qdrant port.

## Local development

```bash
# Start the full stack (app, worker, postgres, redis, qdrant)
docker compose up -d

# Or just the dependency, if running the API on your host with uvicorn:
docker compose up -d qdrant
```

Verify Qdrant is healthy:

```bash
docker compose ps qdrant                 # STATUS should show "healthy"
curl -H "api-key: $QDRANT_API_KEY" http://localhost:6333/collections
```

Inspect logs:

```bash
docker compose logs -f qdrant
```

Stop the service:

```bash
docker compose stop qdrant       # keep data, just stop the container
docker compose down              # stop everything, volumes persist
docker compose down -v           # stop everything AND delete volumes (data loss)
```

Running the API outside Docker (`uvicorn app.main:app --reload` on your
host): set `QDRANT_URL=http://localhost:6333` in your local `.env` instead
of `http://qdrant:6333` (that hostname only resolves inside the compose
network), and keep `docker compose up -d qdrant` running.

## Environment variables

| Variable | Local (Docker Compose) | Local (app on host, Qdrant in Docker) |
| --- | --- | --- |
| `QDRANT_URL` | `http://qdrant:6333` | `http://localhost:6333` |
| `QDRANT_API_KEY` | generated secret, shared by the `qdrant` service and the app | same |
| `QDRANT_COLLECTION` | `ghana_legal_docs` (unchanged) | same |

Generate a key with:

```bash
openssl rand -hex 32
```

Set the same value for `QDRANT_API_KEY` in `.env` — it is used both to start
the `qdrant` container (`QDRANT__SERVICE__API_KEY` in `docker-compose.yml`
reads it from the same variable) and by the app's `QdrantClient`. Never
commit a real value — `.env.example` only holds a placeholder.

## Production deployment

- **Persistent volume**: the `qdrant_data` named volume in `docker-compose.yml`
  maps to `/qdrant/storage` inside the container. Back it with durable
  storage in your orchestrator (an EBS/PD-backed volume, not ephemeral
  instance storage).
- **Backups**: use `scripts/backup_qdrant.sh` (snapshot API export to S3) on
  a schedule — see `docs/BACKUP.md`. Snapshots are point-in-time and include
  vectors + payloads + schema.
- **API keys**: set `QDRANT_API_KEY` from your secrets manager, not a
  checked-in file. Rotate it by updating the secret and restarting both the
  `qdrant` service and the app/worker together (a mismatched key means the
  app fails closed with 403s from Qdrant, logged via
  `legalite_qdrant_failures_total`).
- **Network isolation**: do not publish Qdrant's ports beyond the private
  network your app runs on. If your orchestrator requires an explicit
  network policy (Kubernetes NetworkPolicy, ECS security group, etc.),
  restrict ingress on 6333/6334 to the app and worker only.
- **Restart policy**: `restart: unless-stopped` is set on the `qdrant`
  service; in Kubernetes/ECS use the equivalent "always restart" policy so a
  node failure doesn't leave the vector store down.

## Migrating existing data from Qdrant Cloud

This does not delete anything on Qdrant Cloud — it's a read-only export
followed by a restore into the new self-hosted instance.

1. **Export a snapshot from Qdrant Cloud** (point the env vars at the Cloud
   cluster you're migrating away from):

   ```bash
   QDRANT_URL="https://<your-cluster>.cloud.qdrant.io" \
   QDRANT_API_KEY="<your-cloud-api-key>" \
   QDRANT_COLLECTION="ghana_legal_docs" \
   BACKUP_S3_BUCKET="s3://your-backup-bucket" \
     ./scripts/backup_qdrant.sh
   ```

   This creates a snapshot on the Cloud cluster, downloads it locally, and
   uploads it to S3 (or adapt the script if you'd rather just keep the local
   file — the snapshot is left at `$WORK_DIR` before cleanup if you comment
   out the `trap`).

2. **Bring up the self-hosted instance** (`docker compose up -d qdrant`) and
   confirm it's healthy (see above).

3. **Restore into a scratch collection first**, so you can verify before
   touching anything the app reads from:

   ```bash
   QDRANT_URL="http://localhost:6333" \
   QDRANT_API_KEY="$QDRANT_API_KEY" \
     ./scripts/restore_qdrant.sh \
       s3://your-backup-bucket/qdrant/ghana_legal_docs_<timestamp>.snapshot \
       ghana_legal_docs_restore_check \
       --confirm
   ```

   Verify point count and a couple of payloads match the source:

   ```bash
   curl -s -H "api-key: $QDRANT_API_KEY" \
     http://localhost:6333/collections/ghana_legal_docs_restore_check | python3 -m json.tool
   ```

4. **Restore into the real collection name** the app expects
   (`QDRANT_COLLECTION`, default `ghana_legal_docs`). This drops and rebuilds
   that collection name if it already exists on the target — safe here
   because the target is the fresh self-hosted instance, not the Cloud
   source:

   ```bash
   QDRANT_URL="http://localhost:6333" \
   QDRANT_API_KEY="$QDRANT_API_KEY" \
     ./scripts/restore_qdrant.sh \
       s3://your-backup-bucket/qdrant/ghana_legal_docs_<timestamp>.snapshot \
       ghana_legal_docs \
       --confirm
   ```

5. **Point the app at the self-hosted instance** by setting
   `QDRANT_URL=http://qdrant:6333` (or your production internal DNS name)
   and the matching `QDRANT_API_KEY` in the app's environment, then restart
   the app/worker.

6. **Delete the scratch collection** once verified:

   ```bash
   curl -X DELETE -H "api-key: $QDRANT_API_KEY" \
     http://localhost:6333/collections/ghana_legal_docs_restore_check
   ```

7. Only after the self-hosted instance is confirmed serving traffic
   correctly, decommission the Qdrant Cloud cluster from its console. This
   script never touches or deletes anything on the Cloud side beyond the
   transient in-cluster snapshot `backup_qdrant.sh` already cleans up after
   itself.

## Verification checklist

```bash
docker compose up -d
docker compose ps                                    # qdrant should be "healthy"
curl -s http://localhost:8000/health | python3 -m json.tool   # {"status":"ok","qdrant":"ok",...}

# Collection lifecycle (through the app's existing wrapper — no new code path):
docker compose exec app python -m scripts.qdrant_orphans      # read-only sanity check

# Persistence check
docker compose restart qdrant
curl -s -H "api-key: $QDRANT_API_KEY" http://localhost:6333/collections/ghana_legal_docs \
  | python3 -m json.tool          # points_count unchanged after restart
```

Collection creation, upsert, search, filtering, and delete all go through
the existing `QdrantVectorStore` in `app/vector_store/qdrant_client.py` and
`build_tenant_filter` — none of that logic changed, only where it connects.
