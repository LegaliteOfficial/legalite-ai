# Service Authentication Runbook

`legalite-ai` accepts tenant identity via three headers:

```
x-organization-id: <uuid>
x-user-id:         <uuid>
x-role:            owner | admin | lawyer | associate | viewer
```

In **trust-the-network mode** (no `LEGALITE_GATEWAY_SECRET` set), the
service trusts these headers at face value. Anyone who can reach the
service's HTTP endpoint can claim to be any tenant. That's acceptable
ONLY when the service is reachable solely from within a private
network — typically when it sits behind a backend gateway with no public
ingress.

**In hosted environments** (Railway, Fly, EC2 behind ALB, etc.) the
service is publicly reachable as soon as it has a domain. The
trust-the-network model is unsafe there. This runbook documents the HMAC
mode, which makes forged tenant headers cryptographically impossible.

---

## 1. Generate the shared secret

Once per environment. Use a 32-byte (256-bit) random key:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
# → e.g. "jZk3vQ-eN-T8wL5pYqM2bH-rXcA1fGdB9oVuKsRiPeI"
```

Treat this string like a database password:

- Store it in your secrets manager (Railway Variables, 1Password, AWS
  Secrets Manager, etc.), NEVER in git.
- Set it identically in **both** `legalite-ai` and `legalite-backend` as
  `LEGALITE_GATEWAY_SECRET`.
- Rotate quarterly, or immediately on suspected compromise. The rotation
  procedure is in §5.

---

## 2. The wire protocol

For every tenant request, the backend MUST set two additional headers
alongside the existing tenant tuple:

```
x-legalite-timestamp: <unix seconds, integer>
x-legalite-signature: <hex-encoded HMAC-SHA256>
```

The signed message is the canonical concatenation:

```
<timestamp>.<organization_id>.<user_id>.<role>
```

With these rules (failure to follow → 401):

- Empty string for any missing component (e.g. legacy anonymous requests
  have everything empty; they need no signature).
- All UUIDs and the role string lowercased before signing.
- Both sides use HMAC-SHA256, hex-encoded output.

The receiver enforces freshness: timestamps older than
`GATEWAY_SIGNATURE_MAX_SKEW_SECONDS` (default **300 s** / 5 min) are
rejected as replays.

---

## 3. Backend signing — TypeScript / NestJS

Drop this into legalite-backend. It uses only the Node `crypto` stdlib:

```typescript
// src/auth/legalite-ai-signer.ts
import { createHmac } from "crypto";

const SECRET = process.env.LEGALITE_GATEWAY_SECRET;
if (!SECRET) {
  // Crash on boot if misconfigured — silent fallback would mean every
  // request to legalite-ai is unsigned, which legalite-ai will refuse.
  throw new Error("LEGALITE_GATEWAY_SECRET is required");
}

export interface TenantHeaders {
  "x-organization-id": string;
  "x-user-id": string;
  "x-role": string;
  "x-legalite-timestamp": string;
  "x-legalite-signature": string;
}

export function signTenantHeaders(input: {
  organizationId: string;
  userId: string;
  role: string;
}): TenantHeaders {
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const org = input.organizationId.toLowerCase();
  const usr = input.userId.toLowerCase();
  const role = input.role.toLowerCase();
  const message = `${timestamp}.${org}.${usr}.${role}`;
  const signature = createHmac("sha256", SECRET).update(message).digest("hex");
  return {
    "x-organization-id": input.organizationId,
    "x-user-id": input.userId,
    "x-role": input.role,
    "x-legalite-timestamp": timestamp,
    "x-legalite-signature": signature,
  };
}
```

Wire it into your HTTP client. For Axios:

```typescript
import axios from "axios";
import { signTenantHeaders } from "./legalite-ai-signer";

const legaliteAI = axios.create({
  baseURL: process.env.LEGALITE_AI_URL,
});

legaliteAI.interceptors.request.use((cfg) => {
  // Pull the authenticated tenant off the request context however your
  // backend tracks it (CLS, AsyncLocalStorage, request-scoped service, …).
  const tenant = getCurrentTenant();
  if (tenant) {
    Object.assign(cfg.headers, signTenantHeaders({
      organizationId: tenant.organizationId,
      userId:         tenant.userId,
      role:           tenant.role,
    }));
  }
  return cfg;
});
```

Anonymous requests (e.g. health probes, public reads) don't need the
signing headers at all — legalite-ai accepts them as anonymous.

---

## 4. Verifying it works

### 4.1 From the legalite-ai container

```bash
# Set the same secret legalite-ai is using:
export LEGALITE_GATEWAY_SECRET="<the secret from Railway>"

# Print curl-ready signed headers:
docker compose exec -e LEGALITE_GATEWAY_SECRET="$LEGALITE_GATEWAY_SECRET" app \
  python -m scripts.sign_request \
    --org-id  "$(uuidgen)" \
    --user-id "$(uuidgen)" \
    --role    admin \
    --format  curl
```

Copy the printed command, append the URL, and hit `/admin/upload-law` or
`/documents/private-upload`. If the signature is valid you'll get the
normal 4xx (missing file, etc.), not a 401.

### 4.2 Cross-check signatures match between Python and TypeScript

Pick a fixed `(timestamp, org_id, user_id, role)` and produce the
signature both ways:

```python
# Python side
from app.core.service_auth import compute_signature
print(compute_signature(
    timestamp=1730000000,
    organization_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    user_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    role="lawyer",
    secret="test-secret",
))
```

```typescript
// TypeScript side — same inputs MUST produce the same hex digest.
import { createHmac } from "crypto";
const sig = createHmac("sha256", "test-secret")
  .update("1730000000.aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.lawyer")
  .digest("hex");
console.log(sig);
```

If the strings don't match, the most likely culprits are (in order of
frequency): case mismatch in UUIDs, trailing whitespace in the secret,
or the role field being capitalised differently. The Python helper
lowercases everything before signing; your TS code must do the same.

### 4.3 Negative tests

These MUST all return 401:

```bash
BASE=https://<your-legalite-ai-url>

# 1. Tampered signature
curl -i -X POST "$BASE/admin/upload-law" \
  -H "x-organization-id: $(uuidgen)" \
  -H "x-user-id: $(uuidgen)" \
  -H "x-role: admin" \
  -H "x-legalite-timestamp: $(date +%s)" \
  -H "x-legalite-signature: 00000000000000000000000000000000"

# 2. Stale timestamp (older than 5 min)
STALE=$(($(date +%s) - 10000))
# ... sign with that STALE value, then send

# 3. Right signature but escalated role (sign as lawyer, send as admin)
# legalite-ai's HMAC covers the role, so changing it post-sign invalidates.
```

---

## 5. Rotation procedure

The shared secret rotates without downtime via the "two-secret window"
pattern. legalite-ai's current verifier only supports one secret at a
time, so the procedure is:

1. **Pick the new secret** (`secrets.token_urlsafe(32)`).
2. **Update legalite-backend FIRST** to send the NEW signature.
3. **Switch legalite-ai's `LEGALITE_GATEWAY_SECRET` to the new value.**
   The brief window where backend signs with new + ai verifies with old
   will reject requests with 401.
4. **Cool-down: keep monitoring `legalite_refusal_total{reason="…"}`
   for 24 h.** A spike to 401s indicates a backend node missed the
   rotation; redeploy that node.

If you can't tolerate even a few seconds of 401s during the swap, add a
second secret env (`LEGALITE_GATEWAY_SECRET_PREVIOUS`) to legalite-ai
and update `verify_gateway_signature` to accept either — that's a
~10-line code change, not yet implemented.

---

## 6. What "logged out" looks like

If the secret is set but signatures don't validate, every tenant request
gets HTTP 401 from
[`get_optional_tenant_context`](../app/api/dependencies/tenant.py) /
[`get_required_tenant_context`](../app/api/dependencies/tenant.py) (via
the shared `verify_gateway_signature` call in
[`app/core/service_auth.py`](../app/core/service_auth.py)).

Anonymous requests (no `x-organization-id`) still work — they don't
require a signature, by design, so the legacy `/ask` + `/upload-law`
client doesn't break.

---

## 7. Operational checklist

- [ ] Generated a 32-byte secret with `secrets.token_urlsafe(32)`.
- [ ] Stored it in the secrets manager (NOT in `.env` committed to git).
- [ ] Set `LEGALITE_GATEWAY_SECRET` on **both** Railway services
      (legalite-ai web + worker) if you have a worker.
- [ ] Set the same value as the corresponding env on legalite-backend.
- [ ] Backend code pushes through the Axios / fetch interceptor for
      every tenant request.
- [ ] Smoke-test from the backend hitting `/health` → 200 anonymous,
      then `/cases/similar` with signed headers → 200.
- [ ] Smoke-test the negative paths from §4.3 — each returns 401.
- [ ] Rotation calendar entry set (default: quarterly).
- [ ] Pagerduty alert on `legalite_refusal_total{reason="planner"}` AND
      `legalite_tenant_isolation_score < 1.0` (already in main runbook).
