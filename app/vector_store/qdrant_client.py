"""Qdrant client wrapper.

Encapsulates collection management, upsert, and search so the rest of the app
never imports `qdrant_client` directly. This keeps the abstraction swappable
(e.g. for a different vector DB) and easy to mock in tests.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.telemetry import qdrant_failures

logger = get_logger(__name__)


@dataclass(slots=True)
class VectorPayload:
    """Metadata stored alongside each vector in Qdrant."""

    document_id: str
    chunk_id: str
    law: str | None = None
    doc_type: str | None = None
    title: str | None = None
    chapter: str | None = None
    part: str | None = None
    article: str | None = None
    section: str | None = None
    clause: str | None = None
    subsection: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    content: str = ""
    # v2 metadata
    law_type: str | None = None
    act_number: str | None = None
    effective_year: int | None = None
    legal_topic: str | None = None
    # Tenancy + classification (migration 0004). Required: visibility_scope
    # is always set. organization_id is None for GLOBAL chunks.
    organization_id: str | None = None
    visibility_scope: str = "global"
    document_origin: str | None = None
    confidentiality_level: str | None = None
    # Phase 5 M9 — character offsets for UI source highlighting.
    start_char: int | None = None
    end_char: int | None = None

    def as_dict(self) -> dict[str, Any]:
        # Note (audit fix): VectorPayload is declared @dataclass(slots=True),
        # which means it has __slots__ and NO __dict__. The previous
        # implementation iterated self.__dict__ and crashed at runtime
        # the first time anything actually wrote to Qdrant — the integration
        # test that exercises real ingestion is what caught it. Iterate the
        # declared dataclass fields instead; works regardless of slots.
        from dataclasses import fields

        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            out[f.name] = v
        # Belt-and-braces: ensure visibility_scope is always in the payload.
        out.setdefault("visibility_scope", "global")
        return out


@dataclass(slots=True)
class VectorHit:
    score: float
    payload: dict[str, Any]
    point_id: str


class QdrantVectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._collection = settings.qdrant_collection
        self._dim = settings.openai_embedding_dim
        # ``timeout`` is in seconds. The default (5s) is too aggressive
        # for Qdrant Cloud batched upserts with ``wait=True``: a 256-point
        # batch over a transcontinental link can take 10-30s end-to-end,
        # and search calls during indexing can briefly stall. 120s is
        # large enough to absorb that without masking a truly dead
        # cluster — Arq / FastAPI request budgets will still cap us.
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            prefer_grpc=False,
            timeout=120,
        )

    # ---- collection lifecycle ---------------------------------------------

    def ensure_collection(self) -> None:
        """Create the collection (and required payload indexes) if missing.

        Idempotent. The payload indexes are mandatory for tenant filtering
        — without them Qdrant rejects any ``Filter`` that references those
        fields with HTTP 400. The integration suite caught this bug; before
        the fix, every tenant-aware retrieval in production would have
        failed once a real Qdrant filter was applied.
        """
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            logger.info("qdrant.creating_collection", name=self._collection, dim=self._dim)
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
            )

        # Always (re)assert the payload indexes — cheap when they exist.
        # These four fields are referenced by every retrieval filter in the
        # codebase (see build_tenant_filter + HybridRetriever law_type narrows).
        for field_name in ("visibility_scope", "organization_id", "law_type", "document_origin"):
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:  # noqa: BLE001
                # Already-exists errors come back as 400 with a descriptive
                # body; we treat all "couldn't create" outcomes as idempotent
                # success. A genuinely-broken index will surface as a search
                # error later, which qdrant_failures{search} captures.
                logger.debug(
                    "qdrant.payload_index_skip",
                    field=field_name,
                    collection=self._collection,
                )

    # ---- writes ------------------------------------------------------------

    # Qdrant Cloud rejects JSON payloads >32 MiB with HTTP 400. With
    # 1536-d float embeddings each point is ~6 KB of vector + a few KB of
    # payload text, so a long document (e.g. the Ghana Constitution at
    # several thousand chunks) overflows in a single upsert. Batching at
    # 256 points keeps every request comfortably under the limit.
    _UPSERT_BATCH_SIZE = 256

    def upsert(
        self,
        embeddings: list[list[float]],
        payloads: list[VectorPayload],
    ) -> list[str]:
        if len(embeddings) != len(payloads):
            raise ValueError("embeddings and payloads must be the same length")
        if not embeddings:
            return []

        point_ids = [str(uuid.uuid4()) for _ in embeddings]
        points = [
            qm.PointStruct(id=pid, vector=vec, payload=pl.as_dict())
            for pid, vec, pl in zip(point_ids, embeddings, payloads, strict=True)
        ]
        for start in range(0, len(points), self._UPSERT_BATCH_SIZE):
            batch = points[start : start + self._UPSERT_BATCH_SIZE]
            self._client.upsert(
                collection_name=self._collection,
                points=batch,
                wait=True,
            )
        return point_ids

    def delete_by_document(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="document_id",
                            match=qm.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    # ---- reads -------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        *,
        query_filter: qm.Filter | None = None,
    ) -> list[VectorHit]:
        """Vector search with an optional Qdrant payload filter.

        For tenant-aware searches, callers MUST pass a filter built by
        :func:`build_tenant_filter`. Calling this without a filter returns
        ALL chunks regardless of tenancy — used only by the legacy /ask
        path and the global-corpus operator tooling.
        """
        try:
            results = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                query_filter=query_filter,
            )
        except Exception:
            qdrant_failures.labels(operation="search").inc()
            raise
        return [
            VectorHit(
                score=float(r.score),
                payload=dict(r.payload or {}),
                point_id=str(r.id),
            )
            for r in results
        ]


# ---------------------------------------------------------------------------
# Tenant filter — the load-bearing primitive for knowledge isolation.
# ---------------------------------------------------------------------------


def build_tenant_filter(
    *,
    organization_id: str | uuid.UUID | None,
    extra_must: list[qm.FieldCondition] | None = None,
) -> qm.Filter:
    """Construct the Qdrant filter that enforces tenant boundaries.

    Semantics::

        organization_id is None   →  visibility_scope == "global"
        organization_id is set    →  visibility_scope == "global"
                                     OR organization_id == <org_id>
                                     (matching org also requires visibility_scope == "tenant_private")

    The second branch is encoded as a ``should`` clause inside a ``must``
    block: returned points satisfy at least one of the OR branches AND any
    extra constraints (e.g. ``law_type == case_law``).

    Cross-tenant leakage proof
    --------------------------
    The only way a TENANT_PRIVATE chunk surfaces is if its
    ``organization_id`` payload equals the supplied org_id AND its
    visibility_scope equals ``tenant_private``. There is no path where an
    unrelated org's private chunk passes the filter, because every branch
    pins visibility_scope explicitly.
    """
    org_value: str | None = None
    if organization_id is not None:
        org_value = str(organization_id)

    global_branch = qm.Filter(
        must=[
            qm.FieldCondition(
                key="visibility_scope",
                match=qm.MatchValue(value="global"),
            )
        ]
    )

    branches: list[qm.Filter] = [global_branch]

    if org_value is not None:
        tenant_branch = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="visibility_scope",
                    match=qm.MatchValue(value="tenant_private"),
                ),
                qm.FieldCondition(
                    key="organization_id",
                    match=qm.MatchValue(value=org_value),
                ),
            ]
        )
        branches.append(tenant_branch)

    outer_must: list[qm.FieldCondition | qm.Filter] = []
    if extra_must:
        outer_must.extend(extra_must)
    # The branches go into a `should` so at least one must match.
    return qm.Filter(must=outer_must or None, should=branches)


_store: QdrantVectorStore | None = None


def get_vector_store() -> QdrantVectorStore:
    global _store
    if _store is None:
        _store = QdrantVectorStore()
        _store.ensure_collection()
    return _store
