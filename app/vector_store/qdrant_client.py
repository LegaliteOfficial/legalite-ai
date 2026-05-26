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

    def as_dict(self) -> dict[str, Any]:
        # Keep visibility_scope even when its value equals the default — we
        # rely on it being present for the filter, so dropping it would risk
        # a chunk being unfilterable.
        out: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if v is None:
                continue
            out[k] = v
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
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            prefer_grpc=False,
        )

    # ---- collection lifecycle ---------------------------------------------

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist. Idempotent."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            return
        logger.info("qdrant.creating_collection", name=self._collection, dim=self._dim)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
        )

    # ---- writes ------------------------------------------------------------

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
        self._client.upsert(collection_name=self._collection, points=points, wait=True)
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
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
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
