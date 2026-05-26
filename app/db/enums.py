"""Cross-cutting domain enums used across models, schemas, and Qdrant payloads.

We store them as plain strings in the database (not native Postgres ENUM
types) so adding a new value never requires a DDL migration — important
because document classifications are likely to expand as we onboard new
firms and matter types.
"""

from __future__ import annotations

from enum import StrEnum


class VisibilityScope(StrEnum):
    """Who can retrieve this artifact.

    - GLOBAL: visible to every request. Constitution, Acts, Regulations,
      public case law. ``organization_id`` MUST be null.
    - TENANT_PRIVATE: visible only to requests carrying a matching
      ``organization_id``. Sample cases, memos, templates, etc.
    """

    GLOBAL = "global"
    TENANT_PRIVATE = "tenant_private"


class DocumentOrigin(StrEnum):
    """Broad classification of the source material.

    Distinct from the chunker's ``law_type`` (constitution / act / regulation /
    case_law) — DocumentOrigin extends the taxonomy to tenant-private
    materials that the chunker's regex-based detector does not produce.
    """

    # Global / authoritative
    CONSTITUTION = "constitution"
    ACT = "act"
    REGULATION = "regulation"
    CASE_LAW = "case_law"
    PRECEDENT = "precedent"
    JUDGEMENT = "judgement"

    # Tenant-private
    SAMPLE_CASE = "sample_case"
    INTERNAL_MEMO = "internal_memo"
    STRATEGY_DOCUMENT = "strategy_document"
    TEMPLATE = "template"
    LEGAL_OPINION = "legal_opinion"
    PRIVATE_REFERENCE = "private_reference"


# Origins that may only be uploaded under TENANT_PRIVATE visibility.
TENANT_PRIVATE_ORIGINS: frozenset[DocumentOrigin] = frozenset(
    {
        DocumentOrigin.SAMPLE_CASE,
        DocumentOrigin.INTERNAL_MEMO,
        DocumentOrigin.STRATEGY_DOCUMENT,
        DocumentOrigin.TEMPLATE,
        DocumentOrigin.LEGAL_OPINION,
        DocumentOrigin.PRIVATE_REFERENCE,
    }
)

# Origins that may only be uploaded under GLOBAL visibility (admin only).
GLOBAL_ONLY_ORIGINS: frozenset[DocumentOrigin] = frozenset(
    {
        DocumentOrigin.CONSTITUTION,
        DocumentOrigin.ACT,
        DocumentOrigin.REGULATION,
        DocumentOrigin.CASE_LAW,
        DocumentOrigin.PRECEDENT,
        DocumentOrigin.JUDGEMENT,
    }
)


class ConfidentialityLevel(StrEnum):
    """How sensitive a tenant-private document is.

    Public means "tenant-private but unrestricted within the tenant"; the
    confidentiality scale only ranks among tenant-private docs. Global docs
    are always ``public``.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PRIVILEGED = "privileged"


class TenantRole(StrEnum):
    """Subset of legalite-backend's role taxonomy that this service cares about.

    The full role model lives in legalite-backend; we only enforce role-based
    branching for admin-only routes (e.g. global ingestion). Treat unknown
    role strings as ``viewer`` — least privileged.
    """

    OWNER = "owner"
    ADMIN = "admin"
    LAWYER = "lawyer"
    ASSOCIATE = "associate"
    VIEWER = "viewer"


# Roles permitted to upload to the GLOBAL knowledge base.
GLOBAL_INGEST_ROLES: frozenset[TenantRole] = frozenset({TenantRole.OWNER, TenantRole.ADMIN})
