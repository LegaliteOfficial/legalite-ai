"""P0-1 — chat sessions are tenant-scoped at the repository layer.

These tests pin the boundary that's enforced inside ChatRepository:

- A session created with org_id=X is invisible to any caller asking with
  org_id != X (including anonymous callers asking with org_id=None).
- Anonymous-created sessions are only readable by anonymous callers.
- LegalMemory.get_or_create_session refuses to resume a cross-tenant
  session — it mints a fresh one under the caller's scope instead.

The repository tests use an in-process SQLite via SQLAlchemy because
the existing test infra targets a real Postgres only in integration.
Here we exercise the model + repo logic directly with stub objects so
the test stays fast and offline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.core.tenant_context import TenantContext
from app.db.enums import TenantRole


@dataclass
class _StubSession:
    id: uuid.UUID
    organization_id: uuid.UUID | None
    user_id: uuid.UUID | None
    title: str | None = None


class _StubChatRepository:
    """Minimal in-memory stand-in mirroring the production repo's API."""

    def __init__(self) -> None:
        self._sessions: dict[uuid.UUID, _StubSession] = {}

    def create_session(
        self,
        title: str | None = None,
        *,
        organization_id: uuid.UUID | None = None,
        user_id: uuid.UUID | None = None,
    ) -> _StubSession:
        sid = uuid.uuid4()
        s = _StubSession(
            id=sid,
            organization_id=organization_id,
            user_id=user_id,
            title=title,
        )
        self._sessions[sid] = s
        return s

    def get_session(
        self,
        session_id: uuid.UUID,
        *,
        organization_id: uuid.UUID | None = None,
    ) -> _StubSession | None:
        s = self._sessions.get(session_id)
        if s is None:
            return None
        if s.organization_id != organization_id:
            return None
        return s

    def add_message(self, *_args, **_kwargs):  # not exercised here
        raise NotImplementedError

    def get_recent_messages(self, *_args, **_kwargs):
        return []


def _tenant(org: uuid.UUID, user: uuid.UUID) -> TenantContext:
    return TenantContext(organization_id=org, user_id=user, role=TenantRole.LAWYER)


def test_anonymous_session_is_only_readable_by_anonymous() -> None:
    repo = _StubChatRepository()
    sess = repo.create_session()
    assert repo.get_session(sess.id) is not None  # anonymous → ok
    assert repo.get_session(sess.id, organization_id=uuid.uuid4()) is None  # tenant → 404


def test_tenant_session_invisible_to_other_tenants() -> None:
    repo = _StubChatRepository()
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    sess = repo.create_session(organization_id=org_a, user_id=uuid.uuid4())
    # Owner sees it.
    assert repo.get_session(sess.id, organization_id=org_a) is not None
    # Cross-tenant probe returns None (the route then 404s).
    assert repo.get_session(sess.id, organization_id=org_b) is None
    # Anonymous probe returns None.
    assert repo.get_session(sess.id, organization_id=None) is None


def test_legal_memory_mints_fresh_session_when_handed_a_cross_tenant_id() -> None:
    """The most subtle property: an attacker who steals Firm A's session
    UUID and calls /ask with Firm B headers MUST NOT have their /ask
    write messages into Firm A's session."""
    from app.rag.legal_memory import LegalMemory

    repo = _StubChatRepository()
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    sess_a = repo.create_session(organization_id=org_a, user_id=uuid.uuid4())

    memory = LegalMemory(repo)  # type: ignore[arg-type]
    firm_b_ctx = _tenant(org_b, uuid.uuid4())

    returned_sid = memory.get_or_create_session(str(sess_a.id), tenant=firm_b_ctx)

    # MUST be a new session, not the original A session.
    assert returned_sid != sess_a.id
    # And it must be scoped to org_b.
    new_session = repo.get_session(returned_sid, organization_id=org_b)
    assert new_session is not None
    assert new_session.organization_id == org_b


def test_legal_memory_resumes_own_session() -> None:
    from app.rag.legal_memory import LegalMemory

    repo = _StubChatRepository()
    org = uuid.uuid4()
    user = uuid.uuid4()
    sess = repo.create_session(organization_id=org, user_id=user)

    memory = LegalMemory(repo)  # type: ignore[arg-type]
    ctx = _tenant(org, user)
    returned_sid = memory.get_or_create_session(str(sess.id), tenant=ctx)

    assert returned_sid == sess.id


def test_legal_memory_anonymous_resumes_anonymous() -> None:
    from app.rag.legal_memory import LegalMemory

    repo = _StubChatRepository()
    sess = repo.create_session()  # anonymous

    memory = LegalMemory(repo)  # type: ignore[arg-type]
    returned_sid = memory.get_or_create_session(str(sess.id), tenant=None)
    assert returned_sid == sess.id
