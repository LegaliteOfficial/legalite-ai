"""Live cross-tenant /sessions probing — uses the real FastAPI app.

Asserts that the P0-1 fix actually closes the route. The unit test
exercises the repository layer; this test exercises the same end-to-end
HTTP contract a client would hit, against a real Postgres.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.mark.integration
def test_anonymous_session_not_readable_by_tenant(client):
    created = client.post("/sessions").json()
    sid = created["session_id"]

    resp_anon = client.get(f"/sessions/{sid}")
    assert resp_anon.status_code == 200

    resp_tenant = client.get(
        f"/sessions/{sid}",
        headers={
            "x-organization-id": str(uuid.uuid4()),
            "x-user-id": str(uuid.uuid4()),
        },
    )
    assert resp_tenant.status_code == 404, "anonymous session leaked to a tenant"


@pytest.mark.integration
def test_firm_a_session_not_readable_by_firm_b(client):
    org_a, user_a = str(uuid.uuid4()), str(uuid.uuid4())
    org_b, user_b = str(uuid.uuid4()), str(uuid.uuid4())

    created = client.post(
        "/sessions",
        headers={"x-organization-id": org_a, "x-user-id": user_a},
    ).json()
    sid = created["session_id"]

    own = client.get(
        f"/sessions/{sid}",
        headers={"x-organization-id": org_a, "x-user-id": user_a},
    )
    assert own.status_code == 200

    other = client.get(
        f"/sessions/{sid}",
        headers={"x-organization-id": org_b, "x-user-id": user_b},
    )
    assert other.status_code == 404, "Firm A session leaked to Firm B"

    anon = client.get(f"/sessions/{sid}")
    assert anon.status_code == 404, "Firm A session leaked to anonymous"


@pytest.mark.integration
def test_admin_global_job_invisible_to_anonymous(client):
    """/ingestion/jobs/{random} should never return a global admin job to
    anonymous probers. Even if the UUID happens to match a real one, the
    visibility predicate must return 404."""
    # Probing a random UUID — without a global job row that anonymous can
    # see, this should always be 404. Repeated to exercise the negative path.
    for _ in range(3):
        resp = client.get(f"/ingestion/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404
