"""build_tenant_filter — the security primitive for knowledge isolation.

These tests inspect the structure of the generated Qdrant Filter directly.
They're cheap, deterministic, and would catch the kind of typo (e.g.
flipping must/should) that would silently leak data across tenants.
"""

from __future__ import annotations

import uuid

from qdrant_client.http import models as qm

from app.vector_store.qdrant_client import build_tenant_filter


def _conditions_for_key(branch: qm.Filter, key: str) -> list[qm.FieldCondition]:
    return [c for c in (branch.must or []) if isinstance(c, qm.FieldCondition) and c.key == key]


def test_anonymous_filter_pins_global_only() -> None:
    f = build_tenant_filter(organization_id=None)
    # Outer should has exactly one branch — the GLOBAL branch.
    assert f.should is not None
    assert len(f.should) == 1
    branch = f.should[0]
    assert isinstance(branch, qm.Filter)
    scope_conds = _conditions_for_key(branch, "visibility_scope")
    assert len(scope_conds) == 1
    assert scope_conds[0].match.value == "global"  # type: ignore[union-attr]


def test_tenant_filter_has_global_and_tenant_branches() -> None:
    org = uuid.uuid4()
    f = build_tenant_filter(organization_id=org)
    assert f.should is not None
    assert len(f.should) == 2

    # Map each branch by its visibility_scope value for easier assertions.
    by_scope: dict[str, qm.Filter] = {}
    for branch in f.should:
        assert isinstance(branch, qm.Filter)
        scope_conds = _conditions_for_key(branch, "visibility_scope")
        assert len(scope_conds) == 1
        by_scope[scope_conds[0].match.value] = branch  # type: ignore[union-attr]

    assert set(by_scope) == {"global", "tenant_private"}

    # The tenant branch MUST pin the org id.
    tenant_branch = by_scope["tenant_private"]
    org_conds = _conditions_for_key(tenant_branch, "organization_id")
    assert len(org_conds) == 1
    assert org_conds[0].match.value == str(org)  # type: ignore[union-attr]

    # The global branch must NOT have an organization_id condition (otherwise
    # we'd accidentally exclude global chunks).
    global_branch = by_scope["global"]
    assert _conditions_for_key(global_branch, "organization_id") == []


def test_tenant_filter_cannot_leak_other_orgs() -> None:
    """Property: changing the org_id changes the tenant branch's match value.

    A wrong org_id can never satisfy both visibility_scope=tenant_private AND
    organization_id=<requested_org> — that's the architectural guarantee.
    """
    org_a = uuid.uuid4()
    org_b = uuid.uuid4()
    f_a = build_tenant_filter(organization_id=org_a)
    f_b = build_tenant_filter(organization_id=org_b)

    def org_value(f: qm.Filter) -> str:
        for branch in f.should or []:
            assert isinstance(branch, qm.Filter)
            for c in _conditions_for_key(branch, "organization_id"):
                return c.match.value  # type: ignore[union-attr, no-any-return]
        raise AssertionError("tenant branch missing")

    assert org_value(f_a) != org_value(f_b)
    assert org_value(f_a) == str(org_a)
    assert org_value(f_b) == str(org_b)
