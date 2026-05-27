"""Tenant isolation evaluation suite (Phase 5 — M3).

Hard requirement: cross-tenant chunk leakage is a P0 security bug. This
package ships a runner that asserts ``tenant_isolation_score == 1.0`` and
a set of scenarios that should fail the build if any boundary regresses.
"""
