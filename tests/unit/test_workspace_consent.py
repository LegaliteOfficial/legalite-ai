# ruff: noqa: E501
"""Workspace consent: decision table and one-shot request store."""

from __future__ import annotations

import uuid

from app.rag.workspace_consent import (
    ConsentDecision,
    InMemoryConsentStore,
    describe_calls,
    resolve_access,
    token_subject,
)


def _r(pref, decision=None, redeemed=False, available=True):  # type: ignore[no-untyped-def]
    return resolve_access(
        workspace_available=available, preference=pref, decision=decision, redeemed=redeemed
    ).access


def test_decision_table() -> None:
    allow = ConsentDecision("allow", "x")
    deny = ConsentDecision("deny", "x")
    assert _r("ask", available=False) == "unavailable"
    assert _r("ask") == "ask"
    assert _r("always") == "granted"
    assert _r("never") == "disabled"
    assert _r("never", allow, redeemed=True) == "disabled"
    assert _r("ask", allow, redeemed=True) == "granted"
    assert _r("ask", allow, redeemed=False) == "ask"
    assert _r("always", deny) == "denied"


def test_request_is_one_shot_and_bound_to_question_and_user() -> None:
    now = [1000.0]
    store = InMemoryConsentStore(clock=lambda: now[0])
    sid = uuid.uuid4()
    rid, exp = store.issue(sid, subject="u1", question="Close the Mensah case")
    assert exp == 1000 + 15 * 60
    assert not store.redeem(sid, rid, subject="u2", question="Close the Mensah case")
    rid, _ = store.issue(sid, subject="u1", question="Close the Mensah case")
    assert not store.redeem(sid, rid, subject="u1", question="Something else")
    rid, _ = store.issue(sid, subject="u1", question="Close the Mensah case")
    assert store.redeem(sid, rid, subject="u1", question="close  the mensah case")
    assert not store.redeem(sid, rid, subject="u1", question="Close the Mensah case")
    rid, _ = store.issue(sid, subject="u1", question="q")
    now[0] += 16 * 60
    assert not store.redeem(sid, rid, subject="u1", question="q")


def test_scopes_and_subject() -> None:
    scopes = describe_calls([("find_cases", {"query": "Mensah"}), ("find_cases", {"query": "Mensah"})])
    assert scopes == ["Search your cases for “Mensah”"]
    assert token_subject(None) == "anonymous"
    assert token_subject("abc").startswith("tok:")
