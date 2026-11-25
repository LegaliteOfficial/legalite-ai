# ruff: noqa: E501
"""Workspace consent end to end, plus the record-level tools.

- In "ask" mode the agent stops before its first workspace tool runs and
  the service returns a permission request; nothing is written to memory.
- Resending with a valid allow runs the turn; the request id is one-shot.
- Declining or "never" withholds the tools and tells the model why.
- "always" runs without asking.
- get_case / get_client / list_invoices and the edit proposals.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.rag.chat_agent import ChatAgent
from app.rag.workspace_consent import ConsentDecision, InMemoryConsentStore
from app.rag.workspace_tools import WORKSPACE_TOOL_NAMES, WorkspaceToolkit
from tests.unit.test_chat_agent import _drain, _FakeClient, _service, _text_round
from tests.unit.test_workspace_tools import TODAY, FakeWorkspace, _call_round


class Workspace(FakeWorkspace):
    def __init__(self, *, access: str = "ask", **kw: Any) -> None:
        super().__init__(**kw)
        self.access = access

    def workspace_access(self) -> str:
        self.calls.append("workspace_access")
        return self.access

    def client(self, client_id: str) -> dict[str, Any] | None:
        self._check("client")
        base = next((c for c in self.clients() if c["id"] == client_id), None)
        return {**base, "email": "kofi@example.com", "phone": "020 000 0000"} if base else None

    def case_team(self, case_id: str) -> list[dict[str, Any]]:
        self._check("case_team")
        return [{"member_id": "m1", "name": "Abena Boateng", "assignment_role": "responsible"}]

    def invoices(self) -> list[dict[str, Any]]:
        self._check("invoices")
        return [
            {
                "id": "inv-1",
                "client_id": "cl-1",
                "client_name": "Kofi Mensah",
                "amount_ghs": 2500,
                "status": "Sent",
                "due_date": "2026-12-01",
            },
            {
                "id": "inv-2",
                "client_id": "cl-2",
                "client_name": "Ama Asante",
                "amount_ghs": 900,
                "status": "Paid",
            },
        ]

    def documents(self, **kw: Any) -> list[dict[str, Any]]:
        self._check("documents")
        return [
            {
                "id": "doc-9",
                "title": "Statement of claim",
                "case_id": "case-1",
                "case_title": "Mensah v Owusu (land boundary)",
            },
            {"id": "doc-10", "title": "Site plan", "case_id": None},
        ]


def _kit(**kw: Any) -> tuple[WorkspaceToolkit, Workspace]:
    ws = Workspace(**kw)
    return WorkspaceToolkit(ws, today=TODAY), ws  # type: ignore[arg-type]


def _data_calls(ws: Workspace) -> list[str]:
    return [c for c in ws.calls if c not in ("profile", "workspace_access")]


# ---- tools ---------------------------------------------------------------------


def test_new_tools_are_registered() -> None:
    assert {
        "get_client",
        "list_invoices",
        "propose_case_update",
        "propose_client_update",
        "propose_link_documents",
    } <= WORKSPACE_TOOL_NAMES


def test_get_case_names_client_team_and_documents() -> None:
    kit, _ = _kit()
    out = kit.execute("get_case", {"case_id": "LD/123/2026"}).content
    assert "client_name: Kofi Mensah" in out
    assert "Abena Boateng (responsible)" in out
    assert "Statement of claim" in out


def test_get_client_lists_cases_documents_and_invoices() -> None:
    kit, _ = _kit()
    out = kit.execute("get_client", {"client_id": "Kofi Mensah"}).content
    assert "client_id: cl-1" in out and "kofi@example.com" in out
    assert "Mensah v Owusu" in out
    assert "GHS 2,500.00 [Sent]" in out and "GHS 900.00" not in out


def test_list_invoices_filters_and_totals() -> None:
    kit, _ = _kit()
    out = kit.execute("list_invoices", {"status": "paid"}).content
    assert out.startswith("1 invoice(s), total GHS 900.00")


def test_close_case_proposal_has_diff_and_never_writes() -> None:
    kit, ws = _kit()
    outcome = kit.execute(
        "propose_case_update", {"case_id": "Mensah v Owusu (land boundary)", "status": "closed"}
    )
    action = outcome.proposal
    assert action is not None and action.type == "update_case"
    assert action.summary == "Close case “Mensah v Owusu (land boundary)”"
    assert action.payload == {"id": "case-1", "input": {"status": "Closed"}}
    assert action.changes == [
        {"field": "status", "label": "Status", "from": "Open", "to": "Closed"}
    ]
    assert action.to_dict()["changes"]
    assert all(not c.startswith(("update", "create")) for c in ws.calls)


def test_case_update_skips_unchanged_and_rejects_no_op() -> None:
    kit, _ = _kit()
    out = kit.execute(
        "propose_case_update",
        {"case_id": "case-1", "status": "Open", "court": "High Court, Kumasi"},
    )
    assert out.proposal is None and "nothing would change" in out.content
    moved = kit.execute(
        "propose_case_update",
        {
            "case_id": "case-1",
            "court": "Court of Appeal",
            "next_court_date": "2026-12-03",
            "client_id": "Ama Asante",
        },
    ).proposal
    assert moved is not None
    assert moved.payload["input"] == {
        "court": "Court of Appeal",
        "next_court_date": "2026-12-03",
        "client_id": "cl-2",
    }
    assert [c["label"] for c in moved.changes] == ["Court", "Next court date", "Client"]


def test_client_update_and_link_documents() -> None:
    kit, _ = _kit()
    upd = kit.execute(
        "propose_client_update",
        {"client_id": "cl-1", "phone": "024 111 2222", "email": "kofi@example.com"},
    ).proposal
    assert upd is not None and upd.payload == {"id": "cl-1", "input": {"phone": "024 111 2222"}}
    link = kit.execute(
        "propose_link_documents", {"case_id": "case-1", "document_ids": ["doc-9", "doc-10"]}
    ).proposal
    assert link is not None and link.type == "link_documents"
    assert link.payload == {"case_id": "case-1", "document_ids": ["doc-10"]}
    assert link.changes[0]["from"] == "(no case)"
    missing = kit.execute("propose_link_documents", {"case_id": "case-1", "document_ids": ["nope"]})
    assert missing.proposal is None and "not found" in missing.content


def test_profile_without_tools_does_not_claim_access() -> None:
    kit, _ = _kit()
    text = kit.profile_context(tools_available=False)
    assert text is not None and "Abena Boateng" in text and "tools are available" not in text


# ---- agent gate --------------------------------------------------------------


def test_agent_pauses_before_first_workspace_tool() -> None:
    kit, ws = _kit()
    client = _FakeClient([_call_round("find_cases", {"query": "Mensah"})])
    agent = ChatAgent(client=client, model="gpt-test")
    _, result = _drain(
        agent,
        "which client is the Mensah case for?",
        search_fn=MagicMock(),
        workspace=kit,
        workspace_access="ask",
    )
    assert result.mode == "permission_required"
    assert result.permission_scopes == ["Search your cases for “Mensah”"]
    assert result.permission_tools == ["find_cases"]
    assert "cases" not in ws.calls
    assert len(client.requests) == 1


# ---- service flow ------------------------------------------------------------


def _svc(rounds: list[Any], access: str = "ask") -> tuple[Any, Any, Workspace, _FakeClient]:
    client = _FakeClient(rounds)
    svc, mem, _ = _service(client, [])
    kit, ws = _kit(access=access)
    svc._workspace_for = lambda token: kit if token else None  # type: ignore[method-assign]
    svc._consent_store = InMemoryConsentStore()
    return svc, mem, ws, client


Q = "Which client is the Mensah case for?"


def test_service_pauses_then_resumes_with_one_shot_consent() -> None:
    svc, mem, ws, client = _svc(
        [
            _call_round("find_cases", {"query": "Mensah"}),
            _call_round("find_cases", {"query": "Mensah"}),
            _text_round("It is for Kofi Mensah."),
            _call_round("find_cases", {"query": "Mensah"}),
        ]
    )
    events = list(svc.ask_stream(Q, user_token="tok"))
    names = [e.event for e in events]
    assert "permission_required" in names and names[-1] == "completed"
    done = events[-1].data
    assert done["answer_mode"] == "permission_required"
    req = done["permission_request"]
    assert req["scopes"] == ["Search your cases for “Mensah”"] and req["request_id"].startswith(
        "wsc_"
    )
    assert mem.messages == []  # nothing remembered for a paused turn
    assert _data_calls(ws) == []
    sid = done["session_id"]

    resp = svc.ask(
        Q, sid, user_token="tok", workspace_consent=ConsentDecision("allow", req["request_id"])
    )
    assert resp.answer == "It is for Kofi Mensah."
    assert resp.workspace_access == "granted"
    assert "cases" in ws.calls
    assert [m[0] for m in mem.messages] == ["user", "assistant"]

    # The same request id cannot be replayed.
    again = svc.ask(
        Q, sid, user_token="tok", workspace_consent=ConsentDecision("allow", req["request_id"])
    )
    assert again.answer_mode == "permission_required"


def test_consent_is_bound_to_the_question() -> None:
    svc, _, ws, _ = _svc(
        [_call_round("find_cases", {"query": "Mensah"}), _call_round("list_invoices", {})]
    )
    first = svc.ask(Q, user_token="tok")
    assert first.permission_request is not None
    other = svc.ask(
        "What do my clients owe me?",
        first.session_id,
        user_token="tok",
        workspace_consent=ConsentDecision("allow", first.permission_request.request_id),
    )
    assert other.answer_mode == "permission_required"
    assert _data_calls(ws) == []


def test_deny_withholds_tools_and_explains() -> None:
    svc, mem, ws, client = _svc([_text_round("I did not look at your records.")])
    resp = svc.ask(Q, user_token="tok", workspace_consent=ConsentDecision("deny", "anything"))
    assert resp.workspace_access == "denied"
    offered = {t["function"]["name"] for t in client.requests[0]["tools"]}
    assert "find_cases" not in offered
    assert "declined access" in client.requests[0]["messages"][1]["content"]
    assert _data_calls(ws) == []
    assert [m[0] for m in mem.messages] == ["user", "assistant"]


@pytest.mark.parametrize(
    ("access", "expected"),
    [("never", "disabled"), ("always", "granted")],
)
def test_standing_preference(access: str, expected: str) -> None:
    rounds = (
        [_text_round("ok")]
        if access == "never"
        else [_call_round("find_cases", {"query": "Mensah"}), _text_round("Kofi Mensah.")]
    )
    svc, _, ws, client = _svc(rounds, access=access)
    resp = svc.ask(Q, user_token="tok")
    assert resp.workspace_access == expected
    if access == "never":
        assert "turned off" in client.requests[0]["messages"][1]["content"]
        assert _data_calls(ws) == []
    else:
        assert "cases" in ws.calls


def test_anonymous_caller_is_unavailable() -> None:
    svc, _, _, client = _svc([_text_round("Please sign in.")])
    resp = svc.ask(Q)
    assert resp.workspace_access == "unavailable"
    assert "not signed in" in client.requests[0]["messages"][1]["content"]
