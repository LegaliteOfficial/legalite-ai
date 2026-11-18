# ruff: noqa: E501
# (fixture rows are clearer on one line)
"""Workspace tools — the agent working with the user's own practice data.

Covers:
- read tools render the user's cases / deadlines / documents for the model;
- propose_* tools validate and normalise but never write;
- backend failures degrade to a message instead of killing the turn;
- ChatAgent wires workspace tools in, streams ``action`` statuses, returns
  proposals on the result, and labels the answer ``workspace`` mode;
- without a token the agent is told it cannot see the workspace.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any
from unittest.mock import MagicMock

from app.integrations.workspace_client import WorkspaceError
from app.rag.chat_agent import AgentStatus, ChatAgent
from app.rag.workspace_tools import WORKSPACE_TOOL_NAMES, WorkspaceToolkit
from tests.unit.test_chat_agent import (
    _Choice,
    _Chunk,
    _Delta,
    _drain,
    _FakeClient,
    _Fn,
    _text_round,
    _ToolCallDelta,
    _Usage,
)

TODAY = date(2026, 11, 18)

CASES = [
    {
        "id": "case-1",
        "title": "Mensah v Owusu (land boundary)",
        "client_id": "cl-1",
        "client_name": "Kofi Mensah",
        "status": "Open",
        "case_type": "Land",
        "court": "High Court, Kumasi",
        "suit_number": "LD/123/2026",
        "updated_at": "2026-11-10",
    },
    {
        "id": "case-2",
        "title": "Asante employment dispute",
        "client_id": "cl-2",
        "client_name": "Ama Asante",
        "status": "Closed",
        "case_type": "Employment",
        "updated_at": "2026-09-01",
    },
]
CLIENTS = [
    {"id": "cl-1", "full_name": "Kofi Mensah", "contact_type": "person", "status": "Active"},
    {"id": "cl-2", "full_name": "Ama Asante", "contact_type": "person", "status": "Active"},
]


class FakeWorkspace:
    """Duck-typed stand-in for WorkspaceClient."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def _check(self, name: str) -> None:
        self.calls.append(name)
        if self.fail:
            raise WorkspaceError("backend down")

    def profile(self) -> dict[str, Any]:
        self._check("profile")
        return {
            "me": {"name": "Abena Boateng", "role": "lawyer"},
            "currentFirm": {"name": "Boateng & Co", "city": "Kumasi"},
            "myPracticeAreas": [
                {"name": "Land Law", "is_primary": True},
                {"name": "Employment", "is_primary": False},
            ],
        }

    def cases(self) -> list[dict[str, Any]]:
        self._check("cases")
        return CASES

    def clients(self) -> list[dict[str, Any]]:
        self._check("clients")
        return CLIENTS

    def case_detail(self, case_id: str) -> dict[str, Any]:
        self._check("case_detail")
        return {
            "case": next((c for c in CASES if c["id"] == case_id), None),
            "documents": [
                {"id": "doc-9", "title": "Statement of claim", "template_type": "Pleading"}
            ],
            "deadlines": [
                {
                    "id": "d1",
                    "title": "File reply",
                    "due_date": "2026-11-20",
                    "status": "Pending",
                    "priority": "High",
                }
            ],
            "tasks": [],
        }

    def deadlines(self, **_: Any) -> dict[str, Any]:
        self._check("deadlines")
        return {
            "deadlines": [
                {
                    "id": "d0",
                    "title": "Overdue filing",
                    "due_date": "2026-11-15",
                    "status": "Pending",
                    "priority": "High",
                },
                {
                    "id": "d1",
                    "title": "File reply",
                    "due_date": "2026-11-20",
                    "status": "Pending",
                    "priority": "High",
                    "case_title": "Mensah v Owusu",
                },
                {"id": "d2", "title": "Far away", "due_date": "2027-06-01", "status": "Pending"},
                {"id": "d3", "title": "Already done", "due_date": "2026-11-19", "status": "Done"},
            ],
            "calendarEvents": [
                {
                    "id": "e1",
                    "title": "Hearing",
                    "event_type": "court",
                    "start_time": "2026-11-21T09:00:00Z",
                    "status": "scheduled",
                },
                {
                    "id": "e2",
                    "title": "Cancelled mtg",
                    "start_time": "2026-11-22T09:00:00Z",
                    "status": "cancelled",
                },
            ],
        }

    def tasks(self) -> list[dict[str, Any]]:
        self._check("tasks")
        return [
            {
                "id": "t1",
                "title": "Call client",
                "status": "Pending",
                "priority": "High",
                "due_date": "2026-11-19",
            },
            {"id": "t2", "title": "Old task", "status": "Done"},
        ]

    def documents(self, **_: Any) -> list[dict[str, Any]]:
        self._check("documents")
        return [{"id": "doc-9", "title": "Statement of claim", "case_title": "Mensah v Owusu"}]

    def document(self, document_id: str) -> dict[str, Any] | None:
        self._check("document")
        return {
            "id": document_id,
            "title": "Statement of claim",
            "content": "<p>The plaintiff claims&nbsp;title.</p><p>Second para</p>",
        }

    def overview(self) -> dict[str, Any]:
        self._check("overview")
        return {"active_cases": 4, "pending_tasks": 2, "total_clients": 9, "total_invoices_due": 1}


def _kit(**kw: Any) -> WorkspaceToolkit:
    return WorkspaceToolkit(FakeWorkspace(**kw), today=TODAY)  # type: ignore[arg-type]


# ---- read tools ---------------------------------------------------------------


def test_profile_context_personalises() -> None:
    text = _kit().profile_context()
    assert text is not None
    assert "Abena Boateng" in text and "Boateng & Co" in text and "Land Law" in text


def test_profile_context_none_when_backend_down() -> None:
    assert _kit(fail=True).profile_context() is None


def test_find_cases_ranks_by_query_and_filters_status() -> None:
    out = _kit().execute("find_cases", {"query": "my land case"})
    assert "case_id=case-1" in out.content
    assert "case-2" not in out.content
    closed = _kit().execute("find_cases", {"query": "", "status": "Closed"})
    assert "case-2" in closed.content and "case-1" not in closed.content


def test_get_case_includes_documents_and_deadlines() -> None:
    out = _kit().execute("get_case", {"case_id": "case-1"})
    assert "LD/123/2026" in out.content
    assert "document_id=doc-9" in out.content
    assert "File reply" in out.content and "(in 2 days)" in out.content


def test_list_deadlines_window_overdue_and_cancelled() -> None:
    out = _kit().execute("list_deadlines", {"days_ahead": 7})
    assert "OVERDUE by 3 days" in out.content
    assert "File reply" in out.content
    assert "Far away" not in out.content
    assert "Already done" not in out.content
    assert "Hearing" in out.content
    assert "Cancelled mtg" not in out.content


def test_list_tasks_open_only_by_default() -> None:
    out = _kit().execute("list_tasks", {})
    assert "Call client" in out.content and "Old task" not in out.content


def test_read_document_strips_html() -> None:
    out = _kit().execute("read_document", {"document_id": "doc-9"})
    assert (
        "The plaintiff claims\xa0title." in out.content
        or "The plaintiff claims title." in out.content
    )
    assert "<p>" not in out.content


def test_backend_failure_is_a_message_not_an_exception() -> None:
    out = _kit(fail=True).execute("find_cases", {"query": "x"})
    assert "could not be read" in out.content
    assert out.proposal is None


# ---- proposals ------------------------------------------------------------------


def test_propose_deadline_normalises_and_resolves_case() -> None:
    out = _kit().execute(
        "propose_deadline",
        {
            "title": "File defence",
            "due_date": "2026-12-02",
            "case_id": "case-1",
            "priority": "high",
        },
    )
    assert out.proposal is not None
    p = out.proposal
    assert p.type == "create_deadline"
    assert p.payload == {
        "title": "File defence",
        "due_date": "2026-12-02",
        "priority": "High",
        "status": "Pending",
        "reminder_days": 3,
        "case_id": "case-1",
    }
    assert p.display["case"].startswith("Mensah v Owusu")
    assert "NOT been created" in out.content


def test_propose_rejects_bad_date_and_unknown_case() -> None:
    bad_date = _kit().execute("propose_deadline", {"title": "x", "due_date": "next friday"})
    assert bad_date.proposal is None and "YYYY-MM-DD" in bad_date.content
    bad_case = _kit().execute("propose_task", {"title": "x", "case_id": "nope"})
    assert bad_case.proposal is None and "not found" in bad_case.content


def test_propose_case_needs_a_client() -> None:
    out = _kit().execute("propose_case", {"title": "New matter"})
    assert out.proposal is None
    ok = _kit().execute("propose_case", {"title": "New matter", "new_client_name": "Yaw Darko"})
    assert ok.proposal is not None
    assert ok.proposal.payload["new_client"] == {"full_name": "Yaw Darko", "contact_type": "person"}
    assert ok.proposal.payload["date_opened"] == "2026-11-18"
    existing = _kit().execute("propose_case", {"title": "M2", "client_id": "cl-2"})
    assert existing.proposal is not None
    assert existing.proposal.display["client"] == "Ama Asante"


def test_propose_document_marks_markdown() -> None:
    out = _kit().execute(
        "propose_document",
        {
            "title": "Demand letter",
            "content": "# Demand\n\nPay.",
            "case_id": "case-1",
            "document_type": "Letter",
        },
    )
    assert out.proposal is not None
    assert out.proposal.payload["content_format"] == "markdown"
    assert out.proposal.payload["template_type"] == "Letter"


def test_every_schema_has_a_handler() -> None:
    kit = _kit()
    for name in WORKSPACE_TOOL_NAMES:
        assert hasattr(kit, f"_t_{name}"), name


# ---- agent integration ------------------------------------------------------------


def _call_round(name: str, args: dict[str, Any], idx: int = 0) -> list[_Chunk]:
    return [
        _Chunk(
            choices=[
                _Choice(
                    _Delta(tool_calls=[_ToolCallDelta(idx, f"c{idx}", _Fn(name, json.dumps(args)))])
                )
            ]
        ),
        _Chunk(usage=_Usage()),
    ]


def test_agent_uses_workspace_and_returns_proposal() -> None:
    client = _FakeClient(
        [
            _call_round("find_cases", {"query": "Mensah"}),
            _call_round(
                "propose_deadline",
                {"title": "File reply", "due_date": "2026-11-25", "case_id": "case-1"},
            ),
            _text_round("I've proposed a deadline for 25 November — confirm it below."),
        ]
    )
    agent = ChatAgent(client=client, model="gpt-test")
    items, result = _drain(
        agent,
        "add a deadline to the Mensah case for next tuesday",
        search_fn=MagicMock(),
        workspace=_kit(),
    )

    # Workspace tools offered alongside the library tools.
    offered = {t["function"]["name"] for t in client.requests[0]["tools"]}
    assert {"search_legal_library", "find_cases", "propose_document"} <= offered
    # Profile personalisation in the context message.
    assert "Abena Boateng" in client.requests[0]["messages"][1]["content"]
    # Statuses: a workspace line and an action proposal.
    kinds = [i.kind for i in items if isinstance(i, AgentStatus)]
    assert "workspace" in kinds and "action" in kinds
    assert len(result.proposed_actions) == 1
    assert result.proposed_actions[0].payload["case_id"] == "case-1"
    assert result.mode == "workspace"
    assert result.confidence == "high"


def test_agent_without_workspace_is_told_it_cannot_see_it() -> None:
    client = _FakeClient([_text_round("You need to sign in.")])
    agent = ChatAgent(client=client, model="gpt-test")
    _drain(agent, "what are my deadlines?", search_fn=MagicMock())
    offered = {t["function"]["name"] for t in client.requests[0]["tools"]}
    assert "find_cases" not in offered
    assert "not signed in" in client.requests[0]["messages"][1]["content"]
