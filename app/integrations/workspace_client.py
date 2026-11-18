"""Read-only GraphQL client for the user's Legalite workspace.

The user's practice data (cases, clients, deadlines, tasks, documents,
calendar) lives behind the NestJS backend, which already owns
authentication, firm tenancy and permissions. Rather than reaching into
the database directly (and duplicating every access rule), the agent calls
that same GraphQL API **with the signed-in user's own bearer token**.

Consequences of that design:

- The agent can never see more than the user could see in the app. A
  forged or expired token is rejected by the backend, not by us.
- This service never stores or logs the token; it lives only for the
  duration of one /ask request.
- Every method here is a *read*. Writes (create case, save a draft, add a
  deadline) are proposed by the agent and executed by the frontend after
  the user confirms — see ``app/rag/workspace_tools.py``.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class WorkspaceError(RuntimeError):
    """The backend refused or failed a workspace query."""


_CASE_FIELDS = """
  id title case_code status case_stage case_type court suit_number
  client_id client_name opposing_party assigned_lawyer next_court_date
  date_opened description details notes updated_at
"""

_PROFILE_QUERY = """
query AgentProfile {
  me { id name first_name email role }
  currentFirm { name city firm_type office_address digital_address email }
  myPracticeAreas { name is_primary }
}
"""

_CASES_QUERY = f"query AgentCases {{ cases {{ {_CASE_FIELDS} }} }}"

_CASE_DETAIL_QUERY = f"""
query AgentCase($id: ID!) {{
  case(id: $id) {{ {_CASE_FIELDS} }}
  documents(caseId: $id) {{ id title template_type author created_at updated_at }}
  deadlines {{ id title due_date status priority case_id description }}
  tasks {{ id title due_date status priority case_id notes }}
}}
"""

_DEADLINES_QUERY = """
query AgentDeadlines($upcoming: Boolean, $from: String, $to: String) {
  deadlines(upcoming: $upcoming) {
    id title due_date status priority case_id case_title description
  }
  calendarEvents(from: $from, to: $to) {
    id title event_type start_time end_time status location case_id case_title
  }
}
"""

_TASKS_QUERY = """
query AgentTasks {
  tasks { id title due_date status priority case_id case_title client_name notes }
}
"""

_DOCUMENTS_QUERY = """
query AgentDocuments($caseId: ID, $clientId: ID) {
  documents(caseId: $caseId, clientId: $clientId) {
    id title template_type case_id case_title client_name author updated_at file_mime_type
  }
}
"""

_DOCUMENT_QUERY = """
query AgentDocument($id: ID!) {
  document(id: $id) {
    id title template_type case_id case_title client_name author content
    court judge parties suit_number updated_at file_url file_mime_type
  }
}
"""

_CLIENTS_QUERY = """
query AgentClients {
  clients { id full_name client_code contact_type organization email phone status }
}
"""

_OVERVIEW_QUERY = """
query AgentOverview {
  dashboardStats {
    active_cases pending_tasks total_clients total_invoices_due
  }
}
"""


class WorkspaceClient:
    """Thin, synchronous GraphQL client bound to one user's token."""

    def __init__(
        self,
        *,
        bearer_token: str,
        url: str | None = None,
        timeout: float | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        settings = get_settings()
        self._url = url or settings.legalite_backend_graphql_url or ""
        if not self._url:
            raise WorkspaceError("LEGALITE_BACKEND_GRAPHQL_URL is not configured")
        self._token = bearer_token
        self._http = http or httpx.Client(
            timeout=timeout or settings.workspace_request_timeout_seconds
        )
        self._profile_cache: dict[str, Any] | None = None

    # -- transport -----------------------------------------------------

    def query(self, document: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            res = self._http.post(
                self._url,
                json={"query": document, "variables": variables or {}},
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise WorkspaceError(f"Could not reach the Legalite backend: {exc}") from exc
        if res.status_code in (401, 403):
            raise WorkspaceError("The user's session is not authorised for this data.")
        if res.status_code >= 400:
            raise WorkspaceError(f"Backend returned HTTP {res.status_code}")
        body = res.json()
        errors = body.get("errors") or []
        data = body.get("data") or {}
        if errors and not any(v is not None for v in data.values()):
            msg = str(errors[0].get("message", "unknown error"))
            raise WorkspaceError(msg)
        if errors:
            # Partial data: keep what came back, note the rest.
            logger.info("workspace.partial_errors", count=len(errors))
        return data

    # -- reads ---------------------------------------------------------

    def profile(self) -> dict[str, Any]:
        if self._profile_cache is None:
            self._profile_cache = self.query(_PROFILE_QUERY)
        return self._profile_cache

    def cases(self) -> list[dict[str, Any]]:
        return list(self.query(_CASES_QUERY).get("cases") or [])

    def case_detail(self, case_id: str) -> dict[str, Any]:
        data = self.query(_CASE_DETAIL_QUERY, {"id": case_id})
        return {
            "case": data.get("case"),
            "documents": data.get("documents") or [],
            "deadlines": [d for d in data.get("deadlines") or [] if d.get("case_id") == case_id],
            "tasks": [t for t in data.get("tasks") or [] if t.get("case_id") == case_id],
        }

    def deadlines(
        self, *, upcoming: bool | None, date_from: str | None, date_to: str | None
    ) -> dict[str, Any]:
        return self.query(
            _DEADLINES_QUERY, {"upcoming": upcoming, "from": date_from, "to": date_to}
        )

    def tasks(self) -> list[dict[str, Any]]:
        return list(self.query(_TASKS_QUERY).get("tasks") or [])

    def documents(
        self, *, case_id: str | None = None, client_id: str | None = None
    ) -> list[dict[str, Any]]:
        data = self.query(_DOCUMENTS_QUERY, {"caseId": case_id, "clientId": client_id})
        return list(data.get("documents") or [])

    def document(self, document_id: str) -> dict[str, Any] | None:
        return self.query(_DOCUMENT_QUERY, {"id": document_id}).get("document")

    def clients(self) -> list[dict[str, Any]]:
        return list(self.query(_CLIENTS_QUERY).get("clients") or [])

    def overview(self) -> dict[str, Any]:
        return self.query(_OVERVIEW_QUERY).get("dashboardStats") or {}

    def close(self) -> None:
        self._http.close()


__all__ = ["WorkspaceClient", "WorkspaceError"]
