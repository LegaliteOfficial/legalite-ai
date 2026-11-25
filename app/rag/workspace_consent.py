"""Consent gate for the user's workspace.

Ask Legalite can read a signed-in user's cases, clients, deadlines, tasks,
documents and invoices. It does so only with the user's permission:

``ask`` (default)
    Every request that needs workspace data pauses before the first
    workspace tool runs. The client shows a permission card listing what
    the assistant wants to look at; the user allows or declines, and the
    client resends the same question with that decision. An allow covers
    that one request only.
``always``
    The user turned on "Always allow" (in Settings, or from the card).
    Workspace tools run without asking.
``never``
    The user turned workspace access off. The tools are not offered to
    the model at all; it answers from general knowledge and says how to
    turn access back on.

The standing preference lives in the backend (``aiWorkspaceAccess``) and
is read with the user's own token, so it follows them across devices. An
allow is a one-shot ``request_id`` this service issues when it pauses:
bound to the chat session, the user and the exact question, valid for
15 minutes, and spent on first use. A client cannot mint one, replay one,
or carry one over to a different question.

Proposals (create / update records) are unaffected: whatever the access
mode, nothing is written until the user confirms the card for that change.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from app.core.logging import get_logger

if TYPE_CHECKING:
    import uuid

logger = get_logger(__name__)

# How the agent may treat the workspace on this request.
#   granted     - tools are offered and run
#   ask         - tools are offered; the first call pauses for permission
#   denied      - the user declined for this request; tools are withheld
#   disabled    - the user turned access off in Settings; tools are withheld
#   unavailable - anonymous caller or integration off; tools are withheld
WorkspaceAccess = Literal["granted", "ask", "denied", "disabled", "unavailable"]
Preference = Literal["ask", "always", "never"]

REQUEST_TTL_SECONDS = 15 * 60
MAX_PENDING_PER_SESSION = 5
MAX_SCOPES = 12
MAX_SCOPE_CHARS = 200
_META_KEY = "workspace_consent"


@dataclass(slots=True)
class ConsentDecision:
    """The user's answer to a permission card, sent with the resend."""

    decision: Literal["allow", "deny"]
    request_id: str


@dataclass(slots=True)
class PermissionRequest:
    """What the client needs to render the permission card."""

    request_id: str
    scopes: list[str]
    tools: list[str]
    expires_at: int
    reason: str = "workspace"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "scopes": self.scopes,
            "tools": self.tools,
            "expires_at": self.expires_at,
            "reason": self.reason,
        }


@dataclass(slots=True)
class AccessResolution:
    access: WorkspaceAccess
    preference: Preference = "ask"
    # Set when the user answered a card on this request.
    decision: Literal["allow", "deny"] | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_access(
    *,
    workspace_available: bool,
    preference: Preference,
    decision: ConsentDecision | None,
    redeemed: bool,
) -> AccessResolution:
    """Pure decision table. ``redeemed`` says whether ``decision.request_id``
    was a live, matching, unused request (already spent by the caller)."""
    if not workspace_available:
        return AccessResolution("unavailable", preference)
    if preference == "never":
        # A stale "Allow" card does not override turning access off.
        return AccessResolution("disabled", preference, decision.decision if decision else None)
    if decision is not None and decision.decision == "deny":
        # Declining needs no proof: it only ever narrows access.
        return AccessResolution("denied", preference, "deny")
    if decision is not None and decision.decision == "allow":
        if redeemed:
            return AccessResolution("granted", preference, "allow")
        if preference == "always":
            return AccessResolution("granted", preference, "allow")
        return AccessResolution(
            "ask", preference, None, ["consent request expired, unknown or already used"]
        )
    if preference == "always":
        return AccessResolution("granted", preference)
    return AccessResolution("ask", preference)


# ---------------------------------------------------------------------------
# Scope descriptions (shown on the permission card)
# ---------------------------------------------------------------------------


def _q(args: dict[str, Any], key: str = "query") -> str:
    v = str(args.get(key) or "").strip()
    return v[:80]


def describe_tool_call(name: str, args: dict[str, Any]) -> str:
    """Plain-language line for one workspace tool call."""
    q = _q(args)
    match name:
        case "workspace_overview":
            return "See counts of your cases, tasks, clients and invoices"
        case "find_cases":
            return f"Search your cases for “{q}”" if q else "List your cases"
        case "get_case":
            return "Open a case with its client, team, documents, deadlines and tasks"
        case "find_clients":
            return f"Search your clients for “{q}”" if q else "List your clients"
        case "get_client":
            return "Open a client with their cases, documents and invoices"
        case "list_deadlines":
            days = args.get("days_ahead")
            window = f"the next {days} days" if isinstance(days, int) else "the coming days"
            return f"Read your deadlines, tasks due and calendar for {window}"
        case "list_tasks":
            return "Read your tasks"
        case "find_documents":
            return f"Search your documents for “{q}”" if q else "List your documents"
        case "read_document":
            return "Read the text of one of your documents"
        case "search_archive":
            return f"Search the firm archive for “{q}”" if q else "Search the firm archive"
        case "read_archive_entry":
            return "Read one document from the firm archive"
        case "list_invoices":
            return "Read your invoices"
        case _ if name.startswith("propose_"):
            return (
                "Look up records to prepare a change for you to review "
                "(nothing is saved until you confirm)"
            )
        case _:
            return "Read your workspace"


def describe_calls(calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    seen: list[str] = []
    for name, args in calls:
        line = describe_tool_call(name, args)[:MAX_SCOPE_CHARS]
        if line not in seen:
            seen.append(line)
    return seen[:MAX_SCOPES]


# ---------------------------------------------------------------------------
# One-shot request store
# ---------------------------------------------------------------------------


def question_fingerprint(question: str) -> str:
    norm = " ".join((question or "").lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


def token_subject(token: str | None) -> str:
    """Stable id for the caller. The JWT ``sub`` when the token carries one
    (so a refreshed token still matches), else a hash of the token. The
    backend verifies the token; this only binds a request to its caller."""
    if not token:
        return "anonymous"
    parts = token.split(".")
    if len(parts) == 3:
        try:
            pad = "=" * (-len(parts[1]) % 4)
            claims = json.loads(base64.urlsafe_b64decode(parts[1] + pad))
            sub = claims.get("sub") if isinstance(claims, dict) else None
            if isinstance(sub, str) and sub:
                return "sub:" + hashlib.sha256(sub.encode()).hexdigest()[:24]
        except (ValueError, json.JSONDecodeError):
            pass
    return "tok:" + hashlib.sha256(token.encode()).hexdigest()[:24]


class ConsentStore(Protocol):
    def issue(self, sid: uuid.UUID, *, subject: str, question: str) -> tuple[str, int]: ...

    def redeem(
        self, sid: uuid.UUID, request_id: str, *, subject: str, question: str
    ) -> bool: ...


class _MetaBackend(Protocol):
    def get_meta(self, session_id: uuid.UUID) -> dict: ...  # type: ignore[type-arg]

    def set_meta(self, session_id: uuid.UUID, meta: dict) -> None: ...  # type: ignore[type-arg]


class SessionConsentStore:
    """Pending requests kept on the chat session's metadata."""

    def __init__(self, backend: _MetaBackend, *, clock: Any = time.time) -> None:
        self._backend = backend
        self._clock = clock

    def _pending(self, meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
        now = int(self._clock())
        raw = meta.get(_META_KEY) or {}
        return {
            k: v
            for k, v in raw.items()
            if isinstance(v, dict) and int(v.get("exp") or 0) > now
        }

    def issue(self, sid: uuid.UUID, *, subject: str, question: str) -> tuple[str, int]:
        meta = self._backend.get_meta(sid)
        pending = self._pending(meta)
        if len(pending) >= MAX_PENDING_PER_SESSION:
            oldest = sorted(pending, key=lambda k: pending[k]["exp"])
            for k in oldest[: len(pending) - MAX_PENDING_PER_SESSION + 1]:
                pending.pop(k)
        request_id = "wsc_" + secrets.token_urlsafe(18)
        exp = int(self._clock()) + REQUEST_TTL_SECONDS
        pending[request_id] = {"sub": subject, "q": question_fingerprint(question), "exp": exp}
        meta[_META_KEY] = pending
        self._backend.set_meta(sid, meta)
        return request_id, exp

    def redeem(
        self, sid: uuid.UUID, request_id: str, *, subject: str, question: str
    ) -> bool:
        meta = self._backend.get_meta(sid)
        pending = self._pending(meta)
        entry = pending.pop(request_id, None)
        meta[_META_KEY] = pending
        self._backend.set_meta(sid, meta)
        if entry is None:
            return False
        ok = entry.get("sub") == subject and entry.get("q") == question_fingerprint(question)
        if not ok:
            logger.info("workspace_consent.mismatch", request_id=request_id)
        return ok


class InMemoryConsentStore(SessionConsentStore):
    """Process-local store (tests, and the fallback when memory is off)."""

    def __init__(self, *, clock: Any = time.time) -> None:
        self._metas: dict[uuid.UUID, dict[str, Any]] = {}

        store = self

        class _Backend:
            def get_meta(self, session_id: uuid.UUID) -> dict:  # type: ignore[type-arg]
                return dict(store._metas.get(session_id) or {})

            def set_meta(self, session_id: uuid.UUID, meta: dict) -> None:  # type: ignore[type-arg]
                store._metas[session_id] = dict(meta)

        super().__init__(_Backend(), clock=clock)


__all__ = [
    "REQUEST_TTL_SECONDS",
    "AccessResolution",
    "ConsentDecision",
    "ConsentStore",
    "InMemoryConsentStore",
    "PermissionRequest",
    "SessionConsentStore",
    "WorkspaceAccess",
    "describe_calls",
    "describe_tool_call",
    "question_fingerprint",
    "resolve_access",
    "token_subject",
]
