"""SSE event helpers for ``POST /ask?stream=true`` (Phase 5 — M7).

Streaming preserves the refusal contract: the LLM is still called once with
``response_format=json_object``, but the request handler emits SSE events
around the pipeline so the client can render progress / partial state
before the full structured answer arrives.

We deliberately do NOT stream raw LLM tokens. Token streaming would
require partial JSON parsing and risks emitting half-formed citations to
the client. Instead we emit events at well-defined milestones:

  - ``retrieval_started`` — before the retriever runs
  - ``sources_found``     — after retrieval (count + scope distribution)
  - ``reasoning``         — short status while the LLM call is in flight
  - ``direct_answer``     — once the structured payload is parsed
  - ``citations``         — list of structured citations
  - ``completed``         — full AskResponse mirror, terminal event
  - ``refused``           — refusal terminal event (replaces ``completed``)

Each event line follows the SSE wire format (``event:`` then ``data:`` then
``\n\n``). Data is JSON to keep the wire format simple for any SSE client.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class SSEEvent:
    event: str
    data: dict[str, Any]

    def encode(self) -> bytes:
        # Multi-line data fields are NOT supported here intentionally —
        # JSON-encoding keeps the payload single-line and unambiguous.
        payload = json.dumps(self.data, ensure_ascii=False, default=str)
        return f"event: {self.event}\ndata: {payload}\n\n".encode()


def encode_events(events: Iterable[SSEEvent]) -> bytes:
    """Concatenate multiple events into one bytestring (used by tests)."""
    return b"".join(e.encode() for e in events)


async def to_async(events: Iterable[SSEEvent]) -> AsyncIterator[bytes]:
    """Adapt a synchronous iterable into the async generator FastAPI's
    ``StreamingResponse`` needs."""
    for event in events:
        yield event.encode()
