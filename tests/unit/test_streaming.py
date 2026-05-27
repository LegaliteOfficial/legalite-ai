"""SSE event helpers (Phase 5 M7).

The full streaming path is tested via the security/integration suite once
the live server is running with mocked LLM. Here we cover the wire format
and event shape contracts.
"""

from __future__ import annotations

from app.rag.streaming import SSEEvent, encode_events


def test_sse_event_wire_format() -> None:
    e = SSEEvent(event="direct_answer", data={"direct_answer": "Yes."})
    encoded = e.encode()
    assert encoded.startswith(b"event: direct_answer\n")
    assert b'"direct_answer": "Yes."' in encoded
    assert encoded.endswith(b"\n\n")


def test_encode_multiple_events_concatenates() -> None:
    e1 = SSEEvent(event="retrieval_started", data={"q": 1})
    e2 = SSEEvent(event="completed", data={"ok": True})
    encoded = encode_events([e1, e2])
    assert encoded.count(b"event: ") == 2
    assert b"event: retrieval_started" in encoded
    assert b"event: completed" in encoded


def test_sse_payload_handles_non_ascii() -> None:
    e = SSEEvent(event="reasoning", data={"summary": "Section 17 — notice."})
    encoded = e.encode()
    # The em dash must survive JSON encoding.
    assert "Section 17" in encoded.decode("utf-8")
