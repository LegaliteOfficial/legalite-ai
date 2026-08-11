"""JSONStringFieldStreamer — incremental extraction of ``direct_answer``
from a partially-received JSON object (OpenAI streaming chat completion
with ``response_format={"type": "json_object"}``).
"""

from __future__ import annotations

from app.rag.stream_extractor import JSONStringFieldStreamer


def _feed_all(chunks: list[str]) -> tuple[str, bool]:
    s = JSONStringFieldStreamer()
    out = "".join(s.feed(c) for c in chunks)
    return out, s.done


def test_extracts_value_in_one_chunk() -> None:
    raw = '{"direct_answer": "Yes, notice is required.", "legal_reasoning": "..."}'
    text, done = _feed_all([raw])
    assert text == "Yes, notice is required."
    assert done


def test_extracts_value_streamed_one_character_at_a_time() -> None:
    raw = '{"direct_answer": "Yes, notice is required."}'
    text, done = _feed_all(list(raw))
    assert text == "Yes, notice is required."
    assert done


def test_handles_escape_sequences_split_across_chunks() -> None:
    raw = '{"direct_answer": "Line one\\nsays \\"hi\\"."}'
    # Split at every single character so any escape pair straddles chunks.
    text, done = _feed_all(list(raw))
    assert text == 'Line one\nsays "hi".'
    assert done


def test_handles_unicode_escape_split_across_chunks() -> None:
    raw = '{"direct_answer": "Caf\\u00e9 rules"}'
    text, done = _feed_all(list(raw))
    assert text == "Café rules"
    assert done


def test_key_never_found_yields_nothing_and_never_completes() -> None:
    text, done = _feed_all(['{"other_field": "hello"}'])
    assert text == ""
    assert not done


def test_stops_after_closing_quote_ignores_rest_of_object() -> None:
    raw = '{"direct_answer": "short"}, "legal_reasoning": "ignored downstream"'
    text, done = _feed_all([raw])
    assert text == "short"
    assert done


def test_key_substring_inside_the_value_does_not_confuse_the_scanner() -> None:
    raw = '{"direct_answer": "value with \\"direct_answer\\" inside"}'
    text, done = _feed_all([raw])
    assert text == 'value with "direct_answer" inside'
    assert done
