"""Incremental extractor for one string field inside a partially-received
JSON object, as produced by an OpenAI streaming chat completion with
``response_format={"type": "json_object"}``.

We only need ``direct_answer`` — the first field in the schema (see
``prompt_builder.STRUCTURED_SCHEMA_EXAMPLE``) — streamed to the client as
prose while the model is still generating the rest of the structured JSON
(citations, legal_reasoning, etc.), which the client never sees
incrementally. This is deliberately narrow: it is not a general JSON
parser. It looks for the field's key, skips to the opening quote of its
string value, then unescapes and yields characters until the closing
(unescaped) quote. Everything before or after that value is discarded.

If the key never appears (e.g. malformed model output), ``feed`` simply
yields nothing for every chunk; the caller falls back to the fully
buffered raw text once the stream ends — see
``AnswerGenerator.generate_stream``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from enum import Enum, auto

_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class _State(Enum):
    SEEKING_KEY = auto()
    SEEKING_COLON = auto()
    SEEKING_OPEN_QUOTE = auto()
    IN_STRING = auto()
    IN_ESCAPE = auto()
    IN_UNICODE_ESCAPE = auto()
    DONE = auto()


@dataclass
class JSONStringFieldStreamer:
    """Feed raw text chunks in; get back decoded characters of the target
    field's string value as they become available."""

    key: str = '"direct_answer"'
    _state: _State = field(default=_State.SEEKING_KEY, init=False)
    _match_pos: int = field(default=0, init=False)
    _unicode_buf: str = field(default="", init=False)

    def feed(self, chunk: str) -> str:
        out: list[str] = []
        for ch in chunk:
            if self._state is _State.DONE:
                break
            if self._state is _State.SEEKING_KEY:
                self._advance_key_match(ch)
            elif self._state is _State.SEEKING_COLON:
                if ch == ":":
                    self._state = _State.SEEKING_OPEN_QUOTE
            elif self._state is _State.SEEKING_OPEN_QUOTE:
                if ch == '"':
                    self._state = _State.IN_STRING
            elif self._state is _State.IN_STRING:
                if ch == "\\":
                    self._state = _State.IN_ESCAPE
                elif ch == '"':
                    self._state = _State.DONE
                else:
                    out.append(ch)
            elif self._state is _State.IN_ESCAPE:
                if ch == "u":
                    self._state = _State.IN_UNICODE_ESCAPE
                    self._unicode_buf = ""
                else:
                    out.append(_ESCAPES.get(ch, ch))
                    self._state = _State.IN_STRING
            elif self._state is _State.IN_UNICODE_ESCAPE:
                self._unicode_buf += ch
                if len(self._unicode_buf) == 4:
                    with contextlib.suppress(ValueError):
                        out.append(chr(int(self._unicode_buf, 16)))
                    self._state = _State.IN_STRING
        return "".join(out)

    def _advance_key_match(self, ch: str) -> None:
        if ch == self.key[self._match_pos]:
            self._match_pos += 1
            if self._match_pos == len(self.key):
                self._state = _State.SEEKING_COLON
        elif ch == self.key[0]:
            self._match_pos = 1
        else:
            self._match_pos = 0

    @property
    def done(self) -> bool:
        return self._state is _State.DONE
