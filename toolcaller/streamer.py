"""Incremental (in-band) splitting of raw model output for SSE streaming.

The daemon receives the model's output as a sequence of raw text pieces
(``on_chunk``) and has to turn it into OpenAI ``chat.completion.chunk`` deltas
*while it is being generated* - without buffering the whole answer first.

``IncrementalStreamParser`` is fed one raw piece at a time and emits high-level
events:

======================  ====================================================
event                   payload
======================  ====================================================
``reasoning``           ``str`` - a piece of chain-of-thought text
``content``             ``str`` - a piece of answer text
``tool_start``          ``(index, id, name)`` - new tool call begins
``tool_args``           ``(index, fragment)`` - incremental JSON argument text
======================  ====================================================

The generated event sequence matches what a native OpenAI-compatible
chat-completions server sends: reasoning deltas first, then ``content``, then
``tool_calls[]`` deltas whose ``function.arguments`` fragments concatenate to a
JSON object.  Tool-call markup never leaks into ``content``.

The parser understands the same syntax profiles as
:class:`toolcaller.parser.SyntaxParser`; only the *delimiters* are needed to
split the stream, while the final (complete) decoding is still done by
``SyntaxParser`` on the accumulated text.  This keeps streaming and
non-streaming output identical.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .syntax import SyntaxProfile


# Event kinds
REASONING = "reasoning"
CONTENT = "content"
TOOL_START = "tool_start"
TOOL_ARGS = "tool_args"

Event = Tuple[str, Any]


class IncrementalStreamParser:
    """Split a stream of raw text pieces into reasoning/content/tool-call events.

    Parameters
    ----------
    profile:
        The active syntax profile (delimiters, reasoning markers).
    reasoning_format:
        Mirrors ``SyntaxParser``: ``none`` disables reasoning extraction,
        ``deepseek``/``extract`` treat the leading text as reasoning when no
        markers appear, ``auto``/``qwen`` extract only when markers are seen.
    """

    def __init__(
        self,
        profile: SyntaxProfile,
        *,
        reasoning_format: str = "auto",
        reasoning_prefilled: bool = False,
    ) -> None:
        self.profile = profile
        self.reasoning_format = reasoning_format

        # Which block are we currently inside?
        # ``text`` -> plain content, ``reasoning`` -> inside <think>,
        # ``tool`` -> inside a <tool_call> block.
        self._mode = "text"

        # Buffer not yet classified (may contain a partial delimiter).
        self._pending = ""
        # Raw text of the tool call currently being streamed.
        self._call_buf = ""
        self._call_index = -1
        self._call_name = ""
        self._call_id = ""
        # How many argument bytes of the current call we already emitted.
        self._args_committed = ""
        self._call_count = 0
        # Text already emitted as reasoning (for the deepseek fallback).
        self._reasoning_emitted = False
        self._committed_any = False
        self._leading_close_pending = True

        # Delimiters
        self._r_start = profile.reasoning_start or ""
        self._r_end = profile.reasoning_end or ""
        self._c_start = profile.call_start or ""
        self._c_end = profile.call_end or ""
        # Some profiles have no explicit call_end (single-marker formats).
        self._single_marker = bool(self._c_start) and (
            not self._c_end or self._c_end == self._c_start
        )

        # A profile may pre-fill the reasoning start marker in the prompt
        # (Qwen's generation prompt ends with "<think>"); the model output then
        # begins *inside* the reasoning block.  In that case the stream starts
        # in reasoning mode and ends at the first ``</think>``.
        if (
            reasoning_prefilled
            and self._r_start
            and self._r_end
            and reasoning_format not in ("none", "deepseek-legacy")
        ):
            self._mode = "reasoning"

    # -- public ------------------------------------------------------------

    def feed(self, piece: str) -> List[Event]:
        """Consume one raw text piece, returning the events it produced."""
        if not piece:
            return []
        events: List[Event] = []
        self._pending += piece
        self._drain(events, final=False)
        return events

    def finalize(self) -> List[Event]:
        """Flush any buffered text at the end of the stream."""
        events: List[Event] = []
        self._drain(events, final=True)
        if self._mode == "tool":
            self._close_tool_call(events)
        if self._pending:
            target = REASONING if self._mode == "reasoning" else CONTENT
            events.append((target, self._pending))
            self._pending = ""
        return events

    @property
    def tool_calls_started(self) -> int:
        return self._call_count

    # -- core --------------------------------------------------------------

    def _drain(self, events: List[Event], *, final: bool) -> None:
        """Move as much of ``_pending`` into events as is safe."""
        while self._pending:
            if self._mode == "text":
                if not self._consume_text(events, final=final):
                    return
            elif self._mode == "reasoning":
                if not self._consume_reasoning(events, final=final):
                    return
            else:  # tool
                if not self._consume_tool(events, final=final):
                    return

    #: How many characters to buffer at the start of the stream to decide
    #: whether it begins inside a reasoning block (a bare ``</think>``).
    _LEADING_HOLD = 32

    def _hold_expired(self) -> bool:
        """Has the leading-``</think>`` decision window passed?"""
        return len(self._pending) >= self._LEADING_HOLD or "\n" in self._pending

    def _stray_close_possible(self) -> bool:
        """In text mode, may a bare ``</think>`` still need special handling?

        After the decision window it no longer converts preceding text to
        reasoning, but the marker itself is still dropped so it never leaks
        into ``content``.
        """
        return self.reasoning_format != "none" and bool(self._r_end)

    def _leading_close_reasoning(self) -> bool:
        """Is a leading ``</think>`` (without ``<think>``) still possible?

        Only allowed before any content/reasoning/tool output has been emitted:
        this covers the common case where the template prefilled the reasoning
        marker, or the model starts its answer with ``</think>``.
        """
        return self._leading_close_pending and not self._committed_any and not self._reasoning_emitted

    # -- plain text --------------------------------------------------------

    def _consume_text(self, events: List[Event], *, final: bool) -> bool:
        """Emit content until a reasoning/call marker is reached."""
        markers: List[Tuple[int, str]] = []
        if self._r_start and self._r_end and self.reasoning_format != "none":
            pos = self._pending.find(self._r_start)
            if pos != -1:
                markers.append((pos, "reasoning"))
            elif self._leading_close_reasoning() or self._stray_close_possible():
                # The generation prompt placed the model *inside* a reasoning
                # block without a visible ``<think>`` (e.g. Spark/Template
                # pre-fills it, or the model emits ``</think>`` first): the
                # text before the first ``</think>`` is reasoning.
                pos = self._pending.find(self._r_end)
                if pos != -1:
                    kind = (
                        "close-reasoning"
                        if self._leading_close_reasoning()
                        else "drop-close"
                    )
                    markers.append((pos, kind))
        if self._c_start:
            pos = self._pending.find(self._c_start)
            if pos != -1:
                markers.append((pos, "tool"))

        if markers:
            pos, kind = min(markers, key=lambda item: item[0])
            if pos > 0:
                if kind == "close-reasoning":
                    self._emit_reasoning(events, self._pending[:pos])
                else:
                    self._emit_content(events, self._pending[:pos])
            self._pending = self._pending[pos:]
            if kind in ("close-reasoning", "drop-close"):
                # Drop the closing marker; we are now in plain content mode.
                self._pending = self._pending[len(self._r_end):]
                self._mode = "text"
                self._leading_close_pending = False
                self._committed_any = self._committed_any or kind == "drop-close"
                return True
            self._mode = kind
            if kind == "reasoning":
                # Drop the opening ``<think>`` marker itself.
                self._pending = self._pending[len(self._r_start):]
            else:
                self._pending = self._pending[len(self._c_start):]
                self._begin_tool_call()
            return True

        if final:
            self._emit_content(events, self._pending)
            self._pending = ""
            return False

        # While a leading ``</think>`` (reasoning prefill without a visible
        # ``<think>``) is still possible, hold the buffer before deciding: a
        # real prefill close arrives within the first few tokens.  The hold is
        # bounded so plain answers still start streaming almost immediately.
        if self._leading_close_reasoning() and not self._hold_expired():
            return False

        # Keep a suffix that could be the beginning of a marker.
        keep = self._partial_suffix(
            [m for m in (self._r_start, self._c_start, self._r_end) if m]
        )
        if keep:
            body = self._pending[:-keep]
            if body:
                self._emit_content(events, body)
            self._pending = self._pending[-keep:]
        else:
            self._emit_content(events, self._pending)
            self._pending = ""
        return False

    # -- reasoning ---------------------------------------------------------

    def _consume_reasoning(self, events: List[Event], *, final: bool) -> bool:
        """Emit reasoning until ``</think>`` or a tool call begins."""
        markers: List[Tuple[int, int, str]] = []
        if self._r_end:
            pos = self._pending.find(self._r_end)
            if pos != -1:
                markers.append((pos, len(self._r_end), "text"))
        if self._c_start:
            pos = self._pending.find(self._c_start)
            if pos != -1:
                markers.append((pos, len(self._c_start), "tool"))

        if markers:
            pos, skip, next_mode = min(markers, key=lambda item: item[0])
            if pos > 0:
                self._emit_reasoning(events, self._pending[:pos])
            self._pending = self._pending[pos + skip:]
            self._mode = next_mode
            if next_mode == "tool":
                self._begin_tool_call()
            return True

        if final:
            self._emit_reasoning(events, self._pending)
            self._pending = ""
            return False

        keep = self._partial_suffix(
            [m for m in (self._r_end, self._c_start) if m]
        )
        if keep:
            body = self._pending[:-keep]
            if body:
                self._emit_reasoning(events, body)
            self._pending = self._pending[-keep:]
        else:
            self._emit_reasoning(events, self._pending)
            self._pending = ""
        return False

    # -- tool calls --------------------------------------------------------

    def _begin_tool_call(self) -> None:
        self._call_buf = ""
        self._call_index = self._call_count
        self._call_count += 1
        self._call_name = ""
        self._call_id = ""
        self._args_committed = ""

    def _consume_tool(self, events: List[Event], *, final: bool) -> bool:
        """Buffer the call body until ``</tool_call>`` (or the end)."""
        if self._single_marker:
            # Single-marker formats run until the next marker / end of text.
            nxt = self._pending.find(self._c_start)
            if nxt == -1 and not final:
                self._absorb_tool(events, self._pending)
                self._pending = ""
                return False
            if nxt == -1:
                self._absorb_tool(events, self._pending, final=True)
                self._pending = ""
                self._flush_tool_args(events, final=True)
                self._close_tool_call(events)
                self._mode = "text"
                return False
            self._absorb_tool(events, self._pending[:nxt], final=True)
            self._pending = self._pending[nxt:]
            self._flush_tool_args(events, final=True)
            self._close_tool_call(events)
            self._mode = "text"
            return True

        pos = self._pending.find(self._c_end) if self._c_end else -1
        if pos != -1:
            self._absorb_tool(events, self._pending[:pos], final=True)
            self._pending = self._pending[pos + len(self._c_end):]
            self._flush_tool_args(events, final=True)
            self._close_tool_call(events)
            self._mode = "text"
            return True

        if final:
            self._absorb_tool(events, self._pending, final=True)
            self._pending = ""
            self._flush_tool_args(events, final=True)
            self._close_tool_call(events)
            self._mode = "text"
            return False

        keep = self._partial_suffix([self._c_end] if self._c_end else [])
        if keep:
            self._absorb_tool(events, self._pending[:-keep])
            self._pending = self._pending[-keep:]
        else:
            self._absorb_tool(events, self._pending)
            self._pending = ""
        return False

    def _absorb_tool(self, events: List[Event], text: str, *, final: bool = False) -> None:
        """Append ``text`` to the current call and emit new argument fragments."""
        if not text:
            return
        self._call_buf += text
        self._sync_tool_metadata(events)
        self._flush_tool_args(events, final=final)

    def _sync_tool_metadata(self, events: List[Event]) -> None:
        """Announce the call (``tool_start``) as soon as name/id are known."""
        if self._call_id and self._call_name:
            return
        if self._call_name:
            name = self._call_name
        else:
            name = self._peek_tool_name(self._call_buf)
            if not name:
                return
        self._call_name = name
        if not self._call_id:
            self._call_id = self._make_id(self._call_index)
        events.append((TOOL_START, (self._call_index, self._call_id, self._call_name)))

    def _flush_tool_args(self, events: List[Event], *, final: bool) -> None:
        """Emit the not-yet-sent part of the serialized arguments.

        The serialized arguments are recomputed from the full call body on
        every flush.  For well-behaved formats the new value is an extension of
        the previous one, so only the tail is emitted.  Formats whose argument
        section only becomes locatable *after* the name is known (hermes-style
        JSON objects) can produce a value that does not extend the previous
        one; in that case the client would have concatenated wrong bytes, so we
        only start emitting once the value is stable (prefix-consistent).
        """
        if not self._call_name:
            return
        serialized = self._serialize_args(self._call_buf, final=final)
        if serialized is None:
            return
        if final and not serialized.startswith(self._args_committed):
            # Deltas are append-only, so a value that was already streamed as an
            # open string cannot be retyped (``"3`` -> ``3``) without rewriting
            # bytes the client has.  Keep the streamed form and just close it so
            # the accumulated JSON parses.  Values are therefore strings on the
            # streaming path when they arrive as scalars; anything structured
            # (objects/arrays, hermes-style JSON args) is unaffected.
            serialized = _close_open_json(self._args_committed)
        # Deltas are append-only: only an extension of what was already sent can
        # be emitted.
        if not serialized.startswith(self._args_committed):
            return
        if len(serialized) > len(self._args_committed):
            fragment = serialized[len(self._args_committed):]
            self._args_committed = serialized
            if fragment:
                events.append((TOOL_ARGS, (self._call_index, fragment)))

    def _close_tool_call(self, events: List[Event]) -> None:
        # Ensure the metadata (start) was announced even for an empty body: a
        # call such as ``<tool_call>reset</tool_call>`` has no argument tag at
        # all, so the name is only known from the whole body at the end.
        if not self._call_name:
            whole = self._call_buf.strip()
            name = _sanitize_name(whole)
            if name:
                self._call_name = name
        self._sync_tool_metadata(events)
        self._flush_tool_args(events, final=True)

    # -- helpers -----------------------------------------------------------

    def _emit_content(self, events: List[Event], text: str) -> None:
        if not text:
            return
        if self._mode == "reasoning":
            self._emit_reasoning(events, text)
            return
        self._committed_any = True
        self._leading_close_pending = False
        events.append((CONTENT, text))

    def _emit_reasoning(self, events: List[Event], text: str) -> None:
        if not text:
            return
        self._reasoning_emitted = True
        events.append((REASONING, text))

    def _partial_suffix(self, markers: List[str]) -> int:
        """Length of the longest suffix of ``_pending`` that prefixes a marker."""
        pending = self._pending
        best = 0
        for marker in markers:
            for size in range(min(len(marker) - 1, len(pending)), 0, -1):
                if pending.endswith(marker[:size]):
                    best = max(best, size)
                    break
        return best

    # -- argument serialization -------------------------------------------

    def _peek_tool_name(self, body: str) -> str:
        """Extract the function name from a (possibly partial) call body."""
        profile = self.profile
        open_tag = profile.function_tag_open
        if open_tag:
            if open_tag not in body:
                return ""
            start = body.index(open_tag) + len(open_tag)
            terminator = profile.function_name_terminator or ">"
            end = body.find(terminator, start) if terminator else -1
            if end == -1:
                return ""
            return body[start:end].strip()

        # Formats like Spark put the name right after ``<tool_call>``.
        terminator = profile.function_name_terminator or ""
        if terminator:
            end = body.find(terminator)
            if end != -1:
                candidate = body[:end].strip()
                if not candidate.startswith(("{", "[")):
                    name = _sanitize_name(candidate)
                    if name:
                        return name
        else:
            candidate = body.strip().split(None, 1)[0] if body.strip() else ""
            if candidate and not candidate.startswith(("{", "[")):
                name = _sanitize_name(candidate)
                if name:
                    return name

        return _json_name_value(body)

    def _serialize_args(self, body: str, *, final: bool) -> Optional[str]:
        """Serialize the arguments collected so far to (partial) JSON text.

        The returned string is a valid *prefix* of the final JSON object (or the
        exact leading substring of it), so a client concatenating the fragments
        and parsing progressively always sees a consistent growing object.
        """
        style = self.profile.argument_style
        if style in ("json", "json_array") and not self.profile.function_tag_open:
            # The call body *is* the JSON (hermes / mistral style): the value of
            # its ``arguments`` key is what the client expects.
            return _json_arguments_value(body, final=final)
        if style in ("json", "json_array"):
            after = body
            open_tag = self.profile.function_tag_open
            if open_tag and open_tag in body:
                after = body[body.index(open_tag) + len(open_tag):]
                terminator = self.profile.function_name_terminator or ">"
                if terminator and terminator in after:
                    after = after[after.index(terminator) + len(terminator):]
            if self.profile.argument_location == "before_name":
                before = body
                if open_tag and open_tag in body:
                    before = body[:body.index(open_tag)]
                return self._json_arguments(before, final=final)
            return self._json_arguments(after, final=final)

        # XML-parameter styles (spark, generic xml_params, ...).
        pairs = self._xml_pairs(body, final=final)
        if not pairs and not final:
            return None
        return _pairs_to_json(pairs, final=final)

    def _json_arguments(self, text: str, *, final: bool) -> Optional[str]:
        stripped = text.lstrip()
        if not stripped:
            return None
        start = 0
        while start < len(stripped) and stripped[start] not in "{[":
            start += 1
        if start >= len(stripped):
            return None
        candidate = stripped[start:]
        balanced = _json_complete(candidate)
        if balanced is not None:
            return balanced
        if not final:
            # Emit the raw prefix verbatim; the client tolerates partial JSON.
            return candidate
        return _repair_partial_json(candidate)

    def _xml_pairs(self, body: str, *, final: bool) -> List[Tuple[str, Any]]:
        profile = self.profile
        key_open = profile.parameter_tag_open
        if not key_open:
            return []
        val_open = profile.value_tag_open or ""
        val_close = profile.value_tag_close or ""
        key_close = profile.parameter_tag_close or "</parameter>"

        # Spark uses paired tags (``<arg_key>K</arg_key><arg_value>V``); the
        # generic XML style terminates the *name* and closes the value with
        # ``parameter_tag_close`` (``<parameter=K>V</parameter>``).  Decide by
        # the profile's declared style, not by which tags happen to be set.
        if profile.argument_style == "spark_xml_params":
            return self._paired_pairs(body, key_open, key_close, val_open, val_close, final)

        # Name-terminated style (Qwen): <parameter=K>V</parameter>
        name_term = profile.parameter_name_terminator or ">"
        pairs: List[Tuple[str, Any]] = []
        position = 0
        while True:
            ko = body.find(key_open, position)
            if ko == -1:
                break
            name_end = body.find(name_term, ko + len(key_open))
            if name_end == -1:
                break
            key = body[ko + len(key_open):name_end].strip()
            val_start = name_end + len(name_term)
            ve = body.find(key_close, val_start)
            if ve == -1:
                # Value still being generated (see _paired_pairs).
                if key:
                    raw = _strip_partial_marker(body[val_start:], key_close)
                    value = _xml_value(raw) if final else _xml_partial_value(raw)
                    pairs.append((key, value))
                break
            if key:
                pairs.append((key, _xml_value(body[val_start:ve])))
            position = ve + len(key_close)
        return pairs

    @staticmethod
    def _paired_pairs(
        body: str,
        key_open: str,
        key_close: str,
        val_open: str,
        val_close: str,
        final: bool,
    ) -> List[Tuple[str, Any]]:
        pairs: List[Tuple[str, Any]] = []
        position = 0
        while True:
            ko = body.find(key_open, position)
            if ko == -1:
                break
            ke = body.find(key_close, ko + len(key_open))
            if ke == -1:
                break
            key = body[ko + len(key_open):ke].strip()
            vo = body.find(val_open, ke + len(key_close))
            if vo == -1:
                break
            ve = body.find(val_close, vo + len(val_open))
            if ve == -1:
                # Value still being generated: emit it as a growing string so
                # the client sees arguments arrive token by token instead of
                # only after the closing tag.  A trailing partial terminator
                # ("</arg_val" on its way to "</arg_value>") must not be
                # mistaken for value text.
                if key:
                    raw = _strip_partial_marker(
                        body[vo + len(val_open):], val_close
                    )
                    value = _xml_value(raw) if final else _xml_partial_value(raw)
                    pairs.append((key, value))
                break
            if key:
                pairs.append((key, _xml_value(body[vo + len(val_open):ve])))
            position = ve + len(val_close)
        return pairs

    @staticmethod
    def _make_id(index: int) -> str:
        import uuid

        return "call_" + uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    name = name.strip()
    # Names never contain markup or JSON punctuation.
    if not name or any(ch in name for ch in "{}[]<>,:\n\t "):
        return ""
    return name


def _json_name_value(text: str) -> str:
    """Read the ``"name"`` (or ``"tool"``/``"function"``) key from JSON text.

    Works on a truncated object: returns "" until the value is complete.
    """
    for key in ("name", "tool", "function"):
        value = _json_string_key(text, key)
        if value:
            name = _sanitize_name(value)
            if name:
                return name
    return ""


def _json_string_key(text: str, key: str) -> str:
    """Return the decoded string value of ``key`` in ``text`` (or "")."""
    marker = '"%s"' % key
    pos = text.find(marker)
    if pos == -1:
        return ""
    index = pos + len(marker)
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    if index >= len(text) or text[index] != ":":
        return ""
    index += 1
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    if index >= len(text) or text[index] != '"':
        return ""
    index += 1
    out: List[str] = []
    escaped = False
    while index < len(text):
        char = text[index]
        if escaped:
            out.append(char)
            escaped = False
        elif char == "\\":
            out.append(char)
            escaped = True
        elif char == '"':
            try:
                return json.loads('"' + "".join(out) + '"')
            except ValueError:
                return "".join(out)
        else:
            out.append(char)
        index += 1
    # unterminated string: still decoding
    return ""


def _json_arguments_value(text: str, *, final: bool) -> Optional[str]:
    """Return the raw JSON value of the ``"arguments"`` key in ``text``.

    Emits exactly the substring of the source, so the concatenated fragments
    reconstruct valid JSON.  When the body is not an object-with-arguments
    (e.g. a bare argument object) the whole value is used.
    """
    value_start = _json_value_start(text, "arguments")
    if value_start is None:
        # No ``arguments`` key yet.  For a ``{...}`` object that is clearly a
        # name/arguments wrapper, emitting the wrapper would corrupt the
        # stream - wait until the argument value appears.  A bare argument
        # object (no wrapper keys) is safe to stream as-is.
        stripped = text.lstrip()
        if stripped.startswith("{") and (
            '"arguments"' in text
            or '"name"' in text
            or '"function"' in text
            or '"tool"' in text
        ):
            return None
        return _bare_json_value(text, final=final)
    candidate = text[value_start:]
    complete = _json_first_value(candidate)
    if complete is not None:
        return complete
    if not final:
        return candidate
    return _repair_partial_json(candidate)


def _bare_json_value(text: str, *, final: bool) -> Optional[str]:
    """Return the first JSON object/array found in ``text`` (partial if needed)."""
    start = 0
    while start < len(text) and text[start] not in "{[":
        start += 1
    if start >= len(text):
        return None
    candidate = text[start:]
    complete = _json_complete(candidate)
    if complete is not None:
        return complete
    if not final:
        return candidate
    return _repair_partial_json(candidate)


def _json_value_start(text: str, key: str) -> Optional[int]:
    """Index of the first non-space character of ``key``'s value."""
    marker = '"%s"' % key
    pos = text.find(marker)
    if pos == -1:
        return None
    index = pos + len(marker)
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    if index >= len(text) or text[index] != ":":
        return None
    index += 1
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    if index >= len(text):
        return None
    return index


def _xml_value(raw: str) -> Any:
    from .syntax import clean_xml_value

    return clean_xml_value(raw)


def _close_open_json(text: str) -> str:
    """Close an open-string JSON object produced by :func:`_pairs_to_json`.

    Used when the last value was streamed as an open string and the call ends
    before the value can be retyped (scalars, or generation cut mid-value).
    """
    if not text or text.rstrip().endswith("}"):
        return text
    if not text.startswith("{"):
        return text
    return text + '"}'


def _strip_partial_marker(text: str, marker: str) -> str:
    """Drop a trailing partial ``marker`` from ``text``.

    While streaming, ``</arg_value>`` may arrive as ``</arg_`` + ``value>``.
    The fragment is not value text, so it must not be emitted - otherwise the
    client's accumulated JSON ends up containing the tag.
    """
    if not marker:
        return text
    for size in range(min(len(marker) - 1, len(text)), 0, -1):
        if text.endswith(marker[:size]):
            return text[:-size]
    return text


def _xml_partial_value(raw: str) -> str:
    """Value text for a not-yet-terminated parameter.

    Always treated as a string: the value may still be a prefix of a number or
    boolean (``tr`` on its way to ``true``), so no type coercion is attempted.
    The structural newlines that templates put right after the opening tag are
    dropped, mirroring :func:`clean_xml_value`.
    """
    value = raw
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    # Templates put the closing tag on its own line, so a trailing newline is
    # structural, not value text.  Stripping it keeps the streamed prefix
    # identical to the eventual parsed value.
    return value.rstrip("\r\n")


def _json_complete(text: str) -> Optional[str]:
    """Return ``text`` when it is a complete JSON value, else ``None``."""
    from .syntax import balanced_json_span

    span = balanced_json_span(text)
    if not span:
        return None
    candidate, end = span
    if text[end:].strip():
        return None
    try:
        json.loads(candidate)
    except ValueError:
        return None
    return candidate


def _json_first_value(text: str) -> Optional[str]:
    """Return the first complete JSON object/array in ``text`` (ignoring tail)."""
    from .syntax import balanced_json_span

    span = balanced_json_span(text)
    if not span:
        return None
    candidate, _end = span
    try:
        value = json.loads(candidate)
    except ValueError:
        return None
    if not isinstance(value, (dict, list)):
        return None
    return candidate


def _repair_partial_json(text: str) -> str:
    """Best-effort close of a truncated JSON object/array."""
    text = text.strip()
    if not text:
        return "{}"
    if text[0] not in "{[":
        return "{}"
    depth_stack: List[str] = []
    in_string = False
    escaped = False
    last_safe = 0
    for pos, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth_stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if depth_stack:
                depth_stack.pop()
            if not depth_stack:
                last_safe = pos + 1
                break
        elif char == "," and depth_stack:
            last_safe = pos
    if in_string:
        # close the dangling string
        return text[:last_safe] + '"' + "".join(reversed(depth_stack))
    if depth_stack:
        return text[:last_safe] + "".join(reversed(depth_stack))
    return text


def _pairs_to_json(pairs: List[Tuple[str, Any]], *, final: bool) -> str:
    """Serialise collected ``(key, value)`` pairs to (possibly open) JSON.

    While the call is still being generated the result is a valid *prefix* of the
    final object: values are written with their opening quote but no closing
    quote, so appending more text to the last value simply extends the stream
    (``{"city":"Bud`` -> ``{"city":"Budapest"}``).
    """
    if not pairs:
        return "{}" if final else ""
    pieces: List[str] = []
    for index, (key, value) in enumerate(pairs):
        prefix = "{" if index == 0 else ","
        pieces.append(prefix + json.dumps(key, ensure_ascii=False) + ":")
        last = index == len(pairs) - 1
        if not last or final:
            pieces.append(
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            )
        else:
            # Not final: the last value is still the growing edge of the stream.
            # Always render it as an open JSON string so later text simply
            # extends it.  If the value turns out to be a number/bool, the final
            # flush rewinds and re-emits it typed (see _flush_tool_args).
            pieces.append(json.dumps(str(value), ensure_ascii=False)[:-1])
    return "".join(pieces) + ("}" if final else "")
