"""Turn raw model output into an OpenAI assistant message.

The parser is *profile driven*: :class:`~toolcaller.syntax.SyntaxProfile`
says where calls start/end and how arguments are encoded; everything else
(reasoning extraction, parallel calls, lenient recovery, content cleanup) is
generic.

Two entry points
----------------
:func:`parse_completion`
    Parse the text returned by a raw completion.  ``reasoning_format`` decides
    whether ``<think>`` blocks are moved into ``reasoning_content``.
:class:`toolcaller.parser.SyntaxParser`
    Reusable object when you parse many outputs with the same profile.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .errors import ParseError
from .profiles import (
    detect_profile_from_template,
    get_profile,
)
from .syntax import (
    SyntaxProfile,
    balanced_json_span,
    decode_arguments,
    decode_python_kwargs,
    strip_code_fence,
)

#: ``reasoning_format`` values accepted from the request body.
REASONING_FORMATS = ("none", "auto", "deepseek", "deepseek-legacy", "qwen", "extract")


@dataclass
class ParsedToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]
    raw_arguments: str = ""
    index: int = 0

    def to_openai(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False, separators=(",", ":")),
            },
        }


@dataclass
class ParsedMessage:
    """The parsed assistant answer."""

    content: str = ""
    reasoning_content: str = ""
    tool_calls: List[ParsedToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    warnings: List[str] = field(default_factory=list)
    #: leftover text that looked like a tool call but could not be parsed
    unparsed: List[str] = field(default_factory=list)

    def to_openai_message(self) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": "assistant"}
        message["content"] = self.content if self.content else (None if self.tool_calls else "")
        if self.reasoning_content:
            message["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            message["tool_calls"] = [call.to_openai() for call in self.tool_calls]
        return message


@dataclass
class ParserSelection:
    """How the profile should be chosen."""

    profile: Optional[str] = None
    profile_file: Optional[str] = None
    auto_profile: bool = True


def resolve_profile(
    selection: ParserSelection, template_source: Optional[str] = None
) -> Tuple[SyntaxProfile, Optional[str]]:
    """Return ``(profile, hint)`` where ``hint`` explains an auto-detection."""
    if selection.profile_file:
        from .profiles import load_profile_file

        return load_profile_file(selection.profile_file), f"profile file {selection.profile_file}"
    if selection.profile:
        return get_profile(selection.profile), f"--profile {selection.profile}"
    if selection.auto_profile and template_source:
        detected = detect_profile_from_template(template_source)
        if detected:
            return get_profile(detected), f"auto-detected from template ({detected})"
    return get_profile("qwen"), "default profile (qwen)"


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------

def split_reasoning(
    text: str,
    profile: SyntaxProfile,
    reasoning_format: str = "auto",
) -> Tuple[str, str]:
    """Split ``text`` into ``(reasoning, content)`` per ``reasoning_format``.

    ``none``            -> nothing is extracted (text stays raw)
    ``auto`` / ``qwen`` -> extract if the start marker is present
    ``deepseek``        -> always extract; missing tags mean "all reasoning"
    ``deepseek-legacy`` -> keep the ``<think>`` tags inline in content
    """
    if not text:
        return "", ""
    start = profile.reasoning_start
    end = profile.reasoning_end
    if not start and not end:
        return "", text

    if reasoning_format == "none":
        return "", text
    if reasoning_format in ("deepseek-legacy", "qwen-legacy"):
        return "", text

    chunks = _extract_tag_pairs(text, start, end)
    if chunks is None:
        if reasoning_format in ("deepseek", "extract"):
            return text.strip("\n"), ""
        return "", text

    reasoning_parts, content_parts = chunks
    reasoning = "\n".join(part.strip("\n") for part in reasoning_parts if part.strip())
    content = "".join(content_parts)
    # keep leading structure of the content but drop the newlines that
    # typically follow </think>
    content = content.lstrip("\n")
    return reasoning, content


def _extract_tag_pairs(
    text: str, start: Optional[str], end: Optional[str]
) -> Optional[Tuple[List[str], List[str]]]:
    """Extract every ``start ... end`` block.

    Returns ``None`` when the start marker never appears (so the caller can
    fall back).  Handles the "already inside reasoning" case: when the
    generation prompt pre-filled ``<think>``, the output starts *inside* the
    reasoning block, so the first reasoning chunk runs until the first ``end``.
    """
    if not start or not end:
        return None

    # Pre-filled reasoning: the prompt already ended with ``start``, so the
    # model output begins inside the block and we must not require ``start``.
    if start not in text:
        end_index = text.find(end)
        if end_index != -1:
            return [text[:end_index]], [text[end_index + len(end):]]
        return None

    reasoning_parts: List[str] = []
    content_parts: List[str] = []

    # Text *before* the first start marker is regular content.
    first = text.find(start)
    if first > 0:
        content_parts.append(text[:first])

    cursor = first
    while True:
        begin = text.find(start, cursor)
        if begin == -1:
            content_parts.append(text[cursor:])
            break
        body_start = begin + len(start)
        finish = text.find(end, body_start)
        if finish == -1:
            reasoning_parts.append(text[body_start:])
            break
        reasoning_parts.append(text[body_start:finish])
        cursor = finish + len(end)
    return reasoning_parts, content_parts


# ---------------------------------------------------------------------------
# Tool call scanning
# ---------------------------------------------------------------------------

@dataclass
class _RawCall:
    name: str
    argument_text: str
    raw: str
    call_id: Optional[str] = None


class SyntaxParser:
    """Generic tool-call parser for one :class:`SyntaxProfile`."""

    def __init__(
        self,
        profile: SyntaxProfile,
        *,
        eos_token: str = "",
        known_tool_names: Optional[Iterable[str]] = None,
        strict: bool = False,
    ) -> None:
        self.profile = profile
        self.eos_token = eos_token or ""
        self.known_tool_names = set(n for n in (known_tool_names or ()) if n)
        self.strict = strict
        self._last_error: Optional[str] = None
        self._variants = self._build_variants(profile)

    # -- public ------------------------------------------------------------
    def parse(
        self,
        text: str,
        *,
        reasoning_format: str = "auto",
    ) -> ParsedMessage:
        if reasoning_format not in REASONING_FORMATS:
            raise ParseError(
                f"invalid reasoning_format {reasoning_format!r}; "
                f"expected one of {', '.join(REASONING_FORMATS)}"
            )
        text = self._normalize(text)
        reasoning, body = split_reasoning(text, self.profile, reasoning_format)

        result = ParsedMessage(reasoning_content=reasoning)
        raw_calls, leftover = self._extract_calls(body)
        if self._last_error:
            result.warnings.append(self._last_error)

        if raw_calls:
            for index, call in enumerate(raw_calls):
                try:
                    parsed = self._build_call(call, index)
                except ValueError as exc:
                    result.warnings.append(
                        f"could not parse tool call {call.name or '<unknown>'!r}: {exc}"
                    )
                    result.unparsed.append(call.raw)
                    continue
                result.tool_calls.append(parsed)
            if result.unparsed and self.strict:
                raise ParseError(
                    "failed to parse tool call(s): " + "; ".join(result.warnings)
                )

        content = self._clean_content(leftover)
        result.content = content.strip()
        result.finish_reason = "tool_calls" if result.tool_calls else "stop"
        return result

    # -- normalisation -----------------------------------------------------
    def _normalize(self, text: str) -> str:
        if not isinstance(text, str):
            raise ParseError(f"expected a string, got {type(text).__name__}")
        return text

    def _clean_content(self, content: str) -> str:
        out = content
        for prefix in self.profile.strip_content_prefixes:
            out = out.replace(prefix, "")
        for marker in self.profile.content_cleanup:
            out = out.replace(marker, "")
        for terminator in self._variants_terms:
            out = out.replace(terminator, "")
        # Turn delimiters (e.g. GLM's ``<|user|>``) end the assistant's turn.
        # The raw text is reconstructed as ``content + stop_word`` so reasoning
        # and tool markers survive, which means the delimiter itself has to be
        # removed from the answer.
        for terminator in self.profile.turn_stop_strings:
            out = out.replace(terminator, "")
        if self.eos_token:
            out = out.replace(self.eos_token, "")
        return out

    # -- call extraction ---------------------------------------------------
    def _build_variants(self, profile: SyntaxProfile) -> List[Dict[str, Any]]:
        """Concrete (start, end) markers for the profile's call syntax."""
        self._variants_terms: List[str] = list(profile.all_terminators())
        if self.eos_token:
            self._variants_terms.append(self.eos_token)

        if profile.options.get("array_of_calls") or profile.argument_style == "json_array":
            return [{"kind": "json_array"}]
        if not profile.call_start:
            return [{"kind": "bare_json"}] if profile.options.get("call_object_keys") else []

        variants: List[Dict[str, Any]] = [{"kind": "tagged"}]
        if profile.accept_bare_name_parens:
            # python-style ``name(a=1, b=2[, name2(...)])`` lists must be parsed
            # per call, so they take precedence over block-level scanning.
            variants.insert(0, {"kind": "name_parens"})
        return variants

    def _extract_calls(self, body: str) -> Tuple[List[_RawCall], str]:
        if not body or not body.strip():
            return [], body
        calls: List[_RawCall] = []
        leftover = body
        for variant in self._variants:
            kind = variant["kind"]
            if kind == "json_array":
                found, leftover = self._extract_from_json_array(leftover)
            elif kind == "bare_json":
                found, leftover = self._extract_bare_json(leftover)
            elif kind == "tagged":
                found, leftover = self._extract_tagged(leftover)
            elif kind == "name_parens":
                found, leftover = self._extract_name_parens(leftover)
            else:  # pragma: no cover - defensive
                found = []
            calls.extend(found)
            if calls:
                break

        if not calls:
            # Last resort: models that "escape" the requested format very often
            # fall back to a bare (often ```json-fenced) object/array of calls.
            # Recovering those is much more useful than returning them as text.
            found, leftover = self._extract_bare_json(body)
            calls.extend(found)

        if not calls and self._looks_like_a_broken_call(body):
            message = (
                "output looks like a tool call but no call could be decoded "
                "(malformed arguments?)"
            )
            if self.strict:
                raise ParseError(message + ": " + body.strip()[:400])
            self._last_error = message
        return calls, leftover

    def _looks_like_a_broken_call(self, body: str) -> bool:
        """Heuristic: does ``body`` contain a plausible, but broken, call?"""
        markers = [m for m in (self.profile.call_start, self.profile.function_tag_open) if m]
        return any(marker in body for marker in markers)

    def _extract_tagged(self, body: str) -> Tuple[List[_RawCall], str]:
        """Scan ``call_start ... call_end`` blocks."""
        profile = self.profile
        start_marker = profile.call_start
        end_marker = profile.call_end
        if profile.options.get("harmony"):
            return self._extract_harmony(body)

        calls: List[_RawCall] = []
        removed_spans: List[Tuple[int, int]] = []
        cursor = 0
        while True:
            begin = body.find(start_marker, cursor)
            if begin == -1:
                break
            body_start = begin + len(start_marker)
            if end_marker and end_marker != start_marker:
                finish = body.find(end_marker, body_start)
                if finish == -1:
                    # truncated: treat the rest of the text as the call
                    finish = len(body)
                    end_index = len(body)
                else:
                    end_index = finish + len(end_marker)
                chunk = body[body_start:finish]
            else:
                # single-marker formats ([TOOL_CALLS]): run to the next marker
                # or to the end of the text
                nxt = body.find(start_marker, body_start)
                if nxt == -1:
                    chunk = body[body_start:]
                    end_index = len(body)
                else:
                    chunk = body[body_start:nxt]
                    end_index = nxt
            raw = body[begin:end_index]
            parsed = self._parse_call_chunk(chunk, raw)
            if parsed is not None:
                calls.extend(parsed)
                removed_spans.append((begin, end_index))
            cursor = max(end_index, body_start)
            if cursor >= len(body):
                break

        leftover = _remove_spans(body, removed_spans)
        return calls, leftover

    def _extract_harmony(self, body: str) -> Tuple[List[_RawCall], str]:
        """``<|channel|>commentary to=functions.NAME<|message|>{json}<|call|>``."""
        pattern = re.compile(
            r"<\|channel\|>[^<]*?\bto=(?P<recipient>[^\s<]+)[^<]*?"
            r"(?:<\|constrain\|>[^<]*)?<\|message\|>"
            r"(?P<args>.*?)"
            r"(?:<\|call\|>|<\|return\|>|<\|end\|>|$)",
            re.DOTALL,
        )
        calls: List[_RawCall] = []
        spans: List[Tuple[int, int]] = []
        for match in pattern.finditer(body):
            recipient = match.group("recipient")
            if not recipient.startswith("functions."):
                continue
            name = recipient[len("functions."):]
            calls.append(
                _RawCall(name=name, argument_text=match.group("args"), raw=match.group(0))
            )
            spans.append((match.start(), match.end()))
        return calls, _remove_spans(body, spans)

    def _parse_call_chunk(self, chunk: str, raw: str) -> Optional[List[_RawCall]]:
        profile = self.profile
        text = chunk.strip("\n")

        # -- styles with no name inside the block -------------------------
        if profile.argument_style == "json_array" or profile.options.get("array_of_calls"):
            return self._calls_from_json_blob(text, raw)

        if not profile.function_tag_open and profile.options.get("call_object_keys"):
            return self._calls_from_json_blob(text, raw)

        # -- function name ------------------------------------------------
        name, remainder, call_id = self._extract_name(text)
        if not name:
            # maybe the whole chunk is a JSON object with a name field
            return self._calls_from_json_blob(text, raw)

        if profile.options.get("function_name_prefix") and name.startswith(
            profile.options["function_name_prefix"]
        ):
            name = name[len(profile.options["function_name_prefix"]):]
        separator = profile.options.get("name_index_separator")
        if separator and separator in name:
            name = name.rsplit(separator, 1)[0]

        return [_RawCall(name=name, argument_text=remainder, raw=raw, call_id=call_id)]

    def _extract_name(self, text: str) -> Tuple[str, str, Optional[str]]:
        profile = self.profile

        # Formats where the name is terminated by a marker but has no explicit
        # opening marker (Kimi: `functions.NAME:0<|tool_call_argument_begin|>`).
        if not profile.function_tag_open and profile.function_name_terminator:
            name_end = text.find(profile.function_name_terminator)
            if name_end != -1:
                name = text[:name_end].strip()
                # some formats wrap the name in brackets: [name(...)]
                name = name.lstrip("[\n \t").rstrip("]")
                terminator = profile.function_name_terminator
                # Non-consuming terminators (open a section that should stay):
                # single-char openers ({, (, [) and terminators that are also
                # the argument-section opening tag (e.g. Spark's <arg_key> is
                # both function_name_terminator and parameter_tag_open).
                _keep = terminator in ("{", "(", "[") or (
                    terminator == profile.parameter_tag_open
                    and profile.argument_style in ("xml_params", "spark_xml_params")
                )
                if _keep:
                    remainder = text[name_end:]
                else:
                    remainder = text[name_end + len(terminator):]
                if profile.function_tag_close and profile.function_tag_close in remainder:
                    remainder = remainder.replace(profile.function_tag_close, "")
                if self._valid_name(name):
                    return name, remainder, self._find_call_id(remainder)

        if not profile.function_tag_open:
            return "", text, None
        idx = text.find(profile.function_tag_open)
        name = ""
        remainder = text
        if idx != -1:
            name_start = idx + len(profile.function_tag_open)
            if profile.function_name_terminator:
                name_end = text.find(profile.function_name_terminator, name_start)
                if name_end == -1:
                    name = text[name_start:].strip()
                    remainder = ""
                else:
                    name = text[name_start:name_end].strip()
                    # A terminator that opens the arguments (``{`` for gemma's
                    # ``call:NAME{...}``, ``(`` for python style) must stay in
                    # the remainder, otherwise the JSON/kwargs lose their brace.
                    terminator = profile.function_name_terminator
                    resume = name_end if terminator in ("{", "(", "[") else name_end + len(terminator)
                    remainder = text[resume:]
            else:
                # name runs to the first whitespace or brace
                match = re.match(r"[^\s{}<>\[\]()\"']+", text[name_start:])
                name = match.group(0) if match else ""
                remainder = text[name_start + len(name):]
            # keep anything before the function tag (Mistral puts JSON first)
            prefix = text[:idx]
            remainder = prefix + remainder
        else:
            # no function tag: try a "name(...)" or "name{...}" form
            match = re.match(
                r"\s*(?P<name>[A-Za-z_][\w.\-]*)\s*(?P<rest>[\(\{].*)$", text, re.DOTALL
            )
            if match:
                name = match.group("name")
                remainder = match.group("rest")
            else:
                return "", text, None

        if profile.function_tag_close and profile.function_tag_close in remainder:
            remainder = remainder.replace(profile.function_tag_close, "")

        return name.strip(), remainder, self._find_call_id(remainder)

    _DEFAULT_NAME_RE = re.compile(r"^[A-Za-z_][\w.\-:]*$")

    def _valid_name(self, name: str) -> bool:
        """Reject obviously-wrong names (e.g. brackets, JSON fragments)."""
        if not name:
            return False
        pattern = self.profile.options.get("tool_name_pattern")
        if pattern:
            return re.match(pattern, name) is not None
        return self._DEFAULT_NAME_RE.match(name) is not None

    def _find_call_id(self, text: str) -> Optional[str]:
        for key in self.profile.call_id_keys:
            match = re.search(
                rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*["\']([^"\']+)["\']', text
            )
            if match:
                return match.group(1)
        return None

    def _extract_bare_json(self, body: str) -> Tuple[List[_RawCall], str]:
        calls: List[_RawCall] = []
        spans: List[Tuple[int, int]] = []
        # Models love wrapping JSON in a markdown fence; scan the unwrapped text
        # but keep offsets meaningful by scanning the original when possible.
        scan_text = strip_code_fence(body)
        cursor = 0
        while True:
            span = balanced_json_span(scan_text, cursor)
            if not span:
                break
            json_text, end = span
            parsed = self._calls_from_json_blob(json_text, json_text)
            if parsed:
                calls.extend(parsed)
                spans.append((cursor + scan_text[cursor:end].find(json_text), end))
            cursor = end
            if cursor >= len(scan_text):
                break
        if not calls:
            return [], body
        if scan_text is not body:
            # the fence was stripped: remove the whole original block
            return calls, ""
        return calls, _remove_spans(body, spans)

    def _extract_from_json_array(self, body: str) -> Tuple[List[_RawCall], str]:
        """Mistral: ``[TOOL_CALLS] [{"name":..., "arguments":{...}}]``."""
        return self._extract_bare_json(body)

    def _calls_from_json_blob(self, text: str, raw: str) -> List[_RawCall]:
        """Decode one JSON blob that contains one or more calls."""
        blob = strip_code_fence(text).strip()
        if not blob:
            return []
        data: Any
        try:
            span = balanced_json_span(blob)
            if span:
                data = json.loads(span[0])
            else:
                data = json.loads(blob)
        except ValueError:
            return []
        calls = self._calls_from_object(data, raw)
        return calls

    def _calls_from_object(self, data: Any, raw: str) -> List[_RawCall]:
        if isinstance(data, list):
            out: List[_RawCall] = []
            for item in data:
                out.extend(self._calls_from_object(item, raw))
            return out
        if not isinstance(data, dict):
            return []
        # a wrapper like {"tool_calls": [...]}
        for key in ("tool_calls", "calls"):
            if isinstance(data.get(key), list):
                return self._calls_from_object(data[key], raw)
        name = None
        for key in self.profile.options.get(
            "call_object_keys", ("name", "tool", "function", "tool_name")
        ):
            value = data.get(key)
            if isinstance(value, str) and value:
                name = value
                break
        if name is None and isinstance(data.get("function"), dict):
            name = data["function"].get("name")
        if not name:
            return []
        arguments = None
        for key in ("arguments", "parameters", "args", "input"):
            if key in data:
                arguments = data[key]
                break
        call_id = data.get("id") or data.get("call_id")
        if isinstance(arguments, str):
            argument_text = arguments
        elif arguments is None:
            argument_text = "{}"
        else:
            argument_text = json.dumps(arguments, ensure_ascii=False)
        return [_RawCall(name=name, argument_text=argument_text, raw=raw, call_id=call_id)]

    def _extract_name_parens(self, body: str) -> Tuple[List[_RawCall], str]:
        """``name(arg="v")`` / ``[name(arg=1)]`` style calls."""
        pattern = re.compile(r"(?P<name>[A-Za-z_][\w.\-]*)\s*\((?P<args>[^()]*)\)", re.DOTALL)
        calls: List[_RawCall] = []
        spans: List[Tuple[int, int]] = []
        for match in pattern.finditer(body):
            name = match.group("name")
            if self.known_tool_names and name not in self.known_tool_names:
                continue
            calls.append(_RawCall(name=name, argument_text=match.group("args"), raw=match.group(0)))
            spans.append((match.start(), match.end()))
        return calls, _remove_spans(body, spans)

    # -- call building -----------------------------------------------------
    def _build_call(self, call: _RawCall, index: int) -> ParsedToolCall:
        argument_text = call.argument_text
        if self.profile.options.get("call_list_brackets") and argument_text.strip().startswith("["):
            argument_text = argument_text.strip().strip("[]")

        if not argument_text.strip() or not argument_text.strip("\n \t(){}[]"):
            # a call with no parameters at all, e.g. <function=reset></function>
            arguments: Dict[str, Any] = {}
            raw_arguments = "{}"
        else:
            arguments, raw_arguments = self._decode(argument_text, call.name)
        call_id = call.call_id or self._make_id(index)
        return ParsedToolCall(
            id=call_id,
            name=call.name,
            arguments=arguments,
            raw_arguments=raw_arguments,
            index=index,
        )

    def _decode(self, argument_text: str, name: str) -> Tuple[Dict[str, Any], str]:
        profile = self.profile
        text = argument_text
        if profile.options.get("unquoted_keys"):
            text = _quote_unquoted_keys(text)

        if profile.options.get("deepseek_dsml"):
            decoded = _decode_dsml_parameters(text)
            if decoded is not None:
                return decoded, text

        if profile.argument_style in ("xml_params", "spark_xml_params"):
            decoded = decode_arguments(text, profile)
            return decoded

        if profile.argument_style == "python_kwargs":
            decoded = decode_python_kwargs(text)
            if decoded is not None:
                return decoded, text
            # python kwargs with a JSON fallback
            arguments, leftover = _decode_json_object(text)
            if arguments is not None:
                return arguments, leftover
            raise ValueError("expected key=value arguments")

        arguments, leftover = _decode_json_object(text, profile)
        if arguments is not None:
            return arguments, leftover
        if profile.argument_style in ("xml_params", "spark_xml_params") or profile.options.get("accept_json_arguments"):
            decoded = decode_arguments(text, profile)
            return decoded
        raise ValueError("expected a JSON object")

    # -- helpers -----------------------------------------------------------
    def _make_id(self, index: int) -> str:
        if self.profile.options.get("stable_call_ids"):
            return f"call_{index:05d}"
        return "call_" + uuid.uuid4().hex[:12]


def _decode_json_object(
    text: str, profile: Optional[SyntaxProfile] = None
) -> Tuple[Optional[Dict[str, Any]], str]:
    cleaned = strip_code_fence(text) if (profile is None or profile.allow_json_fence) else text
    span = balanced_json_span(cleaned)
    if not span:
        return None, text
    json_text, end = span
    try:
        value = json.loads(json_text)
    except ValueError:
        return None, text
    if isinstance(value, dict):
        if set(value) == {"arguments"} and isinstance(value["arguments"], str):
            inner, _ = _decode_json_object(value["arguments"], profile)
            if inner is not None:
                return inner, cleaned[end:]
        if set(value) == {"parameters"} and isinstance(value["parameters"], dict):
            return value["parameters"], cleaned[end:]
        return value, cleaned[end:]
    if isinstance(value, list):
        # e.g. arguments: [{"key": ...}] - not a normal shape, keep a wrapper
        return None, text
    return None, text


def _decode_dsml_parameters(text: str) -> Optional[Dict[str, Any]]:
    """DeepSeek: ``<｜DSML｜parameter name="city" string="true">Budapest</｜DSML｜parameter>``.

    The DSML marker uses full-width vertical bars (U+FF5C) around ``DSML``.
    """
    pattern = re.compile(
        r"<\uff5cDSML\uff5c\s*parameter\s+name=\"(?P<name>[^\"]+)\"[^>]*>"
        r"(?P<value>.*?)</\uff5cDSML\uff5c\s*parameter>",
        re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    args: Dict[str, Any] = {}
    from .syntax import clean_xml_value

    for match in matches:
        args[match.group("name")] = clean_xml_value(match.group("value"))
    return args


_UNQUOTED_KEY_RE = re.compile(r"(?P<prefix>[{,]\s*)(?P<key>[A-Za-z_][\w\-]*)(?P<sep>\s*:)")


def _quote_unquoted_keys(text: str) -> str:
    """Turn ``{city:"X"}`` into ``{"city":"X"}`` (gemma-style literals)."""
    def replace(match: re.Match) -> str:
        return f'{match.group("prefix")}"{match.group("key")}"{match.group("sep")}'

    quoted = _UNQUOTED_KEY_RE.sub(replace, text)
    if quoted != text:
        try:
            json.loads(quoted)
            return quoted
        except ValueError:
            return text
    return text


def _remove_spans(text: str, spans: Sequence[Tuple[int, int]]) -> str:
    if not spans:
        return text
    out: List[str] = []
    cursor = 0
    for begin, end in sorted(spans):
        if begin < cursor:
            continue
        out.append(text[cursor:begin])
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def parse_completion(
    text: str,
    *,
    profile: Any,
    eos_token: str = "",
    reasoning_format: str = "auto",
    known_tool_names: Optional[Iterable[str]] = None,
    strict: bool = False,
) -> ParsedMessage:
    """Parse one completion with an already resolved profile."""
    if isinstance(profile, str):
        profile = get_profile(profile)
    parser = SyntaxParser(
        profile, eos_token=eos_token, known_tool_names=known_tool_names, strict=strict
    )
    return parser.parse(text, reasoning_format=reasoning_format)


def message_to_chat_result(parsed: ParsedMessage) -> Dict[str, Any]:
    """``ParsedMessage`` -> dict with ``message``/``finish_reason`` keys."""
    return {
        "message": parsed.to_openai_message(),
        "finish_reason": parsed.finish_reason,
        "warnings": parsed.warnings,
    }
