"""Declarative description of a model family's *output* syntax.

A :class:`SyntaxProfile` is pure data: markers, section names, how arguments
are serialised, how to clean up text.  :class:`toolcaller.parser.SyntaxParser`
then knows how to parse any output that follows that shape.

Nothing in here is model specific except the values themselves, so adding a
new model family = adding one instance to :mod:`toolcaller.profiles`
(or passing ``--profile-file my_model.json`` at runtime).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

#: How arguments of one call are encoded.
#: ``json``   - ``{...}`` JSON object (optionally inside a markdown fence)
#: ``json_array`` - ``[...]`` JSON array of ``{"name":..., "arguments":...}``
#: ``xml_params`` - ``<parameter=name>value</parameter>`` pairs
#: ``spark_xml_params`` - ``<arg_key>key</arg_key><arg_value>val</arg_value>`` pairs
#: ``python_kwargs`` - ``key="value", other=1`` keyword-style snippets
ARG_STYLES = ("json", "json_array", "xml_params", "spark_xml_params", "python_kwargs")


@dataclass
class SyntaxProfile:
    """One model family's tool-call / reasoning output syntax."""

    name: str
    description: str = ""

    #: Opening tag of one tool call, e.g. ``<tool_call>``.
    call_start: str = "<tool_call>"
    #: Closing tag of one tool call, e.g. ``</tool_call>``.
    call_end: str = "</tool_call>"
    #: Opening tag of the function-name section, e.g. ``<function=``.
    #: The name runs from here until ``function_name_terminator``.
    function_tag_open: str = "<function="
    function_name_terminator: str = ">"
    function_tag_close: str = "</function>"

    #: ``json`` | ``json_array`` | ``xml_params`` | ``python_kwargs``
    argument_style: str = "json"
    #: For ``xml_params``: ``<parameter=`` ... ``>`` value ``</parameter>``
    parameter_tag_open: str = "<parameter="
    #: Closing tag of one parameter value (default: ``</parameter>``,
    #: derived from ``parameter_tag_open`` when left empty).
    parameter_tag_close: str = ""
    #: Terminates a parameter *name* (default: ``>``).  Distinct from
    #: ``function_name_terminator`` because some formats use different
    #: punctuation for the two (``<function=f>`` vs ``ARG=x``).
    parameter_name_terminator: str = ">"
    #: For Spark's ``<arg_key>``/``<arg_value>`` format: value tag open/close.
    value_tag_open: str = "<arg_value>"
    value_tag_close: str = "</arg_value>"
    #: For ``json``/``json_array``: where to look for the JSON blob inside the
    #: call. ``after_name`` = right after ``<function=name>``;
    #: ``before_name`` = before ``<function=`` (Mistral puts the JSON first).
    argument_location: str = "after_name"

    #: Where the call id may appear, if the model emits one.
    call_id_keys: Sequence[str] = ("id", "call_id", "tool_call_id")

    #: Text that starts the model's chain of thought.
    reasoning_start: Optional[str] = "<think>"
    #: Text that ends the model's chain of thought.
    reasoning_end: Optional[str] = "</think>"

    #: Preferred stop strings understood by the *server* (token-level).
    #: The parser additionally accepts any of :attr:`call_end`,
    #: ``function_tag_close`` and ``eos_token`` as text-level terminators.
    extra_stop_strings: Sequence[str] = field(default_factory=tuple)

    #: Turn delimiters that end the assistant's turn (e.g. GLM's ``<|user|>``).
    #: Unlike :attr:`extra_stop_strings` these are *always* stops, with or
    #: without tools: without them the model runs past its own turn and starts
    #: writing the next speaker's message.
    turn_stop_strings: Sequence[str] = field(default_factory=tuple)

    #: Strip a leading whitespace-delimited tag such as ``<|channel|>final``
    #: from the content (harmony style final channels).
    strip_content_prefixes: Sequence[str] = field(default_factory=tuple)

    #: If True, a call body that looks like ``name(args)`` is also accepted
    #: without any tags (models that emit bare python-ish calls).
    accept_bare_name_parens: bool = False

    #: If True, argument JSON may be wrapped in a ```json fence.
    allow_json_fence: bool = True

    #: If True, arguments may be emitted as a JSON string
    #: (``{"arguments": "{...}"}``) and will be unwrapped.
    unwrap_string_arguments: bool = True

    #: Remove occurrences of these markers from the assistant content.
    content_cleanup: Sequence[str] = field(default_factory=tuple)

    #: Per-family quirks the generic parser should be aware of.
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.argument_style not in ARG_STYLES:
            raise ValueError(
                f"profile {self.name!r}: argument_style must be one of {ARG_STYLES}, "
                f"got {self.argument_style!r}"
            )
        self.extra_stop_strings = tuple(self.extra_stop_strings)
        self.strip_content_prefixes = tuple(self.strip_content_prefixes)
        self.content_cleanup = tuple(self.content_cleanup)
        self.call_id_keys = tuple(self.call_id_keys)
        # A parameter's value ends at ``parameter_tag_close``.  Default to the
        # conventional ``</parameter>`` unless the profile says otherwise.
        if not self.parameter_tag_close:
            opening = self.parameter_tag_open or "<parameter="
            if opening.startswith("<"):
                # ``<parameter=`` -> ``</parameter>``
                bare = opening.strip("<=\t ")
                self.parameter_tag_close = f"</{bare}>"
            else:
                # ``ARG=``-style: the conventional terminator is the clearest
                # guess; profiles with exotic markers should set it explicitly.
                self.parameter_tag_close = "</parameter>"

    # -- derived helpers ---------------------------------------------------
    def all_terminators(self) -> List[str]:
        """Text-level terminators for lenient parsing (longest first).

        Used by the parser to notice the end of a *truncated* call.  This is
        intentionally **not** the stop list: ``function_tag_close`` and
        ``call_start`` may legitimately appear mid-call, so stopping there
        would truncate perfectly good tool calls.
        """
        terms = {self.call_end, self.function_tag_close}
        terms.update(self.extra_stop_strings)
        return sorted((t for t in terms if t), key=len, reverse=True)

    def stop_strings(self, *, stop_at_call_end: bool = False) -> List[str]:
        """Markers that safely terminate generation.

        By default only markers that cannot appear inside a well-formed
        answer are returned - practically only ``extra_stop_strings`` (e.g.
        ``<|eom_id|>``) plus the template's EOS, which the caller adds.

        ``stop_at_call_end`` additionally stops at ``call_end``
        (``</tool_call>``).  That truncates *parallel* tool calls, so it is off
        by default: on a raw endpoint it is far better to let the model emit
        all calls and stop at end-of-turn, then let the parser split them.

        ``function_tag_close`` (``</function>``) and ``call_start`` never
        qualify: they legitimately occur mid-answer.
        """
        terms = set(self.extra_stop_strings)
        terms.update(self.turn_stop_strings)
        if stop_at_call_end:
            terms.add(self.call_end)
        return sorted((t for t in terms if t), key=len, reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        for key in ("extra_stop_strings", "turn_stop_strings",
                    "strip_content_prefixes", "content_cleanup", "call_id_keys"):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntaxProfile":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown profile keys: {sorted(unknown)}")
        return cls(**data)


# ---------------------------------------------------------------------------
# Argument decoding
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json|jsonc|\.json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def balanced_json_span(text: str, start: int = 0) -> Optional[Tuple[str, int]]:
    """Extract the first balanced JSON value at/after ``start``.

    Returns ``(json_text, end_index)``.  Handles strings and escapes so that
    braces inside strings do not confuse the matcher.
    """
    length = len(text)
    index = start
    while index < length and text[index] not in "{[":
        if not text[index].isspace():
            return None
        index += 1
    if index >= length:
        return None

    opening = text[index]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for pos in range(index, length):
        char = text[pos]
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
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[index: pos + 1], pos + 1
    return None


_XML_PARAM_RE = re.compile(
    r"<parameter\s*=\s*(?P<name>[^>\s]+)\s*>(?P<value>.*?)</parameter>", re.DOTALL
)
_XML_PARAM_OPEN_RE = re.compile(r"<parameter\s*=\s*(?P<name>[^>\s]+)\s*>", re.DOTALL)

#: Mistral-style: [ARGS]{"json"}
_ARGS_MARKER_RE = re.compile(r"\[ARGS\]", re.IGNORECASE)


def decode_xml_parameters(body: str, profile: SyntaxProfile) -> Optional[Dict[str, Any]]:
    """Decode ``<parameter=name>value</parameter>`` pairs.

    The tag shape is taken from the profile so custom formats work:
    ``parameter_tag_open`` + name + ``function_name_terminator`` + value +
    ``parameter_tag_close``.
    """
    opening = profile.parameter_tag_open
    closing = profile.parameter_tag_close or "</parameter>"
    if not opening or opening not in body:
        return None
    name_terminator = profile.parameter_name_terminator or ">"
    args: Dict[str, Any] = {}
    position = 0
    while True:
        start = body.find(opening, position)
        if start == -1:
            break
        name_start = start + len(opening)
        name_end = body.find(name_terminator, name_start)
        if name_end == -1:
            break
        name = body[name_start:name_end].strip()
        value_start = name_end + len(name_terminator)
        close = body.find(closing, value_start)
        if close == -1:
            # unterminated (truncated generation): take the rest
            raw_value = body[value_start:]
            position = len(body)
        else:
            raw_value = body[value_start:close]
            position = close + len(closing)
        if name:
            args[name] = clean_xml_value(raw_value)
        if position >= len(body):
            break
    return args if args else None


def decode_spark_xml_parameters(body: str, profile: SyntaxProfile) -> Optional[Dict[str, Any]]:
    """Decode Spark's ``<arg_key>key</arg_key><arg_value>val</arg_value>`` pairs.

    Spark templates use a paired tag format where keys and values are separate
    XML elements.
    """
    key_open = profile.parameter_tag_open  # e.g. "<arg_key>"
    key_close = profile.parameter_tag_close or "</arg_key>"
    val_open = profile.value_tag_open
    val_close = profile.value_tag_close or "</arg_value>"
    if not key_open or key_open not in body:
        return None
    if val_open not in body:
        return None
    args: Dict[str, Any] = {}
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
            raw_value = body[vo + len(val_open):]
            position = len(body)
        else:
            raw_value = body[vo + len(val_open):ve]
            position = ve + len(val_close)
        if key:
            args[key] = clean_xml_value(raw_value)
        if position >= len(body):
            break
    return args if args else None


def clean_xml_value(raw: str) -> Any:
    value = raw
    # The template writes "<parameter=x>\nVALUE\n</parameter>": drop exactly
    # one structural newline on each side, keep everything else.
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    stripped = value.strip()
    if stripped and stripped[0] in "{[":
        span = balanced_json_span(stripped)
        if span and span[0].strip() == stripped:
            try:
                import json

                return json.loads(span[0])
            except ValueError:
                pass
    if stripped in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[stripped]
    number = _try_number(stripped)
    if number is not None:
        return number
    return value


def _try_number(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        if re.fullmatch(r"-?\d+\.\d*([eE][-+]?\d+)?", text) or re.fullmatch(
            r"-?\d+[eE][-+]?\d+", text
        ):
            return float(text)
    except ValueError:
        return None
    return None


_PY_KWARG_RE = re.compile(
    r"""(?P<key>[A-Za-z_][\w\-]*)\s*=\s*(?P<value>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\[[^\]]*\]|\{[^}]*\}|-?\d+(?:\.\d+)?|true|false|null|True|False|None)""",
    re.DOTALL,
)


def decode_python_kwargs(text: str) -> Optional[Dict[str, Any]]:
    """Decode ``key="value", other=1`` argument snippets."""
    matches = list(_PY_KWARG_RE.finditer(text))
    if not matches:
        return None
    args: Dict[str, Any] = {}
    for match in matches:
        key = match.group("key")
        raw = match.group("value")
        args[key] = _decode_python_literal(raw)
    return args


def _decode_python_literal(raw: str) -> Any:
    import ast

    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        if raw in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[raw]
        if raw in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[raw]
        return raw


def decode_arguments(
    body: str, profile: SyntaxProfile
) -> Tuple[Dict[str, Any], str]:
    """Decode the argument section of one tool call.

    Returns ``(arguments, leftover_text)``.  Raises ``ValueError`` only for
    ``json`` styles where nothing at all could be decoded - the caller then
    decides whether to treat the body as free-form arguments.
    """
    import json

    text = body
    if profile.allow_json_fence:
        text = strip_code_fence(text)
    marker = _ARGS_MARKER_RE.search(text)
    if marker:
        text = text[marker.end():]

    if profile.argument_style in ("xml_params", "spark_xml_params"):
        from .syntax import decode_spark_xml_parameters

        decoded = decode_spark_xml_parameters(body, profile)
        if decoded is None:
            decoded = decode_xml_parameters(body, profile)
        if decoded is not None:
            leftover = text
            # strip out matched arg tags from leftover
            for tag in (
                profile.parameter_tag_open,
                profile.parameter_tag_close,
                profile.value_tag_open,
                profile.value_tag_close,
            ):
                if tag:
                    leftover = leftover.replace(tag, "")
            return decoded, leftover.strip()
        # fall through: some templates use xml params *and* json
        span = balanced_json_span(text)
        if span:
            try:
                value = json.loads(span[0])
            except ValueError as exc:
                raise ValueError(f"invalid JSON arguments: {exc}") from exc
            if isinstance(value, dict):
                return value, (text[: span[1]] + text[span[1]:]).replace(span[0], "").strip()
        raise ValueError("no <parameter=...> blocks and no JSON arguments found")

    if profile.argument_style == "python_kwargs":
        decoded = decode_python_kwargs(text)
        if decoded is None:
            raise ValueError("no keyword arguments found")
        return decoded, ""

    if profile.argument_style == "json_array":
        span = balanced_json_span(text)
        if not span:
            raise ValueError("no JSON array found")
        try:
            value = json.loads(span[0])
        except ValueError as exc:
            raise ValueError(f"invalid JSON arguments: {exc}") from exc
        if isinstance(value, list):
            # {"name":..., "arguments":...} entries are handled by the caller
            raise ValueError("json_array style is handled per-call, not per-body")
        if not isinstance(value, dict):
            raise ValueError("arguments must be a JSON object")
        return value, ""

    # json / json behind <function=...>
    span = balanced_json_span(text)
    if not span:
        raise ValueError("no JSON object found in arguments")
    try:
        value = json.loads(span[0])
    except ValueError as exc:
        raise ValueError(f"invalid JSON arguments: {exc}") from exc
    if isinstance(value, dict) and "parameters" in value and len(value) == 1:
        value = value["parameters"]  # wrapped style: {"parameters": {...}}
    if isinstance(value, dict) and set(value) == {"arguments"} and profile.unwrap_string_arguments:
        value = _unwrap_string_arguments(value["arguments"])
    if not isinstance(value, dict):
        raise ValueError("arguments must be a JSON object")
    return value, text[:span[1]].replace(span[0], "").strip() + text[span[1]:].strip()


def _unwrap_string_arguments(value: Any) -> Any:
    import json

    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "{[":
            try:
                return json.loads(stripped)
            except ValueError:
                return value
    return value
