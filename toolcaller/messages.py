"""OpenAI chat-completion request/message normalisation.

This mirrors ``common_chat_msgs_parse_oaicompat`` from
``ik_llama.cpp/common/chat.cpp`` so that prompts rendered here are
byte-for-byte what llama.cpp/ik_llama.cpp would render with ``--jinja``.

A *normalised message* is a plain ``dict`` with (optionally) these keys:

======================  =====================================================
``role``                ``system`` / ``developer`` / ``user`` / ``assistant``
                        / ``tool`` / ``function``
``content``             ``str`` **or** list of content parts
                        (``{"type": "text", "text": ...}``,
                        ``{"type": "image_url", ...}`` ...)
``tool_calls``          list of ``{"id", "type": "function",
                        "function": {"name", "arguments"}}`` where
                        ``arguments`` is a JSON **string** unless the
                        template supports object arguments
``tool_call_id``        id of the tool call a ``tool`` message answers
``name``                function name for legacy ``function``/``tool`` msgs
``reasoning_content``   pre-extracted chain of thought
======================  =====================================================

``None`` is used for absent values, exactly like llama.cpp does (its
``common_chat_msg`` only emits fields that are non-empty).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .errors import RequestError

#: Roles accepted in the OpenAI wire format.
VALID_ROLES = frozenset(
    {"system", "developer", "user", "assistant", "tool", "function"}
)

#: Content part types that are pure text.
TEXT_PART_TYPES = frozenset({"text", "input_text"})

#: Image/video part types (unsupported by a pure text-completion backend
#: unless the template/model can handle them, e.g. via vision tokens).
MEDIA_PART_TYPES = frozenset(
    {
        "image_url",
        "image",
        "input_image",
        "video",
        "video_url",
        "input_video",
    }
)


def dump_json(value: Any) -> str:
    """Serialise JSON the way llama.cpp's ``nlohmann::json::dump()`` does.

    That is: compact separators, ``ensure_ascii=false`` and (crucially)
    preserved key order.  Python 3.7+ dicts are insertion ordered, so
    ``json.dumps`` reproduces the original order of the incoming request.
    """
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def parse_json_leniently(raw: str) -> Any:
    """Parse a JSON string, returning the raw string if it is not valid JSON."""
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _normalize_content_part(part: Any, index: int) -> Dict[str, Any]:
    if not isinstance(part, dict):
        raise RequestError(
            f"messages[{index}].content part must be an object, got {type(part).__name__}"
        )
    part_type = part.get("type")
    if not isinstance(part_type, str):
        raise RequestError(
            f"messages[{index}].content part is missing a string 'type': {dump_json(part)}"
        )

    if part_type in MEDIA_PART_TYPES:
        # Keep the original part untouched; the template decides whether it
        # can render it (vision tokens) or raises an exception.
        return dict(part)

    if part_type in TEXT_PART_TYPES:
        text = part.get("text")
        if text is None and "content" in part:  # some clients send {"type":"text","content":...}
            text = part["content"]
        return {"type": "text", "text": "" if text is None else str(text)}

    # Unknown part: pass it through - custom templates may understand it.
    return dict(part)


def normalize_content(content: Any, index: int = 0) -> Any:
    """Normalise the ``content`` member of a message.

    Returns ``None`` (absent), a ``str``, or a list of content-part dicts.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return [_normalize_content_part(p, index) for p in content]
    if isinstance(content, dict):
        # lenient: {"type":"text","text":"..."} or {"text": "..."}
        if "type" in content or "text" in content:
            return [_normalize_content_part(content, index)]
        raise RequestError(
            f"messages[{index}].content must be a string or array, got object: {dump_json(content)}"
        )
    if isinstance(content, (int, float, bool)):
        return str(content)
    raise RequestError(
        f"messages[{index}].content must be a string or array, got {type(content).__name__}"
    )


def normalize_tool_call(call: Any, index: int, msg_index: int) -> Dict[str, Any]:
    """Convert one OpenAI ``tool_calls[]`` entry into our internal form."""
    if not isinstance(call, dict):
        raise RequestError(
            f"messages[{msg_index}].tool_calls[{index}] must be an object"
        )
    call_type = call.get("type", "function")
    if call_type != "function":
        raise RequestError(
            f"messages[{msg_index}].tool_calls[{index}]: unsupported type {call_type!r}"
        )
    function = call.get("function")
    if not isinstance(function, dict):
        raise RequestError(
            f"messages[{msg_index}].tool_calls[{index}] is missing 'function'"
        )
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise RequestError(
            f"messages[{msg_index}].tool_calls[{index}].function.name is required"
        )
    arguments = function.get("arguments")
    if arguments is None:
        arguments = ""
    elif not isinstance(arguments, str):
        # OpenAI allows an object here; llama.cpp stores the compact JSON dump
        # and later converts it to an object again if the template wants one.
        arguments = dump_json(arguments)

    call_id = call.get("id")
    return {
        "id": call_id if isinstance(call_id, str) else None,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def normalize_message(message: Any, index: int = 0) -> Dict[str, Any]:
    """Convert one OpenAI message object into our internal form."""
    if not isinstance(message, dict):
        raise RequestError(f"messages[{index}] must be an object")
    role = message.get("role")
    if not isinstance(role, str) or not role:
        raise RequestError(f"messages[{index}] is missing 'role'")

    normalized: Dict[str, Any] = {"role": role}

    has_content = "content" in message
    has_tool_calls = message.get("tool_calls") is not None
    has_function_call = message.get("function_call") is not None

    if not (has_content or has_tool_calls or has_function_call):
        raise RequestError(
            f"messages[{index}] must contain 'content', 'tool_calls' or 'function_call' "
            "(ref: llama.cpp issue #8367)"
        )

    if has_content:
        normalized["content"] = normalize_content(message.get("content"), index)
    else:
        normalized["content"] = None

    if has_tool_calls:
        raw_calls = message["tool_calls"]
        if not isinstance(raw_calls, (list, tuple)):
            raise RequestError(f"messages[{index}].tool_calls must be an array")
        normalized["tool_calls"] = [
            normalize_tool_call(call, i, index) for i, call in enumerate(raw_calls)
        ]

    if has_function_call:
        # Deprecated single-function-call API: map it onto tool_calls.
        normalized["tool_calls"] = [
            normalize_tool_call(
                {"type": "function", "function": message["function_call"]}, 0, index
            )
        ]
        if "function_call" in message and message.get("name"):
            normalized.setdefault("name", message.get("name"))

    tool_call_id = message.get("tool_call_id")
    if isinstance(tool_call_id, str) and tool_call_id:
        normalized["tool_call_id"] = tool_call_id

    name = message.get("name")
    if isinstance(name, str) and name:
        normalized["name"] = name

    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        normalized["reasoning_content"] = reasoning

    return normalized


def normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    """Normalise the ``messages`` array of a chat-completion request."""
    if not isinstance(messages, (list, tuple)):
        raise RequestError("'messages' must be an array")
    if not messages:
        raise RequestError("'messages' must not be empty (ref: llama.cpp issue #8367)")
    return [normalize_message(m, i) for i, m in enumerate(messages)]


def _coerce_arguments_for_jinja(
    raw: str, object_arguments: bool
) -> Any:
    """What the jinja context sees for ``tool_call.function.arguments``.

    llama.cpp stores arguments as a compact JSON string and then converts
    them to an object only when the template actually indexes into them
    (``supports_object_arguments``).  Templates that print
    ``tool_call.arguments`` directly (Qwen) or slice it (``out[:-1]``,
    Mistral) need the *string*; templates that iterate ``items()`` need the
    object.  If the string is not valid JSON we always keep the string so
    that the template can at least render it verbatim.
    """
    if not object_arguments:
        return raw
    parsed = parse_json_leniently(raw)
    return parsed


def messages_to_json(
    messages: Sequence[Dict[str, Any]],
    *,
    object_arguments: bool = False,
    concat_typed_text: bool = False,
    legacy_arguments: bool = False,
) -> List[Dict[str, Any]]:
    """Turn normalised messages into the JSON list handed to the template.

    Parameters
    ----------
    object_arguments:
        Emit ``function.arguments`` as a JSON object instead of a string.
    concat_typed_text:
        Render list-style ``content`` as a single newline-joined string
        (templates that do ``message.content + ...`` need this).
    legacy_arguments:
        Keep ``function.arguments`` as the *original* object when the client
        sent an object and the template does not support object arguments -
        identical to what the client sent, matching llama.cpp's lossy path.
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        jmsg: Dict[str, Any] = {"role": msg["role"]}

        content = msg.get("content")
        if isinstance(content, list):
            if concat_typed_text:
                jmsg["content"] = "\n".join(
                    str(part.get("text", ""))
                    for part in content
                    if part.get("type") in TEXT_PART_TYPES
                )
            else:
                jmsg["content"] = [dict(part) for part in content]
        elif content is None:
            jmsg["content"] = ""
        else:
            jmsg["content"] = content

        reasoning = msg.get("reasoning_content")
        if reasoning:
            jmsg["reasoning_content"] = reasoning

        name = msg.get("name")
        if name:
            jmsg["name"] = name

        tool_call_id = msg.get("tool_call_id")
        if tool_call_id:
            jmsg["tool_call_id"] = tool_call_id

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            jcalls = []
            for call in tool_calls:
                jcall = {
                    "id": call.get("id") or None,
                    "type": "function",
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": _coerce_arguments_for_jinja(
                            call["function"]["arguments"], object_arguments
                        ),
                    },
                }
                jcalls.append(jcall)
            jmsg["tool_calls"] = jcalls

        out.append(jmsg)
    return out


def normalize_request(request: Any) -> Dict[str, Any]:
    """Normalise a whole chat-completion request body.

    Returns a dict with ``messages`` (normalised) plus the raw request under
    ``_raw`` so callers can still read ``tools``, ``stop``, sampling params...
    """
    if not isinstance(request, dict):
        raise RequestError("request body must be a JSON object")

    normalized = dict(request)
    normalized["messages"] = normalize_messages(request.get("messages"))

    tools = request.get("tools")
    if tools is not None and not isinstance(tools, (list, tuple)):
        raise RequestError("'tools' must be an array")
    return normalized


def load_chat_request(path: str) -> Dict[str, Any]:
    """Read and normalise a chat-completion request from a JSON file.

    The file may either be a bare object with ``messages``/``tools`` or a
    wrapper like ``{"request": {...}}`` / ``{"body": {...}}``.
    """
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except ValueError as exc:
            raise RequestError(f"{path}: invalid JSON: {exc}") from exc

    if isinstance(data, dict):
        for key in ("request", "body", "payload", "chat"):
            inner = data.get(key)
            if isinstance(inner, dict) and "messages" in inner:
                return normalize_request(inner)
    return normalize_request(data)


def iter_message_texts(messages: Iterable[Dict[str, Any]]) -> Iterable[str]:
    """Yield every text chunk of every message (used by string scanners)."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for part in content:
                text = part.get("text")
                if isinstance(text, str):
                    yield text
