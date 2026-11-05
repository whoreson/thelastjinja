"""Template capability detection - a port of ``ik_llama.cpp/common/jinja/caps.cpp``.

Knowing what a template can do lets us emit the right ``content``/``arguments``
shape, and lets us fail loudly (instead of silently dropping data) when a
request uses a feature the template ignores - e.g. sending ``tools`` to a
template that never renders them.

Detection strategy
------------------
1. **Static analysis** (fast, no rendering): find which globals the template
   references.  This is what Minja does with its ``stats.used`` bookkeeping.
2. **Probe rendering** (authoritative): render tiny synthetic conversations
   like ``caps.cpp`` does and check whether the result is *consistent* with
   what the probe asked for.  This catches templates that reference a global
   but ignore it (``{% set tools = tools or [] %}`` followed by no output).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .environment import TemplateEnvironment
from .messages import messages_to_json

#: Templates that do ``{% set enable_thinking = enable_thinking | default(...) %}``
#: don't show ``enable_thinking`` in the used-globals AST walk (the ``store``
#: masks the ``load``), but the variable is clearly a template knob.
_re_enable_thinking = re.compile(
    r"(?x)\{%-?\s*set\s+enable_thinking\s*=\s*enable_thinking\s*\|\s*default",
)

#: A tool definition as the ``tools`` argument of the template sees it.
PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}

#: Second probe tool (used for parallel tool call detection).
PROBE_TOOL_2 = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}


@dataclass(frozen=True)
class TemplateCaps:
    """What one chat template is able to represent."""

    supports_tools: bool = True
    supports_tool_calls: bool = True
    supports_parallel_tool_calls: bool = True
    supports_system_role: bool = True
    supports_preserve_reasoning: bool = False
    #: ``function.arguments`` must be a JSON *object* (template iterates it)
    supports_object_arguments: bool = False
    supports_string_content: bool = True
    supports_typed_content: bool = False
    #: Whether ``enable_thinking`` reaches the template (False => baked/absent)
    supports_enable_thinking: bool = False
    #: Static analysis: globals the template reads
    used_globals: frozenset = frozenset()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["used_globals"] = sorted(self.used_globals)
        return data

    #: llama.cpp-style map (for API compatibility / debugging output)
    def to_map(self) -> Dict[str, bool]:
        return {
            "supports_string_content": self.supports_string_content,
            "supports_typed_content": self.supports_typed_content,
            "supports_tools": self.supports_tools,
            "supports_tool_calls": self.supports_tool_calls,
            "supports_parallel_tool_calls": self.supports_parallel_tool_calls,
            "supports_system_role": self.supports_system_role,
            "supports_preserve_reasoning": self.supports_preserve_reasoning,
            "supports_object_arguments": self.supports_object_arguments,
        }

    def __str__(self) -> str:  # pragma: no cover
        lines = [f"  {k}={str(v).lower()}" for k, v in self.to_map().items()]
        lines.append(f"  supports_enable_thinking={str(self.supports_enable_thinking).lower()}")
        return "Caps(\n" + "\n".join(lines) + "\n)"


def _probe_context(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    *,
    object_arguments: bool = False,
    add_generation_prompt: bool = True,
) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "messages": messages_to_json(
            messages, object_arguments=object_arguments, concat_typed_text=False
        ),
        "bos_token": "",
        "eos_token": "",
    }
    if tools:
        ctx["tools"] = tools
    if add_generation_prompt:
        ctx["add_generation_prompt"] = True
    return ctx


def _user(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}


def _assistant(text: str, tool_calls: Optional[List[Dict[str, Any]]] = None,
               reasoning: Optional[str] = None) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning:
        msg["reasoning_content"] = reasoning
    return msg


def _tool_call(name: str, arguments: Any, call_id: str = "call_00001") -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _single_tool_conversation(arguments: Any) -> List[Dict[str, Any]]:
    return [
        _user("Probe the system."),
        _assistant("", [_tool_call("get_weather", arguments)]),
        {
            "role": "tool",
            "content": "probe-tool-response",
            "tool_call_id": "call_00001",
        },
        _assistant("The probe returned 'probe-tool-response'."),
        _user("Probe again."),
    ]


def analyze_caps(env: TemplateEnvironment) -> TemplateCaps:
    """Detect capabilities of a template.  Results are computed once per env."""
    cached = getattr(env, "_caps_cache", None)
    if cached is not None:
        return cached

    used = env.used_globals()

    supports_string_content = True
    supports_typed_content = False
    supports_system_role = True
    supports_tools = True
    supports_tool_calls = True
    supports_parallel_tool_calls = True
    supports_preserve_reasoning = False
    supports_object_arguments = False
    supports_enable_thinking = (
        ("enable_thinking" in used and not env.bakes("enable_thinking"))
        or (
            not env.bakes("enable_thinking")
            and bool(_re_enable_thinking.search(env.source))
        )
    )

    # -- typed vs string content ------------------------------------------
    if not used & {"messages", "message"}:
        supports_typed_content = False
        supports_string_content = True
    else:
        typed_probe = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first part"},
                ],
            }
        ]
        if "content" in used:
            typed_render_ok = env.renders_with(**_probe_context(typed_probe))
            # A template that only does `content + "str"` still works on a
            # concatenated string, so it "supports string content". It only
            # *needs* typed content when it looks like it iterates parts.
            if typed_render_ok and env.renders_with(
                **_probe_context(
                    [{"role": "user", "content": [{"type": "text", "text": "a"},
                                                  {"type": "text", "text": "b"}]}]
                )
            ):
                supports_typed_content = True
        string_probe = [_user("plain string probe")]
        if "content" in used and not env.renders_with(**_probe_context(string_probe)):
            supports_string_content = False

    # -- system role -------------------------------------------------------
    if used & {"messages", "message"}:
        system_probe = [
            {"role": "system", "content": "SYSTEM-PROBE-MARKER"},
            _user("user probe"),
        ]
        try:
            rendered = env.render(_probe_context(system_probe))
        except Exception:  # noqa: BLE001 - template rejects system messages
            rendered = ""
        if "SYSTEM-PROBE-MARKER" not in rendered:
            supports_system_role = False

    # -- tools -------------------------------------------------------------
    tool_probe = [
        _user("Call the probe tool."),
        _assistant("", [_tool_call("get_weather", {"city": "Budapest"})]),
        {"role": "tool", "content": "sunny", "tool_call_id": "call_00001"},
    ]
    uses_messages = bool(used & {"messages", "message"})
    mentions_tools = bool(used & {"tools", "tool"})

    # (a) are tool *definitions* rendered into the prompt?
    #     Compare output with and without tools: if anything changes, the
    #     template makes use of them (templates that list only descriptions or
    #     parameter schemas never print the name).
    if not mentions_tools:
        supports_tools = False
    else:
        try:
            with_tools = env.render(_probe_context(tool_probe, [PROBE_TOOL, PROBE_TOOL_2]))
        except Exception:  # noqa: BLE001
            with_tools = ""
        try:
            without_tools = env.render(_probe_context(tool_probe, None))
        except Exception:  # noqa: BLE001
            without_tools = ""
        if not with_tools or with_tools == without_tools:
            supports_tools = False

    # (b) tool_calls, argument shape and parallel calls.  These are properties
    #     of the *message* rendering, so probe them independently of (a):
    #     templates that never list definitions can still render past calls.
    if not uses_messages:
        supports_tool_calls = False
        supports_parallel_tool_calls = False
    else:
        # object arguments?
        try:
            obj_render = env.render(
                _probe_context(
                    _single_tool_conversation({"city": "Budapest"}),
                    [PROBE_TOOL],
                    object_arguments=True,
                )
            )
        except Exception:  # noqa: BLE001
            obj_render = ""
        obj_ok = "get_weather" in obj_render and "Budapest" in obj_render

        try:
            str_render = env.render(
                _probe_context(
                    _single_tool_conversation(json.dumps({"city": "Budapest"})),
                    [PROBE_TOOL],
                    object_arguments=False,
                )
            )
        except Exception:  # noqa: BLE001
            str_render = ""
        str_ok = "get_weather" in str_render and "Budapest" in str_render

        # Templates that do `tool_call.arguments|items` only work with an
        # object; templates that print/slice the raw string only work with a
        # string. Prefer whichever probe actually produces both the tool name
        # and the argument value; objects win when both (llama.cpp default).
        if obj_ok and not str_ok:
            supports_object_arguments = True
        elif str_ok and not obj_ok:
            supports_object_arguments = False
        elif not obj_ok and not str_ok:
            supports_object_arguments = True
        else:
            supports_object_arguments = _looks_like_object_arguments(env)

        # tool_calls rendered at all?
        if "get_weather" not in str_render and "get_weather" not in obj_render:
            supports_tool_calls = False

        # parallel tool calls: two calls -> both names must appear
        two_calls = [
            _user("Call both probe tools."),
            _assistant(
                "",
                [
                    _tool_call("get_weather", {"city": "Budapest"}, "call_00001"),
                    _tool_call("get_time", {"city": "Budapest"}, "call_00002"),
                ],
            ),
            {"role": "tool", "content": "sunny", "tool_call_id": "call_00001"},
            {"role": "tool", "content": "noon", "tool_call_id": "call_00002"},
        ]
        try:
            parallel_render = env.render(
                _probe_context(two_calls, [PROBE_TOOL, PROBE_TOOL_2],
                               object_arguments=supports_object_arguments)
            )
        except Exception:  # noqa: BLE001
            parallel_render = ""
        if "get_weather" not in parallel_render or "get_time" not in parallel_render:
            supports_parallel_tool_calls = False

    # -- reasoning preservation -------------------------------------------
    if "reasoning_content" in used:
        reasoning_probe = [
            _user("first probe"),
            _assistant("assistant probe", reasoning="REASONING-PROBE-MARKER"),
            _user("second probe"),
        ]
        try:
            rendered = env.render(_probe_context(reasoning_probe))
        except Exception:  # noqa: BLE001
            rendered = ""
        supports_preserve_reasoning = "REASONING-PROBE-MARKER" in rendered

    caps = TemplateCaps(
        supports_tools=supports_tools,
        supports_tool_calls=supports_tool_calls,
        supports_parallel_tool_calls=supports_parallel_tool_calls,
        supports_system_role=supports_system_role,
        supports_preserve_reasoning=supports_preserve_reasoning,
        supports_object_arguments=supports_object_arguments,
        supports_string_content=supports_string_content,
        supports_typed_content=supports_typed_content,
        supports_enable_thinking=supports_enable_thinking,
        used_globals=used,
    )
    setattr(env, "_caps_cache", caps)
    return caps


_COMMON_OBJECT_HINTS = ("|items", "|items()", ".items()", "|dictsort", "args_name", "args_value")


def _looks_like_object_arguments(env: TemplateEnvironment) -> bool:
    """Heuristic used when both argument shapes render identically."""
    source = env.source
    return any(hint in source for hint in _COMMON_OBJECT_HINTS)
