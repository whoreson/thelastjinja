"""Built-in model-family profiles.

Each profile describes the *output* syntax (tool calls + reasoning) of one
model family.  The parser is generic; only these values differ, so adding a
family is a matter of adding a :class:`~toolcaller.syntax.SyntaxProfile`.

Formats are based on ``ik_llama.cpp/common/chat.cpp`` (the specialised
handlers) and the corresponding HuggingFace templates.

Use ``python -m toolcaller --list-profiles`` to print them, and
``--profile-file my_model.json`` to load your own.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .errors import ConfigError
from .syntax import SyntaxProfile


# ---------------------------------------------------------------------------
# Qwen 2.5 / 3.x (Hermes-style XML)
# ---------------------------------------------------------------------------
# <tool_call>
# <function=get_weather>
# <parameter=city>
# Budapest
# </parameter>
# </function>
# </tool_call>
QWEN = SyntaxProfile(
    name="qwen",
    description="Qwen 2.5/3.x Hermes XML: <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>",
    call_start="<tool_call>",
    call_end="</tool_call>",
    function_tag_open="<function=",
    function_name_terminator=">",
    function_tag_close="</function>",
    argument_style="xml_params",
    parameter_tag_open="<parameter=",
    reasoning_start="<think>",
    reasoning_end="</think>",
    # Qwen happily emits JSON arguments when the prompt asks for it, and some
    # fine-tunes mix both; the parser handles that fallback automatically.
    content_cleanup=(),
    options={"accept_json_arguments": True},
)

# Qwen3-Coder sometimes emits a bare JSON array of calls.
QWEN_JSON = SyntaxProfile(
    name="qwen-json",
    description="Qwen-style but expects JSON tool calls (no <tool_call> wrapper)",
    call_start="<tool_call>",
    call_end="</tool_call>",
    function_tag_open="<function=",
    function_name_terminator=">",
    argument_style="json",
    reasoning_start="<think>",
    reasoning_end="</think>",
    accept_bare_name_parens=True,
)

# ---------------------------------------------------------------------------
# Mistral / Ministral: [TOOL_CALLS] [{"name": ..., "arguments": {...}, "id": ...}]
# ---------------------------------------------------------------------------
MISTRAL = SyntaxProfile(
    name="mistral",
    description="Mistral [TOOL_CALLS] [{\"name\":...,\"arguments\":{...}}]",
    call_start="[TOOL_CALLS]",
    call_end="[TOOL_CALLS]",
    function_tag_open="",
    function_name_terminator="",
    function_tag_close="",
    argument_style="json_array",
    argument_location="after_name",
    reasoning_start=None,
    reasoning_end=None,
    options={"array_of_calls": True, "call_object_keys": ("name", "arguments", "id")},
)

# ---------------------------------------------------------------------------
# Llama 3.x python-style: <|python_tag|>{"name": ..., "parameters": {...}}
# ---------------------------------------------------------------------------
LLAMA3 = SyntaxProfile(
    name="llama3",
    description="Llama 3.x <|python_tag|>{\"name\":...,\"parameters\":{...}}",
    call_start="<|python_tag|>",
    call_end="<|eom_id|>",
    function_tag_open="",
    function_name_terminator="",
    function_tag_close="",
    argument_style="json",
    reasoning_start=None,
    reasoning_end=None,
    accept_bare_name_parens=True,
    extra_stop_strings=("<|eom_id|>", "<|eot_id|>"),
    options={"call_object_keys": ("name", "parameters")},
)

# ---------------------------------------------------------------------------
# Hermes / NousResearch generic:
#   <tool_call>{"name": "x", "arguments": {...}}</tool_call>
# ---------------------------------------------------------------------------
HERMES = SyntaxProfile(
    name="hermes",
    description="Hermes/JSON-in-tag: <tool_call>{\"name\":...,\"arguments\":{...}}</tool_call>",
    call_start="<tool_call>",
    call_end="</tool_call>",
    function_tag_open="",
    function_name_terminator="",
    function_tag_close="",
    argument_style="json",
    reasoning_start="<think>",
    reasoning_end="</think>",
    options={"call_object_keys": ("name", "arguments")},
)

# ---------------------------------------------------------------------------
# DeepSeek V3 / V3.1 / V3.2 DSML
#   <｜DSML｜function_calls>
#   <｜DSML｜invoke name="get_weather">
#   <｜DSML｜parameter name="city" string="true">Budapest</｜DSML｜parameter>
#   </｜DSML｜invoke>
#   </｜DSML｜function_calls>
# ---------------------------------------------------------------------------
DEEPSEEK = SyntaxProfile(
    name="deepseek",
    description="DeepSeek DSML XML tool calls",
    call_start="<｜DSML｜invoke",
    call_end="</｜DSML｜invoke>",
    function_tag_open='name="',
    function_name_terminator='"',
    function_tag_close="",
    argument_style="xml_params",
    parameter_tag_open="<｜DSML｜parameter",
    reasoning_start="<think>",
    reasoning_end="</think>",
    extra_stop_strings=("</｜DSML｜function_calls>", "</｜DSML｜tool_calls>", "</｜DSML｜ calls>"),
    options={
        "deepseek_dsml": True,
        "dsml_parameter_marker": "｜DSML｜parameter",
    },
)

# ---------------------------------------------------------------------------
# Spark X2.6 (Alibaba)
#   <tool_call>get_weather<arg_key>city</arg_key><arg_value>Budapest</arg_value></tool_call>
# ---------------------------------------------------------------------------
SPARK = SyntaxProfile(
    name="spark",
    description="Spark X2.6: <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>",
    call_start="<tool_call>",
    call_end="</tool_call>",
    function_tag_open="",
    function_name_terminator="<arg_key>",
    function_tag_close="",
    argument_style="spark_xml_params",
    parameter_tag_open="<arg_key>",
    parameter_tag_close="</arg_key>",
    value_tag_open="<arg_value>",
    value_tag_close="</arg_value>",
    reasoning_start="<think>",
    reasoning_end="</think>",
)

# ---------------------------------------------------------------------------
# GPT-OSS / harmony
#   <|start|>assistant<|channel|>commentary to=functions.get_weather
#   <|constrain|>json<|message|>{"city":"Budapest"}<|call|>
# ---------------------------------------------------------------------------
GPT_OSS = SyntaxProfile(
    name="gpt-oss",
    description="gpt-oss harmony channels: <|channel|>commentary to=functions.NAME<|message|>{json}",
    call_start="<|channel|>",
    call_end="<|call|>",
    function_tag_open=" to=functions.",
    function_name_terminator="<|",
    function_tag_close="",
    argument_style="json",
    argument_location="after_name",
    reasoning_start="<|channel|>analysis<|message|>",
    reasoning_end="<|end|>",
    extra_stop_strings=("<|return|>", "<|end|>", "<|call|>"),
    content_cleanup=("<|message|>",),
    options={"harmony": True},
)

# ---------------------------------------------------------------------------
# Kimi K2 Thinking
#   <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0
#   <|tool_call_argument_begin|>{"city":"Budapest"}<|tool_call_end|>
#   <|tool_calls_section_end|>
# ---------------------------------------------------------------------------
KIMI = SyntaxProfile(
    name="kimi",
    description="Kimi K2: <|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{json}<|tool_call_end|>",
    call_start="<|tool_call_begin|>",
    call_end="<|tool_call_end|>",
    function_tag_open="",
    function_name_terminator="<|tool_call_argument_begin|>",
    function_tag_close="",
    argument_style="json",
    argument_location="after_name",
    reasoning_start="<think>",
    reasoning_end="</think>",
    extra_stop_strings=("<|tool_calls_section_end|>", "<|tool_calls_section_begin|>"),
    options={
        "function_name_prefix": "functions.",
        "name_index_separator": ":",
    },
)

# ---------------------------------------------------------------------------
# LFM2 / LFM2.5 python-style
#   <|tool_call_start|>[get_weather(city="Budapest")]<|tool_call_end|>
# ---------------------------------------------------------------------------
LFM2 = SyntaxProfile(
    name="lfm2",
    description="LFM2 python-style: <|tool_call_start|>[name(arg=\"v\")]<|tool_call_end|>",
    call_start="<|tool_call_start|>",
    call_end="<|tool_call_end|>",
    function_tag_open="",
    function_name_terminator="(",
    function_tag_close=")",
    argument_style="python_kwargs",
    reasoning_start="<think>",
    reasoning_end="</think>",
    accept_bare_name_parens=True,
    options={"call_list_brackets": True},
)

# ---------------------------------------------------------------------------
# Ministral 3 / Magistral: [THINK] / [TOOL_CALLS]name[ARGS]{json}
# ---------------------------------------------------------------------------
MINISTRAL3 = SyntaxProfile(
    name="ministral3",
    description="Ministral 3: [TOOL_CALLS]name[ARGS]{json}",
    call_start="[TOOL_CALLS]",
    call_end="[TOOL_CALLS]",
    function_tag_open="",
    function_name_terminator="[ARGS]",
    function_tag_close="",
    argument_style="json",
    argument_location="after_name",
    reasoning_start="[THINK]",
    reasoning_end="[/THINK]",
    extra_stop_strings=("[TOOL_CALLS]", "[THINK]"),
)

# ---------------------------------------------------------------------------
# Gemma 4 / generic gemma tool calls
#   <|tool_call>call:get_weather{city:"Budapest"}<tool_call|>
# ---------------------------------------------------------------------------
GEMMA4 = SyntaxProfile(
    name="gemma4",
    description="Gemma: <|tool_call>call:NAME{key:value}<tool_call|>",
    call_start="<|tool_call>",
    call_end="<tool_call|>",
    function_tag_open="call:",
    function_name_terminator="{",
    function_tag_close="",
    argument_style="json",
    reasoning_start="<|channel>thought",
    reasoning_end="<channel|>",
    options={"unquoted_keys": True},
)

# ---------------------------------------------------------------------------
# Plain "no special syntax": the model is expected to emit a JSON object.
# Useful for JSON-mode models with tools described in the prompt.
# ---------------------------------------------------------------------------
PLAIN_JSON = SyntaxProfile(
    name="plain-json",
    description="No markers: parse a bare JSON object {\"name\":...,\"arguments\":{...}}",
    call_start="",
    call_end="",
    function_tag_open="",
    function_name_terminator="",
    function_tag_close="",
    argument_style="json",
    reasoning_start="<think>",
    reasoning_end="</think>",
    options={"call_object_keys": ("name", "arguments", "parameters")},
)


_BUILTINS: Dict[str, SyntaxProfile] = {
    profile.name: profile
    for profile in (
        QWEN,
        QWEN_JSON,
        HERMES,
        MISTRAL,
        MINISTRAL3,
        LLAMA3,
        DEEPSEEK,
        GPT_OSS,
        KIMI,
        LFM2,
        GEMMA4,
        SPARK,
        PLAIN_JSON,
    )
}

#: Aliases so users can say ``--profile qwen3`` / ``--profile mistral``.
_ALIASES: Dict[str, str] = {
    "qwen3": "qwen",
    "qwen3.6": "qwen",
    "qwen-3": "qwen",
    "qwen2.5": "qwen",
    "qwen3-coder": "qwen",
    "hermes": "hermes",
    "nous": "hermes",
    "mistral-small": "mistral",
    "magistral": "ministral3",
    "llama": "llama3",
    "llama3.1": "llama3",
    "llama3.2": "llama3",
    "deepseek-v3": "deepseek",
    "deepseek-v3.2": "deepseek",
    "deepseek-r1": "deepseek",
    "gpt-oss": "gpt-oss",
    "gpt_oss": "gpt-oss",
    "harmony": "gpt-oss",
    "kimi-k2": "kimi",
    "kimi-k2-thinking": "kimi",
    "lfm": "lfm2",
    "lfm2.5": "lfm2",
    "gemma": "gemma4",
    "gemma4": "gemma4",
    "json": "plain-json",
    "plain": "plain-json",
    "none": "plain-json",
    "spark": "spark",
    "spark-x": "spark",
    "spark-x2.6": "spark",
}


def available_profiles() -> List[str]:
    """All profile names (without aliases), sorted."""
    return sorted(_BUILTINS)


def register_profile(profile: SyntaxProfile, *, replace: bool = False) -> None:
    """Register a custom profile so it can be selected by name."""
    if profile.name in _BUILTINS and not replace:
        raise ConfigError(f"profile {profile.name!r} already exists; pass replace=True")
    _BUILTINS[profile.name] = profile


def get_profile(name: str) -> SyntaxProfile:
    """Look up a built-in profile by name or alias."""
    key = (name or "").strip().lower()
    if key in _BUILTINS:
        return _BUILTINS[key]
    if key in _ALIASES:
        return _BUILTINS[_ALIASES[key]]
    raise ConfigError(
        f"unknown profile {name!r}; available: {', '.join(available_profiles())} "
        f"(or pass --profile-file)"
    )


def load_profile_file(path: str) -> SyntaxProfile:
    """Load a ``SyntaxProfile`` from a JSON file (see ``--dump-profile-example``)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        raise ConfigError(f"cannot read profile file {path!r}: {exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: profile must be a JSON object")
    try:
        return SyntaxProfile.from_dict(data)
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def detect_profile_from_template(source: str) -> Optional[str]:
    """Best-effort detection of a built-in profile from a jinja template.

    Only used to *suggest* a profile (``--auto-profile``); never silently
    applied unless the user asks for it.
    """
    text = source or ""
    checks = (
        ("gpt-oss", ("<|channel|>", "to=functions.")),
        ("kimi", ("<|tool_call_argument_begin|>",)),
        ("deepseek", ("｜DSML｜",)),
        ("ministral3", ("[THINK]", "[ARGS]")),
        ("gemma4", ("<|tool_call>", "call:")),
        ("lfm2", ("<|tool_call_start|>",)),
        ("spark", ("<arg_key>", "<arg_value>")),
        ("qwen", ("<function=", "<parameter=")),
        ("mistral", ("[TOOL_CALLS]",)),
        ("llama3", ("<|python_tag|>",)),
    )
    for name, markers in checks:
        if any(marker in text for marker in markers):
            return name
    if "<tool_call>" in text and "tojson" in text:
        return "hermes"
    return None


def dumps_profile(profile: SyntaxProfile) -> str:
    return json.dumps(profile.to_dict(), indent=2, ensure_ascii=False)


def dump_profiles() -> str:
    """Human readable table of all profiles."""
    lines = []
    for name in available_profiles():
        profile = _BUILTINS[name]
        lines.append(f"{name:<14} {profile.description}")
    return "\n".join(lines)


def write_example_profile(path: str) -> None:
    """Write a commented example profile users can copy and edit."""
    example = SyntaxProfile(
        name="my-model",
        description="Example: copy me and adjust the markers",
        call_start="<tool_call>",
        call_end="</tool_call>",
        function_tag_open="<function=",
        function_name_terminator=">",
        function_tag_close="</function>",
        argument_style="xml_params",
        parameter_tag_open="<parameter=",
        reasoning_start="<think>",
        reasoning_end="</think>",
        extra_stop_strings=[],
        options={},
    )
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumps_profile(example) + "\n")
