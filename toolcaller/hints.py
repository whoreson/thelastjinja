"""Optional prompt-side hints derived from the parser profile.

Raw completion endpoints have no way to enforce the tool-call format (unless
a grammar is used).  A short, model-agnostic reminder appended to the system
prompt dramatically improves the rate at which the model emits *exactly* the
syntax the parser understands.

Hints are **opt-in** (``--hint-mode``) because they change the prompt.  The
default (``auto``) only adds a hint when tool definitions are present *and*
the template itself does not already explain the format.

The generated text is deliberately derived from the profile so that a new
model family gets correct hints for free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .syntax import SyntaxProfile

#: ``hint_mode`` values.
HINT_MODES = ("auto", "always", "never")


@dataclass
class TokenHints:
    """Prompt-side additions for one request."""

    system_suffix: str = ""
    """Text appended to the end of the system prompt (or first user turn)."""

    grammar: Optional[str] = None
    """GBNF grammar (when ``--grammar-style gbnf``)."""

    grammar_triggers: List[Dict[str, Any]] = field(default_factory=list)
    """llama.cpp ``grammar_triggers`` (lazy grammars)."""

    preserved_tokens: List[str] = field(default_factory=list)
    """Special tokens the sampler must not treat as ordinary text."""

    def is_empty(self) -> bool:
        return not (self.system_suffix or self.grammar or self.grammar_triggers)


def _example_call(profile: SyntaxProfile, tool_name: str,
                  sample_args: Sequence[str] = ("city",)) -> str:
    """Render a small example call in the profile's syntax."""
    args = {name: "value" for name in sample_args}
    if profile.name == "mistral":
        payload = json.dumps(
            [{"name": tool_name, "arguments": args, "id": "call_00001"}], ensure_ascii=False
        )
        return f"{profile.call_start} {payload}"
    if profile.name == "gpt-oss":
        return (
            f"{profile.reasoning_end or '<|end|>'}"
            f"<|start|>assistant<|channel|>commentary to=functions.{tool_name}"
            f"<|constrain|>json<|message|>{json.dumps(args)}<|call|>"
        )
    if profile.name == "kimi":
        return (
            f"{profile.call_start}functions.{tool_name}:0"
            f"<|tool_call_argument_begin|>{json.dumps(args)}<|tool_call_end|>"
        )
    if profile.name == "deepseek":
        params = "".join(
            f'<｜DSML｜parameter name="{key}" string="true">{value}</｜DSML｜parameter>'
            for key, value in args.items()
        )
        return (
            f'<｜DSML｜invoke name="{tool_name}">{params}</｜DSML｜invoke>'
        )
    if profile.name == "lfm2":
        kwargs = ", ".join(f'{key}="{value}"' for key, value in args.items())
        return f"{profile.call_start}[{tool_name}({kwargs})]{profile.call_end}"
    if profile.name == "gemma4":
        body = "{" + ", ".join(f'{key}:"{value}"' for key, value in args.items()) + "}"
        return f"{profile.call_start}call:{tool_name}{body}{profile.call_end}"
    if profile.name == "ministral3":
        return f"{profile.call_start}{tool_name}[ARGS]{json.dumps(args)}"

    # default: <function=NAME><parameter=K>V</parameter></function> style
    if profile.argument_style == "xml_params":
        params = "".join(
            f"{profile.parameter_tag_open}{key}{profile.function_name_terminator}\n"
            f"{value}\n</parameter>"
            for key, value in args.items()
        )
        inner = f"{profile.function_tag_open}{tool_name}{profile.function_name_terminator}{params}"
        if profile.function_tag_close:
            inner += profile.function_tag_close
        return f"{profile.call_start}\n{inner}\n{profile.call_end}"

    payload = json.dumps({"name": tool_name, "arguments": args}, ensure_ascii=False)
    if profile.call_start:
        return f"{profile.call_start}\n{payload}\n{profile.call_end}"
    return payload


def build_hints(
    profile: SyntaxProfile,
    *,
    has_tools: bool,
    template_source: str = "",
    mode: str = "auto",
    grammar_style: str = "none",
    tool_names: Sequence[str] = (),
) -> TokenHints:
    """Build prompt-side hints.

    Parameters
    ----------
    mode:
        ``always`` - always add the format reminder when tools exist
        ``auto``   - only when the template does not already describe the format
        ``never``  - never add anything
    grammar_style:
        ``none`` | ``gbnf`` | ``none-prefix``
    """
    if mode not in HINT_MODES:
        from .errors import ConfigError

        raise ConfigError(f"hint_mode must be one of {HINT_MODES}, got {mode!r}")

    hints = TokenHints(preserved_tokens=_preserved_tokens(profile))
    if not has_tools or mode == "never":
        if grammar_style == "gbnf" and has_tools:
            hints.grammar, hints.grammar_triggers = _tool_grammar(profile)
        return hints

    template_mentions_format = any(
        marker and marker in template_source
        for marker in (profile.call_start, profile.function_tag_open)
    )
    if mode == "auto" and template_mentions_format:
        # The template already documents the syntax; do not duplicate it.
        if grammar_style == "gbnf":
            hints.grammar, hints.grammar_triggers = _tool_grammar(profile)
        return hints

    example = _example_call(profile, tool_names[0] if tool_names else "example_function")
    hint = _hint_text(profile, example)
    hints.system_suffix = hint
    if grammar_style == "gbnf":
        hints.grammar, hints.grammar_triggers = _tool_grammar(profile)
    return hints


def _hint_text(profile: SyntaxProfile, example: str) -> str:
    if profile.options.get("harmony"):
        return (
            "\n\n# Tool calls\n"
            "To call a function, emit a commentary message with the recipient set to the "
            "function name, then the JSON arguments, then <|call|>. Example:\n"
            f"{example}\n"
            "Do not write tool calls inside the final channel."
        )
    if profile.name == "deepseek":
        return (
            "\n\n# Tool calls\n"
            "Tool calls use the DSML XML format inside <｜DSML｜function_calls>. Example:\n"
            f"<｜DSML｜function_calls>\n{example}\n</｜DSML｜function_calls>\n"
            "Only use the tools listed above; arguments must satisfy the JSON schema."
        )
    if profile.name in ("mistral", "ministral3"):
        return (
            "\n\n# Tool calls\n"
            "To call a tool, use the available tools format exactly. Example:\n"
            f"{example}\n"
            "Emit the call without any surrounding prose."
        )
    if profile.name == "kimi":
        return (
            "\n\n# Tool calls\n"
            "Tool calls are wrapped in <|tool_calls_section_begin|> and "
            "<|tool_calls_section_end|>. Example:\n"
            f"{example}\n"
            "The function id has the form functions.<name>:<index>."
        )
    if profile.name == "lfm2":
        return (
            "\n\n# Tool calls\n"
            "Call functions using the python-style list form. Example:\n"
            f"{example}\n"
            "Use double-quoted keyword arguments."
        )
    if profile.name == "gemma4":
        return (
            "\n\n# Tool calls\n"
            "Call functions with the call: syntax. Example:\n"
            f"{example}\n"
        )
    # generic XML / JSON
    if profile.argument_style == "xml_params":
        return (
            "\n\nIf you choose to call a function, respond ONLY in the following format "
            "with no suffix:\n\n"
            f"{example}\n\n"
            "<IMPORTANT>\n"
            "Function calls MUST follow the specified format: an inner "
            f"{profile.function_tag_open}...{profile.function_tag_close or '>'} block must be nested "
            f"within {profile.call_start}...{profile.call_end} tags. "
            "Required parameters MUST be specified. "
            "You may provide optional reasoning BEFORE the function call, but NOT after.\n"
            "</IMPORTANT>"
        )
    return (
        "\n\nIf you choose to call a function, reply with JSON only, e.g.\n"
        f"{example}\n"
        "Do not add any text after the call."
    )


def _preserved_tokens(profile: SyntaxProfile) -> List[str]:
    tokens = [
        profile.call_start,
        profile.call_end,
        profile.function_tag_open,
        profile.function_tag_close,
        profile.parameter_tag_open,
        profile.reasoning_start or "",
        profile.reasoning_end or "",
    ]
    tokens.extend(profile.extra_stop_strings)
    seen: List[str] = []
    for token in tokens:
        if token and token not in seen:
            seen.append(token)
    return seen


def _tool_grammar(profile: SyntaxProfile) -> tuple:
    """A deliberately small GBNF grammar that permits free text + one call.

    Only used with ``--grammar-style gbnf``.  It is *lazy*: the grammar only
    activates once the model starts writing the call start marker, so normal
    prose (and chain of thought) is unaffected.

    The grammar accepts ``[call_start] <call_body> [call_end]`` rather than
    the exact argument schema: a wrong-but-parseable call is far more useful
    than a hard constraint that makes the model emit garbage - and the
    Python parser validates/normalises the arguments anyway.  Use
    ``json_schema``/``tools`` on a ``--jinja`` server if you want strict
    argument constraining.
    """
    start = profile.call_start
    if not start:
        return None, []
    end = profile.call_end or ""
    triggers = [{"type": 1, "value": start}]  # 1 == COMMON_GRAMMAR_TRIGGER_TYPE_WORD

    if profile.options.get("array_of_calls") or profile.argument_style == "json_array":
        grammar = (
            "root ::= prose? head\n"
            "prose ::= [^\\[]+\n"
            f"head ::= {_gbnf_literal(start)} space json space\n"
            "json ::= \"[\" space (json-call (space \",\" space json-call)*)? space \"]\"\n"
            'json-call ::= "{" space json-member (space "," space json-member)* space "}"\n'
            + _JSON_MEMBER_RULES
            + _JSON_RULES
        )
        return grammar, triggers

    # Everything else: free prose, then exactly one call block whose body is
    # unconstrained except that it must not swallow the closing marker.
    grammar = (
        "root ::= prose? head\n"
        "prose ::= [^<]+ | \"<\" [^!]\n"
        f"head ::= {_gbnf_literal(start)} body{_gbnf_optional(end)}\n"
        f"body ::= {_gbnf_not_char(end[:1])}*\n"
    )
    return grammar, triggers


def _gbnf_literal(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _gbnf_optional(text: str) -> str:
    return (" " + _gbnf_literal(text)) if text else ""


def _gbnf_not_char(first_char: str) -> str:
    """GBNF negated char class that excludes ``first_char``."""
    if not first_char:
        return "[^]"
    if first_char in "\\]^-":
        first_char = "\\" + first_char
    return f"[^{first_char}]"


_JSON_MEMBER_RULES = """
json-member ::= json-string space ":" space json-value
"""


_JSON_RULES = """
json-value ::= json-object | json-string | json-number | json-bool | json-null
json-object ::= "{" space (json-string space ":" space json-value (space "," space json-string space ":" space json-value)*)? space "}"
json-string ::= "\\"" ( [^"\\\\] | "\\\\" . )* "\\""
json-number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
json-bool ::= "true" | "false"
json-null ::= "null"
space ::= [ \\t\\n\\r]*
"""
