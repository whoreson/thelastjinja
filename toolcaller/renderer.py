"""Render an OpenAI chat request into a raw text-completion prompt.

The rendering pipeline replicates ``common_chat_templates_apply_jinja`` from
``ik_llama.cpp/common/chat.cpp``:

1. normalise OpenAI messages  (:mod:`toolcaller.messages`)
2. analyse the template once  (:mod:`toolcaller.caps`)
3. apply llama.cpp's required *workarounds*

   * map ``developer`` -> ``system`` (except for harmony/GPT-OSS templates)
   * if the template cannot represent a system role, fold the system message
     into the first user message
   * if the template supports tool calls, make sure ``content`` is never null
   * if the template indexes ``arguments`` as an object, send an object

4. render twice - once with ``add_generation_prompt=false`` and once with
   ``true`` - and keep the rendered **suffix** as the *generation prompt*.
   The suffix is the piece of text the template expects the model to continue
   from (``<|im_start|>assistant\\n<think>\\n`` for Qwen).  Knowing it lets us

   * feed it to the server's grammar sampler (``grammar_prefill``) and
   * re-attach it to the model output before tool-call parsing

5. prepend ``bos_token`` only when the template itself does not emit it.
"""

from __future__ import annotations

import datetime as _datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .caps import TemplateCaps, analyze_caps
from .environment import TemplateEnvironment
from .errors import RequestError
from .messages import messages_to_json, normalize_messages

#: Templates that implement the OpenAI "harmony" format understand the
#: ``developer`` role natively; everything else gets it mapped to ``system``.
HARMONY_MARKER = "<|channel|>"


@dataclass
class RenderResult:
    """Everything the caller needs after rendering a chat request."""

    prompt: str
    """The full prompt string to POST to ``/completion``."""

    generation_prompt: str
    """Suffix added by the template for ``add_generation_prompt=true``."""

    chat_messages: List[Dict[str, Any]]
    """Messages exactly as handed to the template (post-workarounds)."""

    caps: TemplateCaps
    add_bos: bool = False
    add_eos: bool = False
    warnings: List[str] = field(default_factory=list)
    hints: Any = None
    """``toolcaller.hints.TokenHints`` used for this render (or ``None``)."""

    @property
    def prompt_for_parser(self) -> str:
        """Prompt that the generation prompt was appended to."""
        if self.generation_prompt and self.prompt.endswith(self.generation_prompt):
            return self.prompt[: -len(self.generation_prompt)]
        return self.prompt


def map_developer_role_to_system(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """llama.cpp's ``workaround::map_developer_role_to_system``."""
    out = []
    for msg in messages:
        if msg.get("role") == "developer":
            msg = dict(msg)
            msg["role"] = "system"
        out.append(msg)
    return out


def system_message_not_supported(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold a leading system message into the first user message."""
    if not messages:
        return messages
    system_texts: List[str] = []
    while messages and messages[0].get("role") == "system":
        msg = messages.pop(0)
        content = msg.get("content")
        if isinstance(content, str):
            system_texts.append(content)
        elif isinstance(content, list):
            system_texts.extend(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
    if not system_texts:
        return messages
    prefix = "\n\n".join(t for t in system_texts if t)
    for index, msg in enumerate(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                merged = _remove_special_token_fragments(content)
                msg = dict(msg)
                msg["content"] = prefix + "\n\n" + merged
                messages[index] = msg
            else:
                parts = list(content or [])
                parts.insert(0, {"type": "text", "text": prefix})
                msg = dict(msg)
                msg["content"] = parts
                messages[index] = msg
            return messages
    # no user message: keep the system text as a user message
    messages.insert(0, {"role": "user", "content": prefix})
    return messages


_SPECIAL_TOKEN_FRAGMENTS = ("<bos>", "<|begin_of_text|>", "<|start_header_id|>")

_BOS_MARKERS = (
    "<s>",
    "<|endoftext|>",
    "<|begin_of_text|>",
    "<s>",
)


def _remove_special_token_fragments(text: str) -> str:
    """Strip literal special-token strings from user text (llama.cpp does this)."""
    for fragment in _SPECIAL_TOKEN_FRAGMENTS:
        text = text.replace(fragment, "")
    return text


def requires_non_null_content(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """llama.cpp's ``workaround::requires_non_null_content``."""
    out = []
    for msg in messages:
        if msg.get("tool_calls") and msg.get("content") is None:
            msg = dict(msg)
            msg["content"] = ""
        out.append(msg)
    return out


#: ``reasoning_effort`` values that mean "do not think".
_THINKING_OFF_EFFORTS = {"none", "off", "disabled", "minimal", ""}
#: ``reasoning_effort`` values that mean "think", with the matching template hint.
_THINKING_EFFORT_HINTS = {"low": "low", "medium": "high", "high": "high"}


def resolve_thinking(
    request: Optional[Dict[str, Any]],
) -> tuple:
    """Decide whether the model should think, and at what effort.

    Clients signal this in several incompatible ways, so all of them are
    accepted (first match wins):

    ==========================================  ==================
    signal                                      meaning
    ==========================================  ==================
    ``enable_thinking: false``                  off
    ``chat_template_kwargs.enable_thinking``    off/on
    ``reasoning_effort: "none"``                off
    ``reasoning_effort: "low"``                 on, effort low
    ``reasoning_effort: "medium"``/``"high"``   on, effort high
    ``thinking: {type: "disabled"}``            off
    ``thinking: {type: "enabled"}``             on
    ``reasoning: {enabled: false}``             off
    ``reasoning_enabled: false``                off
    ``reasoning.max_tokens: 0``                 off
    *(nothing)*                                 on (server default)
    ==========================================  ==================

    Returns ``(enabled, effort)`` where ``effort`` is ``None`` for the
    template's default.
    """
    if not request:
        return True, None

    def as_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("false", "0", "no", "off", "none", "disabled"):
                return False
            if lowered in ("true", "1", "yes", "on", "enabled"):
                return True
        if isinstance(value, (int, float)):
            return bool(value)
        return None

    # 1. Explicit template variables win (enable_thinking / chat_template_kwargs).
    for container in (request, request.get("chat_template_kwargs")):
        if isinstance(container, dict) and "enable_thinking" in container:
            flag = as_bool(container["enable_thinking"])
            if flag is not None:
                return flag, None

    # 2. reasoning_effort: absence means "off" for some clients, but the
    #    server default (on) is kept - only explicit values are acted upon.
    if "reasoning_effort" in request:
        raw = request.get("reasoning_effort")
        if raw is None:
            return False, None
        effort = str(raw).strip().lower()
        if effort in _THINKING_OFF_EFFORTS:
            return False, None
        return True, _THINKING_EFFORT_HINTS.get(effort, None)

    # 3. Nested thinking / reasoning objects.
    thinking = request.get("thinking")
    if isinstance(thinking, dict):
        kind = thinking.get("type")
        if isinstance(kind, str):
            if kind.strip().lower() in ("disabled", "none", "off"):
                return False, None
            if kind.strip().lower() in ("enabled", "on"):
                return True, None
        budget = thinking.get("budget_tokens")
        if isinstance(budget, (int, float)) and budget <= 0:
            return False, None
    if isinstance(thinking, bool):
        return thinking, None

    reasoning = request.get("reasoning")
    if isinstance(reasoning, dict):
        if "enabled" in reasoning:
            flag = as_bool(reasoning["enabled"])
            if flag is not None:
                return flag, None
        max_tokens = reasoning.get("max_tokens")
        if isinstance(max_tokens, (int, float)) and max_tokens <= 0:
            return False, None
    if isinstance(reasoning, bool):
        return reasoning, None

    if "reasoning_enabled" in request:
        flag = as_bool(request["reasoning_enabled"])
        if flag is not None:
            return flag, None

    return True, None


def _extra_context(now: Optional[_datetime.datetime] = None) -> Dict[str, Any]:
    """llama.cpp's ``common_chat_extra_context``."""
    now = now or _datetime.datetime.now()
    return {
        "datetime": now.strftime("%b %d %Y"),
        "date_string": now.strftime("%d %b %Y"),
    }


class GenerationPromptError(Exception):
    """The add_generation_prompt suffix could not be determined.

    Not a fatal error: callers fall back to an empty generation prompt and add
    a warning, because the suffix is only an optimisation hint.
    """


def _split_generation_prompt(full: str, base: str) -> str:
    """Return the suffix ``full`` adds on top of ``base`` (longest match)."""
    if not base:
        return full
    if full.startswith(base):
        return full[len(base):]
    # Templates with time/random-dependent content make the two renders differ
    # in the middle.  Fall back to the longest common prefix; if even that is
    # not a prefix of ``full`` the caller gets "" (no generation prompt), which
    # costs nothing but the grammar-prefill hint.
    limit = min(len(full), len(base))
    index = 0
    while index < limit and full[index] == base[index]:
        index += 1
    if index >= limit:
        return full[index:]
    raise GenerationPromptError(
        "the template does not produce a stable prefix for the generation prompt "
        "(add_generation_prompt changes earlier output); continuing without a "
        "generation-prompt suffix"
    )


class PromptRenderer:
    """Turns OpenAI chat requests into prompts using one jinja template."""

    def __init__(
        self,
        env: TemplateEnvironment,
        *,
        bos_token: str = "",
        eos_token: str = "",
        add_bos: bool = False,
        add_eos: bool = False,
        default_template_kwargs: Optional[Dict[str, Any]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
        strict: bool = True,
    ) -> None:
        self.env = env
        self.bos_token = bos_token or ""
        self.eos_token = eos_token or ""
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.default_template_kwargs = dict(default_template_kwargs or {})
        self.extra_context = _extra_context() if extra_context is None else dict(extra_context)
        self.strict = strict
        self.caps = analyze_caps(env)

    # -- public API --------------------------------------------------------
    def render(
        self,
        request: Dict[str, Any],
        *,
        template_kwargs: Optional[Dict[str, Any]] = None,
        generation_prompt_diff: bool = True,
        now: Optional[_datetime.datetime] = None,
    ) -> RenderResult:
        request = self._check_request(request)
        chat_messages = normalize_messages(request.get("messages"))

        tools = request.get("tools") or []
        if tools and not self.caps.supports_tools:
            if self.strict:
                raise RequestError(
                    "this template does not render tool definitions ('tools'), so the model "
                    "would never see them; drop 'tools' or use a template with tool support"
                )
            tools = []
        if tools:
            self._check_parallel(request, tools)

        chat_messages = apply_message_workarounds(chat_messages, self.caps, self.env.source)

        context = self._build_context(
            chat_messages,
            tools,
            template_kwargs=template_kwargs,
            now=now,
            request=request,
        )
        warnings: List[str] = []

        prompt = self.env.render(dict(context, add_generation_prompt=True))
        generation_prompt = ""
        if generation_prompt_diff:
            base_context = self._build_context(
                chat_messages, tools, template_kwargs=template_kwargs, now=now, request=request
            )
            base_context.pop("add_generation_prompt", None)
            try:
                base_prompt = self.env.render(base_context)
                generation_prompt = _split_generation_prompt(prompt, base_prompt)
            except GenerationPromptError as exc:
                warnings.append(str(exc))

        if self.add_bos and self.bos_token and not prompt.startswith(self.bos_token):
            prompt = self.bos_token + prompt

        return RenderResult(
            prompt=prompt,
            generation_prompt=generation_prompt,
            chat_messages=context["messages"],
            caps=self.caps,
            add_bos=self.add_bos,
            add_eos=self.add_eos,
            warnings=warnings,
        )

    def render_prompt(self, request: Dict[str, Any], **kwargs: Any) -> str:
        return self.render(request, **kwargs).prompt

    # -- internals ---------------------------------------------------------
    def _check_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(request, dict):
            raise RequestError("request body must be a JSON object")
        if "messages" not in request:
            raise RequestError("request body must contain 'messages'")
        return request

    def _check_parallel(self, request: Dict[str, Any], tools: Sequence[Any]) -> None:
        if self.caps.supports_parallel_tool_calls:
            return
        parallel = request.get("parallel_tool_calls")
        if parallel:
            raise RequestError("this template does not support parallel tool calls")

    def _build_context(
        self,
        messages: List[Dict[str, Any]],
        tools: Sequence[Any],
        *,
        template_kwargs: Optional[Dict[str, Any]],
        now: Optional[_datetime.datetime],
        request: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "messages": messages_to_json(
                messages,
                object_arguments=self.caps.supports_object_arguments,
                concat_typed_text=self._concat_typed_text(),
            ),
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        if tools:
            context["tools"] = _tools_to_json(tools)
        # Thinking is controlled by several client conventions; see
        # ``resolve_thinking``.
        enabled, effort = resolve_thinking(request)
        if self.caps.supports_enable_thinking:
            # Templates with ``enable_thinking`` get it explicitly; default on
            # unless the client asked otherwise.
            context["enable_thinking"] = enabled
        if effort:
            # ``reasoning_effort`` is separate: some templates (GLM) have no
            # on/off switch at all and only honour the effort *level*, so it
            # must be forwarded even when ``enable_thinking`` is unsupported.
            context["reasoning_effort"] = effort
        context.update(self.extra_context if now is None else _extra_context(now))
        kwargs = dict(self.default_template_kwargs)
        if template_kwargs:
            kwargs.update(template_kwargs)
        context.update(kwargs)
        return context

    def _concat_typed_text(self) -> bool:
        """Templates that do ``content + ...`` need a string, not parts."""
        return self.caps.supports_string_content and not self.caps.supports_typed_content


def apply_message_workarounds(
    messages: List[Dict[str, Any]], caps: TemplateCaps, template_source: str
) -> List[Dict[str, Any]]:
    """Full llama.cpp workaround chain given template knowledge."""
    messages = [dict(m) for m in messages]
    if HARMONY_MARKER not in template_source:
        messages = map_developer_role_to_system(messages)
    if not caps.supports_system_role:
        messages = system_message_not_supported(messages)
    if caps.supports_tool_calls:
        messages = requires_non_null_content(messages)
    return messages


def _tools_to_json(tools: Sequence[Any]) -> List[Dict[str, Any]]:
    """Normalise the OpenAI ``tools`` array.

    Mirrors ``common_chat_tools_to_json_oaicompat``: only ``type`` /
    ``function.name`` / ``function.description`` / ``function.parameters`` are
    kept, so templates that iterate ``tool | items`` see a stable shape.
    """
    out: List[Dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise RequestError(f"tools[{index}] must be an object")
        tool_type = tool.get("type", "function")
        if tool_type != "function":
            raise RequestError(f"tools[{index}]: unsupported tool type {tool_type!r}")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise RequestError(f"tools[{index}] is missing 'function'")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise RequestError(f"tools[{index}].function.name is required")
        parameters = function.get("parameters")
        if parameters is None:
            parameters = {"type": "object", "properties": {}}
        elif not isinstance(parameters, dict):
            raise RequestError(f"tools[{index}].function.parameters must be an object")
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": function.get("description", "") or "",
                    "parameters": parameters,
                },
            }
        )
    return out
