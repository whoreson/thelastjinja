"""High level orchestration: chat request in, chat completion out.

    request (OpenAI JSON)
        -> PromptRenderer            (jinja -> prompt + generation_prompt)
        -> TokenHints                (profile -> prompt hints / grammar)
        -> CompletionClient          (/completion, token-exact stops)
        -> SyntaxParser              (raw text -> content/reasoning/tool_calls)
        -> ChatResult                (OpenAI chat.completion JSON)
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .caps import TemplateCaps, analyze_caps
from .client import CompletionClient, CompletionResult, format_timings
from .constraints import (
    GenerationConstraints,
    build_constraints,
    strip_generation_reasoning,
)
from .environment import TemplateEnvironment, load_template_file
from .errors import ConfigError, ParseError, RequestError, ServerError
from .hints import HINT_MODES, TokenHints, build_hints
from .messages import normalize_request
from .parser import (
    REASONING_FORMATS,
    ParsedMessage,
    ParserSelection,
    SyntaxParser,
    parse_completion,
    resolve_profile,
)
from .profiles import SyntaxProfile, get_profile
from .renderer import PromptRenderer, RenderResult


#: Sampling parameters we forward verbatim to ``/completion``.
SAMPLING_PARAMS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "tfs_z",
    "repeat_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "penalize_nl",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "xtc_probability",
    "xtc_threshold",
    "dynatemp_range",
    "dynatemp_exponent",
    "adaptive_target",
    "adaptive_decay",
    "seed",
    "logit_bias",
    "min_keep",
    "samplers",
    "banned_strings",
    "n_probs",
    "ignore_eos",
)

#: Request keys that are OpenAI-specific and must not be forwarded.
OPENAI_ONLY_KEYS = frozenset(
    {
        "messages",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "functions",
        "function_call",
        "response_format",
        "stream",
        "stop",
        "max_tokens",
        "max_completion_tokens",
        "n",
        "model",
        "user",
        "metadata",
        "store",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "seed",
        "service_tier",
        "frequency_penalty",
        "presence_penalty",
        "temperature",
        "top_p",
        "reasoning_format",
        "chat_template_kwargs",
        "chat_template",
    }
)


@dataclass
class ChatResult:
    """Result of one chat completion."""

    message: Dict[str, Any]
    finish_reason: str
    completion: CompletionResult
    parsed: ParsedMessage
    render: RenderResult
    profile: SyntaxProfile
    warnings: List[str] = field(default_factory=list)
    elapsed: float = 0.0
    model: str = ""

    # -- OpenAI-shaped output ---------------------------------------------
    def to_response(self, *, request_id: Optional[str] = None,
                    created: Optional[int] = None, model: Optional[str] = None,
                    include_timings: bool = True) -> Dict[str, Any]:
        created = created if created is not None else int(time.time())
        response: Dict[str, Any] = {
            "id": request_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": created,
            "model": model or self.model or "unknown",
            "choices": [
                {
                    "index": 0,
                    "message": self.message,
                    "logprobs": None,
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": self.usage(),
        }
        if include_timings:
            response["timings"] = self.completion.timings
        return response

    def usage(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.completion.tokens_evaluated,
            "completion_tokens": self.completion.tokens_predicted,
            "total_tokens": self.completion.tokens_evaluated
            + self.completion.tokens_predicted,
        }

    def stream_chunks(self, *, request_id: Optional[str] = None,
                      created: Optional[int] = None, model: Optional[str] = None,
                      ) -> List[Dict[str, Any]]:
        """Split this result into OpenAI ``chat.completion.chunk`` objects.

        Reasoning is streamed first, then content, then one chunk per tool
        call, then the ``finish_reason`` chunk.  This mirrors what a
        chat-completions endpoint would send, which is all a client needs.
        """
        chunk_id = request_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = created if created is not None else int(time.time())
        model_name = model or self.model or "unknown"

        def chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None) -> Dict[str, Any]:
            return {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": delta, "logprobs": None,
                     "finish_reason": finish_reason}
                ],
            }

        chunks = [chunk({"role": "assistant", "content": ""})]
        if self.parsed.reasoning_content:
            chunks.append(chunk({"reasoning_content": self.parsed.reasoning_content}))
        if self.message.get("content"):
            chunks.append(chunk({"content": self.message["content"]}))
        for index, call in enumerate(self.parsed.tool_calls):
            chunks.append(
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(
                                        call.arguments,
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    ),
                                },
                            }
                        ]
                    }
                )
            )
        chunks.append(chunk({}, self.finish_reason))
        return chunks


class ChatPipeline:
    """Everything wired together, with per-request overrides."""

    def __init__(
        self,
        *,
        template_path: Optional[str] = None,
        template_source: Optional[str] = None,
        base_url: str = "http://localhost:8082",
        profile: Optional[str] = None,
        profile_file: Optional[str] = None,
        auto_profile: bool = True,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        add_bos: bool = False,
        add_eos: bool = False,
        hint_mode: str = "auto",
        grammar_style: str = "none",
        reasoning_format: str = "auto",
        template_kwargs: Optional[Dict[str, Any]] = None,
        strict_template: bool = True,
        strip_generation_thinking: bool = False,
        extra_stops: Sequence[str] = (),
        stop_at_call_end: bool = False,
        client: Optional[CompletionClient] = None,
        timeout: float = 3600.0,
        discover_tokens: bool = True,
        debug: bool = False,
    ) -> None:
        if template_source is None:
            if template_path is None:
                raise ConfigError("either template_path or template_source is required")
            self.env = load_template_file(template_path)
            self.template_path = template_path
        else:
            self.env = TemplateEnvironment(template_source, name="<inline template>")
            self.template_path = None

        self.client = client or CompletionClient(base_url, timeout=timeout)

        if hint_mode not in HINT_MODES:
            raise ConfigError(f"hint_mode must be one of {HINT_MODES}")
        if reasoning_format not in REASONING_FORMATS:
            raise ConfigError(
                f"reasoning_format must be one of {', '.join(REASONING_FORMATS)}"
            )
        self.hint_mode = hint_mode
        self.grammar_style = grammar_style
        self.reasoning_format = reasoning_format
        self.template_kwargs = dict(template_kwargs or {})
        self.strip_generation_thinking = strip_generation_thinking
        self.extra_stops = tuple(extra_stops)
        self.stop_at_call_end = stop_at_call_end
        self.add_bos_flag = add_bos
        self.add_eos_flag = add_eos
        self.debug = debug
        self.discover_tokens = discover_tokens
        self.warnings: List[str] = []

        # tokens: explicit args win, else ask the server once
        self.bos_token = bos_token or ""
        self.eos_token = eos_token or ""
        if discover_tokens and (not self.bos_token or not self.eos_token):
            self._discover_tokens()

        self.selection = ParserSelection(
            profile=profile, profile_file=profile_file, auto_profile=auto_profile
        )
        self.profile, self.profile_hint = self._resolve_profile(profile, profile_file, auto_profile)
        self.caps: TemplateCaps = self.env and analyze_caps(self.env)

        self.renderer = PromptRenderer(
            self.env,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            add_bos=self.add_bos_flag,
            add_eos=self.add_eos_flag,
            default_template_kwargs=self.template_kwargs,
            strict=strict_template,
        )

    # -- setup helpers -----------------------------------------------------
    def _discover_tokens(self) -> None:
        try:
            props = self.client.props()
        except ServerError:
            return
        if not self.bos_token:
            self.bos_token = str(props.get("bos_token") or "")
        if not self.eos_token:
            self.eos_token = str(props.get("eos_token") or "")
        if self.debug:
            self.warnings.append(f"tokens from /props: bos={self.bos_token!r} eos={self.eos_token!r}")

    def _resolve_profile(
        self, profile: Optional[str], profile_file: Optional[str], auto_profile: bool
    ) -> tuple:
        if profile_file:
            from .profiles import load_profile_file

            return load_profile_file(profile_file), f"profile file {profile_file}"
        if profile:
            return get_profile(profile), f"--profile {profile}"
        if auto_profile:
            from .profiles import detect_profile_from_template

            detected = detect_profile_from_template(self.env.source)
            if detected:
                return get_profile(detected), f"auto-detected from template ({detected})"
        return get_profile("qwen"), "default profile (qwen)"

    # -- main entry points -------------------------------------------------
    def render(self, request: Dict[str, Any], **kwargs: Any) -> RenderResult:
        """Render a chat request to a prompt without calling the server."""
        request = normalize_request(request)
        return self.renderer.render(
            request,
            template_kwargs=self.template_kwargs,
            **kwargs,
        )

    def complete(
        self,
        request: Dict[str, Any],
        *,
        profile: Optional[str] = None,
        reasoning_format: Optional[str] = None,
        hint_mode: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        n_predict: Optional[int] = None,
        sampling: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        strict_parse: bool = False,
        raise_on_parse_error: bool = True,
        cancel: Optional[threading.Event] = None,
    ) -> ChatResult:
        started = time.time()
        request = normalize_request(request)
        warnings: List[str] = []

        # per-request profile override
        if profile:
            active_profile = get_profile(profile)
        else:
            active_profile = self.profile

        render = self._render_with_hints(
            request,
            active_profile,
            reasoning_format=reasoning_format,
            hint_mode=hint_mode,
        )
        warnings.extend(render.warnings)

        effective_reasoning = reasoning_format or self.reasoning_format
        tool_names = _tool_names(request)

        # stops & grammar
        constraints = build_constraints(
            active_profile,
            eos_token=self.eos_token,
            extra_stops=self.extra_stops,
            stop_at_call_end=self.stop_at_call_end,
            request_stops=_request_stops(request),
            has_tools=bool(tool_names),
        )
        constraints.merge(render.hints)

        # The EOS token must never be stripped from the sampler's vocabulary;
        # losing it makes models run past end-of-turn into the next turn.
        if self.eos_token and self.eos_token not in constraints.preserved_tokens:
            constraints.preserved_tokens.append(self.eos_token)

        prompt = render.prompt
        if self.strip_generation_thinking:
            # The template pre-fills the reasoning start marker (e.g. Qwen's
            # trailing `<think>`), so the model never emits `<think>` itself and
            # `reasoning_content` cannot be split off.  Dropping the marker makes
            # the model produce it - at the cost of a slightly looser prompt.
            stripped = strip_generation_reasoning(render.generation_prompt, active_profile)
            if stripped != render.generation_prompt:
                prompt = render.prompt_for_parser + stripped
                render.generation_prompt = stripped

        # When the generation prompt's reasoning marker was stripped, the output
        # starts inside the reasoning block; ``split_reasoning`` handles that by
        # accepting a leading ``</think>`` without a preceding ``<think>``.

        completion = self._run_completion(
            prompt,
            constraints=constraints,
            request=request,
            n_predict=n_predict,
            sampling=sampling,
            stream=stream,
            on_chunk=on_chunk,
            cancel=cancel,
        )
        warnings.extend(completion.warnings)

        text = completion.text
        text = _strip_generation_prompt(text, render)

        parser = SyntaxParser(
            active_profile,
            eos_token=self.eos_token,
            known_tool_names=tool_names,
            strict=strict_parse,
        )
        try:
            parsed = parser.parse(text, reasoning_format=effective_reasoning)
        except ParseError:
            if raise_on_parse_error:
                raise
            parsed = ParsedMessage(content=text, finish_reason=completion.finish_reason)
            warnings.append("tool-call parsing failed; returning raw text as content")

        if completion.stopped_limit and not parsed.tool_calls:
            parsed.finish_reason = "length"

        warnings.extend(parsed.warnings)
        message = parsed.to_openai_message()

        # tool_choice: "none" -> never surface tool calls
        if _tool_choice(request) == "none" and parsed.tool_calls:
            warnings.append("model emitted tool calls but tool_choice='none'; dropped them")
            for call in parsed.tool_calls:
                message["content"] = (message.get("content") or "")
            parsed.tool_calls = []
            message.pop("tool_calls", None)
            parsed.finish_reason = "stop"

        return ChatResult(
            message=message,
            finish_reason=parsed.finish_reason,
            completion=completion,
            parsed=parsed,
            render=render,
            profile=active_profile,
            warnings=warnings,
            elapsed=time.time() - started,
            model=completion.model,
        )

    # -- internals ---------------------------------------------------------
    def _render_with_hints(
        self,
        request: Dict[str, Any],
        active_profile: SyntaxProfile,
        *,
        reasoning_format: Optional[str],
        hint_mode: Optional[str],
    ) -> Any:
        """Render, optionally appending hint text to the last message."""
        mode = hint_mode or self.hint_mode
        tool_names = _tool_names(request)
        hints = build_hints(
            active_profile,
            has_tools=bool(tool_names),
            template_source=self.env.source,
            mode=mode,
            grammar_style=self.grammar_style,
            tool_names=tool_names,
        )

        render_request = request
        if hints.system_suffix:
            render_request = _append_hint(request, hints.system_suffix)

        render = self.renderer.render(
            render_request, template_kwargs=self.template_kwargs
        )
        # stash hints on the result for the caller
        setattr(render, "hints", hints)
        return render

    def _run_completion(
        self,
        prompt: str,
        *,
        constraints: GenerationConstraints,
        request: Dict[str, Any],
        n_predict: Optional[int],
        sampling: Optional[Dict[str, Any]],
        stream: bool,
        on_chunk: Optional[Callable[[str], None]],
        cancel: Optional[threading.Event] = None,
    ) -> CompletionResult:
        sampling_params = _sampling_from_request(request)
        if sampling:
            sampling_params.update(sampling)
        max_tokens = n_predict
        if max_tokens is None:
            max_tokens = _max_tokens(request)
        # llama.cpp uses -1 for "unlimited"; map OpenAI's None to that.
        n = -1 if max_tokens is None else int(max_tokens)
        if n == 0:
            n = -1

        stops = constraints.stops
        grammar = constraints.grammar
        triggers = constraints.grammar_triggers
        preserved = [t for t in constraints.preserved_tokens if t]

        if self.debug:
            self.warnings.append(
                f"completion request: stops={stops!r} grammar={'yes' if grammar else 'no'}"
            )

        if stream:
            return self.client.complete_stream(
                prompt,
                on_chunk=on_chunk,
                stops=stops,
                n_predict=n,
                grammar=grammar,
                grammar_triggers=triggers,
                preserved_tokens=preserved,
                sampling=sampling_params,
                cancel=cancel,
            )
        return self.client.complete(
            prompt,
            stops=stops,
            n_predict=n,
            grammar=grammar,
            grammar_triggers=triggers,
            preserved_tokens=preserved,
            sampling=sampling_params,
            cancel=cancel,
        )

    # -- misc --------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            "template": self.template_path or "<inline>",
            "profile": self.profile.name,
            "profile_selection": self.profile_hint,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "caps": self.caps.to_dict() if self.caps else {},
            "url": self.client.completion_url,
            "hint_mode": self.hint_mode,
            "reasoning_format": self.reasoning_format,
        }


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def _tool_names(request: Dict[str, Any]) -> List[str]:
    tools = request.get("tools") or []
    names: List[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") or {}
        name = function.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    for legacy in request.get("functions") or []:
        if isinstance(legacy, dict) and isinstance(legacy.get("name"), str):
            names.append(legacy["name"])
    return names


def _tool_choice(request: Dict[str, Any]) -> Any:
    choice = request.get("tool_choice")
    if isinstance(choice, dict):
        return "required"
    return choice or "auto"


def _request_stops(request: Dict[str, Any]) -> List[str]:
    stop = request.get("stop")
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, (list, tuple)):
        return [s for s in stop if isinstance(s, str)]
    return []


def _max_tokens(request: Dict[str, Any]) -> Optional[int]:
    for key in ("max_completion_tokens", "max_tokens"):
        value = request.get(key)
        if isinstance(value, int):
            return value
    return None


def _sampling_from_request(request: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key in SAMPLING_PARAMS:
        if key in request and request[key] is not None:
            params[key] = request[key]
    if "max_tokens" in params:
        params.pop("max_tokens")
    return params


def _append_hint(request: Dict[str, Any], hint: str) -> Dict[str, Any]:
    """Append ``hint`` to the (last) system message, inserting one if absent.

    Appending at the *end* of the system prompt works for every template
    without knowledge of its structure, and keeps the prompt prefix stable
    for KV-cache reuse when the hint does not change between turns.
    """
    messages = [dict(message) for message in request["messages"]]
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") in ("system", "developer"):
            content = messages[index].get("content")
            if isinstance(content, str) or content is None:
                messages[index]["content"] = (content or "") + hint
            elif isinstance(content, list):
                parts = list(content)
                parts.append({"type": "text", "text": hint})
                messages[index]["content"] = parts
            new_request = dict(request)
            new_request["messages"] = messages
            return new_request
    # No system message: insert one at the front (llama.cpp appends a system
    # message only at position 0, so this is also the conservative choice).
    messages.insert(0, {"role": "system", "content": hint.lstrip("\n")})
    new_request = dict(request)
    new_request["messages"] = messages
    return new_request


def _strip_generation_prompt(text: str, render: RenderResult) -> str:
    """Remove the generation prompt prefix if the server echoed it back."""
    prefix = render.generation_prompt
    if prefix and text.startswith(prefix):
        return text[len(prefix):]
    return text
