"""Command line interface for toolcaller.

Examples
--------
Render only (inspect the prompt)::

    python -m toolcaller --template chat_template.jinja --chat request.json --render-only

Render + call the server + parse tool calls::

    python -m toolcaller --template chat_template.jinja \\
        --url http://localhost:8082 --chat request.json

Parse already generated text (no server needed)::

    python -m toolcaller --template chat_template.jinja --parse-file out.txt --profile qwen

Print an OpenAI ``chat.completion`` object (ready for any client)::

    python -m toolcaller --template t.jinja --chat req.json --response-format openai

Inspect what the wrapper detected about a template::

    python -m toolcaller --template chat_template.jinja --inspect-template
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from .api import ChatPipeline
from .client import CompletionClient, format_timings
from .errors import ConfigError, ServerError, ToolcallerError
from .hints import HINT_MODES
from .messages import load_chat_request, normalize_request
from .parser import REASONING_FORMATS, SyntaxParser, resolve_profile, ParserSelection
from .profiles import (
    available_profiles,
    detect_profile_from_template,
    dump_profiles,
    get_profile,
    load_profile_file,
    write_example_profile,
)

PROG = "toolcaller"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Render OpenAI chat-completion JSON with a Jinja template, send it to a "
            "raw llama.cpp-style /completion endpoint and parse model-specific tool calls."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s --template chat_template.jinja --chat request.json\n"
            "  %(prog)s --template chat_template.jinja --chat request.json --render-only\n"
            "  %(prog)s --template chat_template.jinja --parse-file answer.txt --profile qwen\n"
            "  %(prog)s --inspect-template --template chat_template.jinja\n"
            "  %(prog)s --list-profiles\n"
        ),
    )

    # --- input ---------------------------------------------------------
    inp = parser.add_argument_group("input")
    inp.add_argument("--template", "-t", help="Jinja chat template file")
    inp.add_argument(
        "--chat",
        "-c",
        help="OpenAI chat-completion request JSON file ('-' for stdin)",
    )
    inp.add_argument(
        "--messages",
        help="Inline JSON array of messages (alternative to --chat)",
    )
    inp.add_argument(
        "--tools",
        help="Inline JSON array of tools (overrides the tools in --chat)",
    )
    inp.add_argument(
        "--system",
        help="Inline system prompt (added if no system message is present)",
    )
    inp.add_argument(
        "--parse-file",
        help="Parse this raw model output file instead of calling the server",
    )
    inp.add_argument(
        "--stdin",
        action="store_true",
        help="Read the chat request from stdin (same as --chat -)",
    )

    # --- server --------------------------------------------------------
    srv = parser.add_argument_group("server")
    srv.add_argument(
        "--serve",
        action="store_true",
        help="Run as an HTTP daemon (OpenAI-compatible /v1/chat/completions)",
    )
    srv.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("TOOLCALLER_PORT", "8080")),
        help="Daemon listen port (default: %(default)s, env TOOLCALLER_PORT)",
    )
    srv.add_argument(
        "--host",
        default=os.environ.get("TOOLCALLER_HOST", "0.0.0.0"),
        help="Daemon bind address (default: %(default)s, env TOOLCALLER_HOST)",
    )
    srv.add_argument("--url", dest="url",
        default=os.environ.get("TOOLCALLER_URL", "http://localhost:8082"),
        help="Upstream /completion server URL (default: %(default)s, env TOOLCALLER_URL)",
    )
    srv.add_argument("--backend-url", dest="backend_url",
        help="Alias for --url (explicit upstream backend)",
    )
    srv.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("TOOLCALLER_TIMEOUT", "3600")),
        help="HTTP timeout in seconds (default: %(default)s)",
    )
    srv.add_argument("--api-key", help="Bearer token for the daemon (client auth)")
    srv.add_argument("--backend-api-key", help="Bearer token for the upstream backend")
    srv.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Stream from the server and print chunks as they arrive (default)",
    )
    srv.add_argument(
        "--no-stream",
        action="store_true",
        help=(
            "Disable streaming: fetch/wait for the full response and return one "
            "JSON body. With --serve, the daemon ignores the client's "
            "'stream' field and always answers with a single chat.completion."
        ),
    )
    srv.add_argument(
        "--max-tokens",
        "-n",
        type=int,
        help="Override max tokens (default: request max_tokens, else unlimited)",
    )
    srv.add_argument(
        "--cors",
        action="store_true",
        help="Enable CORS (allow all origins) on the daemon",
    )
    srv.add_argument(
        "--cors-origins",
        help="Comma-separated list of allowed CORS origins (overrides --cors)",
    )

    # --- template ------------------------------------------------------
    tpl = parser.add_argument_group("template")
    tpl.add_argument(
        "--template-kwarg",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Extra template variable, JSON-decoded (repeatable), e.g. enable_thinking=false",
    )
    tpl.add_argument("--bos-token", help="Template bos_token (default: from /props)")
    tpl.add_argument("--eos-token", help="Template eos_token (default: from /props)")
    tpl.add_argument(
        "--add-bos",
        action="store_true",
        help="Prepend bos_token when the template does not emit it",
    )
    tpl.add_argument(
        "--strip-generation-thinking",
        action="store_true",
        help=(
            "Remove the trailing reasoning-start marker (e.g. Qwen's '<think>') from "
            "the generation prompt so the model emits it itself and reasoning can be "
            "split from content"
        ),
    )
    tpl.add_argument(
        "--no-strict-template",
        action="store_true",
        help="Do not fail when the request uses features the template ignores",
    )

    # --- parsing -------------------------------------------------------
    prs = parser.add_argument_group("parsing")
    prs.add_argument(
        "--profile",
        help="Tool-call syntax profile (" + ", ".join(available_profiles()) + ")",
    )
    prs.add_argument("--profile-file", help="Load a custom profile from JSON")
    prs.add_argument(
        "--no-auto-profile",
        action="store_true",
        help="Never guess the profile from the template",
    )
    prs.add_argument(
        "--reasoning-format",
        default="auto",
        choices=list(REASONING_FORMATS),
        help="How to split reasoning out of the output (default: %(default)s)",
    )
    prs.add_argument(
        "--stop",
        action="append",
        default=[],
        help="Extra stop string (repeatable)",
    )
    prs.add_argument(
        "--stop-at-call-end",
        action="store_true",
        help="Also stop at </tool_call> (faster, but breaks parallel calls)",
    )
    prs.add_argument(
        "--hint-mode",
        default="auto",
        choices=list(HINT_MODES),
        help="Prompt-side format reminder: auto/always/never (default: %(default)s)",
    )
    prs.add_argument(
        "--grammar-style",
        default="none",
        choices=["none", "gbnf"],
        help="Send a lazy GBNF grammar to constrain tool-call syntax (default: %(default)s)",
    )

    # --- output --------------------------------------------------------
    out = parser.add_argument_group("output")
    out.add_argument(
        "--response-format",
        default="openai",
        choices=["openai", "simple", "raw", "both"],
        help=(
            "openai = chat.completion object; simple = message + finish_reason; "
            "raw = raw text; both = openai plus raw/prompt (default: %(default)s)"
        ),
    )
    out.add_argument("--render-only", action="store_true", help="Only print the prompt")
    out.add_argument("--show-prompt", action="store_true", help="Print the prompt to stderr")
    out.add_argument("--show-reasoning", action="store_true", help="Print reasoning to stderr")
    out.add_argument("--no-timings", action="store_true", help="Omit timings from the response")
    out.add_argument("--output", "-o", help="Write the result to this file instead of stdout")
    out.add_argument("--indent", type=int, default=2, help="JSON indent (default: %(default)s)")
    out.add_argument("--quiet", "-q", action="store_true", help="Suppress stderr diagnostics")
    out.add_argument("--debug", action="store_true", help="Verbose diagnostics on stderr")

    # --- utilities -----------------------------------------------------
    util = parser.add_argument_group("utilities")
    util.add_argument("--list-profiles", action="store_true", help="List built-in profiles")
    util.add_argument(
        "--dump-profile",
        metavar="NAME",
        help="Print a profile as JSON",
    )
    util.add_argument(
        "--write-example-profile",
        metavar="PATH",
        help="Write an example custom profile to PATH",
    )
    util.add_argument(
        "--inspect-template",
        action="store_true",
        help="Print detected template capabilities and a sample render",
    )
    util.add_argument(
        "--check-server",
        action="store_true",
        help="Query /health and /props and print what the server provides",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_template_kwargs(pairs: List[str]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"--template-kwarg expects KEY=VALUE, got {pair!r}")
        key, raw = pair.split("=", 1)
        key = key.strip()
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        kwargs[key] = value
    return kwargs


def load_request(args: argparse.Namespace) -> Dict[str, Any]:
    request: Optional[Dict[str, Any]] = None
    if args.chat and args.chat != "-":
        request = load_chat_request(args.chat)
    elif (args.chat == "-") or args.stdin:
        try:
            data = json.load(sys.stdin)
        except ValueError as exc:
            raise ConfigError(f"stdin: invalid JSON: {exc}") from exc
        request = normalize_request(data)
    elif args.messages:
        try:
            messages = json.loads(args.messages)
        except ValueError as exc:
            raise ConfigError(f"--messages: invalid JSON: {exc}") from exc
        request = normalize_request({"messages": messages})
    elif args.parse_file or args.inspect_template:
        return {}

    if request is None:
        raise ConfigError("no input: pass --chat FILE, --messages JSON or --stdin")

    if args.tools:
        try:
            request["tools"] = json.loads(args.tools)
        except ValueError as exc:
            raise ConfigError(f"--tools: invalid JSON: {exc}") from exc

    if args.system:
        messages = list(request.get("messages") or [])
        if not any(m.get("role") in ("system", "developer") for m in messages):
            messages.insert(0, {"role": "system", "content": args.system})
            request["messages"] = messages

    if args.max_tokens is not None:
        request["max_tokens"] = args.max_tokens
    return request


def emit(text: str, args: argparse.Namespace) -> None:
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
            if not text.endswith("\n"):
                fh.write("\n")
        if not args.quiet:
            print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")


def dumps(obj: Any, args: argparse.Namespace) -> str:
    return json.dumps(obj, indent=args.indent if args.indent > 0 else None, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list_profiles() -> str:
    return dump_profiles()


def cmd_check_server(args: argparse.Namespace) -> str:
    client = CompletionClient(args.url, timeout=min(args.timeout, 30.0), api_key=args.api_key)
    report: Dict[str, Any] = {"url": args.url}
    try:
        report["health"] = client.health()
    except ServerError as exc:
        report["health"] = {"error": str(exc)}
    try:
        props = client.props()
        template = props.get("chat_template") or ""
        report["props"] = {
            "model": props.get("model_name") or props.get("model_path"),
            "bos_token": props.get("bos_token"),
            "eos_token": props.get("eos_token"),
            "n_ctx": props.get("n_ctx"),
            "chat_template_length": len(template),
            "chat_template_hint": detect_profile_from_template(template),
            "modalities": props.get("modalities"),
        }
    except ServerError as exc:
        report["props"] = {"error": str(exc)}
    if args.template:
        report["tokenize"] = client.tokenize("hello world") is not None
    return dumps(report, args)


def cmd_inspect_template(args: argparse.Namespace) -> str:
    if not args.template:
        raise ConfigError("--inspect-template requires --template")
    from .environment import load_template_file
    from .caps import analyze_caps

    env = load_template_file(args.template)
    caps = analyze_caps(env)
    profile_name = detect_profile_from_template(env.source)
    report: Dict[str, Any] = {
        "template": args.template,
        "size_bytes": len(env.source.encode("utf-8")),
        "detected_profile": profile_name or "(none - use --profile)",
        "used_globals": sorted(env.used_globals()),
        "caps": caps.to_dict(),
    }
    try:
        pipeline = ChatPipeline(
            template_path=args.template,
            base_url=args.url,
            profile=args.profile,
            profile_file=args.profile_file,
            auto_profile=not args.no_auto_profile,
            template_kwargs=parse_template_kwargs(args.template_kwarg),
            bos_token=args.bos_token,
            eos_token=args.eos_token,
            discover_tokens=False,
            hint_mode=args.hint_mode,
        )
        sample = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the weather in Budapest?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        }
        render = pipeline.render(sample)
        report["selected_profile"] = pipeline.profile.name
        report["selected_profile_reason"] = pipeline.profile_hint
        report["sample_prompt"] = render.prompt
        report["sample_generation_prompt"] = render.generation_prompt
    except ToolcallerError as exc:
        report["render_error"] = str(exc)
    return dumps(report, args)


def cmd_parse_file(args: argparse.Namespace) -> str:
    try:
        with open(args.parse_file, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise ConfigError(f"cannot read {args.parse_file!r}: {exc}") from exc

    template_source = None
    eos_token = args.eos_token or ""
    if args.template:
        from .environment import load_template_file

        env = load_template_file(args.template)
        template_source = env.source
    elif args.url:
        try:
            client = CompletionClient(args.url, timeout=min(args.timeout, 30.0))
            props = client.props()
            template_source = props.get("chat_template")
            eos_token = eos_token or str(props.get("eos_token") or "")
        except ServerError:
            pass

    selection = ParserSelection(
        profile=args.profile,
        profile_file=args.profile_file,
        auto_profile=not args.no_auto_profile,
    )
    profile, reason = resolve_profile(selection, template_source)
    parser = SyntaxParser(profile, eos_token=eos_token)
    parsed = parser.parse(text, reasoning_format=args.reasoning_format)

    result = {
        "message": parsed.to_openai_message(),
        "finish_reason": parsed.finish_reason,
        "profile": profile.name,
        "profile_reason": reason,
    }
    if parsed.warnings:
        result["warnings"] = parsed.warnings
    if args.response_format == "raw":
        return text
    return dumps(result, args)


def cmd_complete(args: argparse.Namespace, request: Dict[str, Any]) -> str:
    pipeline = ChatPipeline(
        template_path=args.template,
        base_url=args.url,
        profile=args.profile,
        profile_file=args.profile_file,
        auto_profile=not args.no_auto_profile,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        add_bos=args.add_bos,
        hint_mode=args.hint_mode,
        grammar_style=args.grammar_style,
        reasoning_format=args.reasoning_format,
        template_kwargs=parse_template_kwargs(args.template_kwarg),
        strict_template=not args.no_strict_template,
        extra_stops=args.stop,
        stop_at_call_end=args.stop_at_call_end,
        strip_generation_thinking=args.strip_generation_thinking,
        timeout=args.timeout,
        debug=args.debug,
    )
    if not args.template:
        raise ConfigError("--template is required")

    if args.render_only:
        render = pipeline.render(request)
        output = {
            "prompt": render.prompt,
            "generation_prompt": render.generation_prompt,
            "profile": pipeline.profile.name,
            "profile_reason": pipeline.profile_hint,
            "caps": render.caps.to_dict(),
            "warnings": render.warnings,
        }
        if args.response_format == "raw":
            return render.prompt
        return dumps(output, args)

    if args.show_prompt:
        render = pipeline.render(request)
        print("----- prompt -----", file=sys.stderr)
        print(render.prompt, file=sys.stderr)
        print("----- end prompt -----", file=sys.stderr)

    do_stream = args.stream and not args.no_stream

    def on_chunk(chunk: str) -> None:
        if do_stream and not args.quiet:
            sys.stderr.write(chunk)
            sys.stderr.flush()

    result = pipeline.complete(
        request,
        on_chunk=on_chunk if do_stream else None,
        stream=do_stream,
        strict_parse=False,
    )
    if do_stream and not args.quiet:
        sys.stderr.write("\n")

    if args.debug or (result.warnings and not args.quiet):
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
    if args.show_reasoning and result.parsed.reasoning_content:
        print("----- reasoning -----", file=sys.stderr)
        print(result.parsed.reasoning_content, file=sys.stderr)
        print("----- end reasoning -----", file=sys.stderr)
    if not args.quiet and result.completion.timings:
        summary = format_timings(result.completion.timings)
        if summary:
            print(summary, file=sys.stderr)

    if args.response_format == "raw":
        return result.completion.text
    if args.response_format == "simple":
        payload = {
            "message": result.message,
            "finish_reason": result.finish_reason,
            "profile": result.profile.name,
        }
        if result.warnings:
            payload["warnings"] = result.warnings
        return dumps(payload, args)

    response = result.to_response(include_timings=not args.no_timings)
    if args.response_format == "both":
        response["raw_text"] = result.completion.text
        response["prompt"] = result.render.prompt
        response["profile"] = result.profile.name
        if result.warnings:
            response["warnings"] = result.warnings
    return dumps(response, args)


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the HTTP daemon."""
    from .server import ServerConfig, run_server

    # Resolve backend URL: --backend-url wins, else --url
    backend_url = args.backend_url or args.url

    # Build template kwargs
    template_kwargs = parse_template_kwargs(args.template_kwarg)

    # CORS
    cors_origins = None
    if args.cors_origins:
        cors_origins = [o.strip() for o in args.cors_origins.split(",") if o.strip()]
    elif args.cors:
        cors_origins = ["*"]

    config = ServerConfig(
        host=args.host,
        port=args.port,
        template_path=args.template,
        template_source=None,
        backend_url=backend_url,
        backend_timeout=args.timeout,
        api_key=args.api_key,
        backend_api_key=args.backend_api_key,
        profile=args.profile,
        profile_file=args.profile_file,
        auto_profile=not args.no_auto_profile,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        add_bos=args.add_bos,
        hint_mode=args.hint_mode,
        grammar_style=args.grammar_style,
        reasoning_format=args.reasoning_format,
        template_kwargs=template_kwargs if template_kwargs else None,
        strict_template=not args.no_strict_template,
        extra_stops=args.stop,
        stop_at_call_end=args.stop_at_call_end,
        strip_generation_thinking=args.strip_generation_thinking,
        cors_origins=cors_origins,
        no_stream=args.no_stream,
        debug=args.debug,
    )

    # Logging
    level = logging.DEBUG if args.debug else (logging.INFO if not args.quiet else logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        run_server(config)
    except KeyboardInterrupt:
        pass

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.list_profiles:
            emit(cmd_list_profiles(), args)
            return 0
        if args.dump_profile:
            emit(json.dumps(get_profile(args.dump_profile).to_dict(), indent=2, ensure_ascii=False), args)
            return 0
        if args.write_example_profile:
            write_example_profile(args.write_example_profile)
            if not args.quiet:
                print(f"wrote {args.write_example_profile}", file=sys.stderr)
            return 0
        if args.check_server:
            emit(cmd_check_server(args), args)
            return 0
        if args.inspect_template:
            emit(cmd_inspect_template(args), args)
            return 0
        if args.parse_file:
            emit(cmd_parse_file(args), args)
            return 0

        if args.serve:
            return cmd_serve(args)

        request = load_request(args)
        if not args.template:
            raise ConfigError("--template is required (see --list-profiles / --inspect-template)")
        emit(cmd_complete(args, request), args)
        return 0

    except ToolcallerError as exc:
        if not args.quiet:
            print(f"{PROG}: error: {exc}", file=sys.stderr)
        if args.debug:
            raise
        return 2 if exc.status == 400 else 1
    except KeyboardInterrupt:
        print(f"\n{PROG}: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
