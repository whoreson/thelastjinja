"""Lightweight HTTP daemon serving OpenAI-compatible chat completions.

Binds a port, accepts ``POST /v1/chat/completions`` (and health/props
endpoints), and dispatches each request through a pre-configured
``ChatPipeline`` to a llama.cpp ``/completion`` backend.

Usage::

    python -m toolcaller --serve \
        --template chat_template.jinja \
        --url http://localhost:8082 \
        --port 8080

No extra dependencies beyond what is already required (``jinja2``,
``requests``).  The stdlib ``http.server`` + ``threading`` is sufficient
because the heavy work (LLM inference) is done by the upstream server.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .api import ChatPipeline, ChatResult
from .client import CompletionClient
from .errors import ConfigError, ServerError, ToolcallerError
from .hints import HINT_MODES
from .parser import REASONING_FORMATS

logger = logging.getLogger("toolcaller.server")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    """Daemon configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    template_path: Optional[str] = None
    template_source: Optional[str] = None
    backend_url: str = "http://localhost:8082"
    backend_timeout: float = 3600.0
    api_key: Optional[str] = None
    backend_api_key: Optional[str] = None
    profile: Optional[str] = None
    profile_file: Optional[str] = None
    auto_profile: bool = True
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None
    add_bos: bool = False
    hint_mode: str = "auto"
    grammar_style: str = "none"
    reasoning_format: str = "auto"
    template_kwargs: Optional[Dict[str, Any]] = None
    strict_template: bool = True
    extra_stops: List[str] = field(default_factory=list)
    stop_at_call_end: bool = False
    strip_generation_thinking: bool = False
    max_request_body: int = 16 * 1024 * 1024  # 16 MB
    cors_origins: Optional[List[str]] = None
    # Force non-streaming: ignore the client's "stream" field and always
    # return a single JSON chat.completion response.
    no_stream: bool = False
    debug: bool = False


# ---------------------------------------------------------------------------
# Threading HTTP server
# ---------------------------------------------------------------------------

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a new thread.

    ``daemon_threads = True`` ensures the server can shut down even if
    upstream LLM calls are still in flight.
    """

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size: int = 128


# ---------------------------------------------------------------------------
# Request routing helpers
# ---------------------------------------------------------------------------

def stream_requested(request: Dict[str, Any], config: ServerConfig) -> bool:
    """Decide whether to answer with SSE.

    The client asks for streaming via ``"stream": true``, but the daemon can
    force non-streaming with ``--no-stream`` (``config.no_stream``).  In that
    case every request is answered with a single ``chat.completion`` JSON body
    regardless of what the client asked for.
    """
    if config.no_stream:
        return False
    return bool(request.get("stream", False))


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class ChatHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible HTTP handler."""

    protocol_version = "HTTP/1.1"

    # Silence per-request log lines; the pipeline already logs what matters.
    def log_message(self, format, *args):
        logger.debug("HTTP: " + format, *args)

    # -- routing ----------------------------------------------------------

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/health" or path == "/v1/health":
            self._json_response(200, {"status": "ok", "upstream": self._upstream_ok()})
        elif path == "/props" or path == "/v1/props":
            self._json_response(200, self.server.config.pipeline.describe())
        elif path == "/v1/models":
            self._list_models()
        elif path == "/" or path == "/v1":
            self._json_response(200, {
                "message": "toolcaller daemon",
                "endpoints": [
                    "GET  /health",
                    "GET  /props",
                    "GET  /v1/models",
                    "POST /v1/chat/completions",
                ],
            })
        else:
            self._json_response(404, {"error": f"not found: {path}"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/v1/chat/completions" or path == "/chat/completions":
            self._handle_chat()
        else:
            self._json_response(404, {"error": f"not found: {path}"})

    # -- chat completion --------------------------------------------------

    def _handle_chat(self):
        pipeline = self.server.config.pipeline

        # Auth check
        if self.server.config.api_key:
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != self.server.config.api_key:
                self._json_response(401, {"error": "invalid api key"})
                return

        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > self.server.config.max_request_body:
            self._json_response(413, {"error": "request body too large"})
            return

        try:
            body = self.rfile.read(content_length)
            request = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            self._json_response(400, {"error": f"invalid JSON: {exc}"})
            return

        stream = stream_requested(request, self.server.config)

        if stream:
            self._stream_response(request, pipeline)
        else:
            self._nonstream_response(request, pipeline)

    def _nonstream_response(self, request: Dict, pipeline: ChatPipeline):
        cancel = threading.Event()
        try:
            start = time.time()
            result = pipeline.complete(request, strict_parse=False, cancel=cancel)
            elapsed = time.time() - start

            if result.completion.cancelled:
                return  # client already gone, nothing to write

            response = result.to_response(include_timings=True)
            response["elapsed"] = round(elapsed, 3)

            self._json_response(200, response)

        except BrokenPipeError:
            cancel.set()  # abort the backend
        except ConfigError as exc:
            self._json_response(400, {"error": str(exc)})
        except ServerError as exc:
            self._json_response(502, {"error": f"upstream error: {exc}"})
        except Exception as exc:
            logger.exception("unhandled error in chat completion")
            self._json_response(500, {"error": f"internal error: {exc}"})

    def _stream_response(self, request: Dict, pipeline: ChatPipeline):
        """Stream response as SSE (server-sent events).

        The backend /completion stream is consumed in a worker thread.
        Chunks are pushed into a queue that the main handler drains and
        writes to the client — so the client sees tokens as they arrive,
        not after the model finishes.
        """

        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        cancel = threading.Event()
        q: "queue.Queue" = queue.Queue(maxsize=256)
        done = threading.Event()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self._apply_cors_headers()
        self.end_headers()

        def worker():
            """Consume the backend stream and enqueue parsed chunks."""
            try:
                result = pipeline.complete(
                    request,
                    stream=True,
                    strict_parse=False,
                    cancel=cancel,
                    on_chunk=lambda chunk: q.put(chunk),
                )
                q.put(result)  # sentinel: the ChatResult
            except Exception as exc:
                q.put(exc)  # sentinel: the error
            finally:
                done.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # Drain the queue and write SSE to the client.
        try:
            self._drain_sse_queue(q, done, cancel, request_id, created)
        except BrokenPipeError:
            cancel.set()
        except ConnectionResetError:
            cancel.set()
        except Exception as exc:
            cancel.set()
            logger.exception("unhandled error in streaming chat completion")

    # -- helpers ----------------------------------------------------------

    def _drain_sse_queue(
        self,
        q: "queue.Queue",
        done: threading.Event,
        cancel: threading.Event,
        request_id: str,
        created: int,
    ):
        """Drain items from the worker's queue and write SSE to the client.

        Raw text chunks (str) are forwarded immediately as content deltas
        so the client sees tokens as they arrive.  When the worker finishes,
        the ChatResult sentinel is sent; if it contains tool calls they are
        appended as additional chunks.
        """
        model_name: Optional[str] = "unknown"
        content_sent: List[str] = []  # track what we already streamed

        def sse_write(data: Any) -> None:
            line = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()

        # Initial role chunk
        sse_write({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}],
        })

        while True:
            if cancel.is_set():
                return
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                if done.is_set():
                    continue
                continue

            # Sentinel: Exception from the worker.
            if isinstance(item, Exception):
                sse_write({"error": str(item)})
                sse_write("[DONE]")
                return

            # Sentinel: ChatResult (worker finished).
            from .api import ChatResult
            if isinstance(item, ChatResult):
                result = item
                model_name = result.model or "unknown"
                if result.completion.cancelled:
                    sse_write("[DONE]")
                    return
                # Send any tool calls that weren't in the raw text stream.
                for call in result.parsed.tool_calls:
                    sse_write({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"tool_calls": [{
                            "index": 0,
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
                        }]}}],
                    })
                # Finish.
                sse_write({
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": result.finish_reason}],
                })
                sse_write("[DONE]")
                return

            # Regular text chunk from on_chunk callback.
            if isinstance(item, str):
                content_sent.append(item)
                sse_write({
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": item}}],
                })

    def _upstream_ok(self) -> str:
        try:
            client = CompletionClient(
                self.server.config.backend_url,
                timeout=10,
                api_key=self.server.config.backend_api_key,
            )
            health = client.health()
            return health.get("status", "ok")
        except Exception as exc:
            return f"error: {exc}"

    def _list_models(self):
        try:
            client = CompletionClient(
                self.server.config.backend_url,
                timeout=10,
                api_key=self.server.config.backend_api_key,
            )
            props = client.props()
            model_name = props.get("model_name") or props.get("model_path", "unknown")
        except Exception:
            model_name = "unknown"

        models = {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "toolcaller",
                }
            ],
        }
        self._json_response(200, models)

    def _apply_cors(self):
        self._apply_cors_headers()

    def _apply_cors_headers(self):
        origins = self.server.config.cors_origins
        if origins is None or "*" in origins:
            self.send_header("Access-Control-Allow-Origin", "*")
        else:
            origin = self.headers.get("Origin", "")
            if origin in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._apply_cors_headers()
        self.end_headers()

    def _json_response(self, status: int, data: Any):
        body = json.dumps(data, ensure_ascii=False, indent=None).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._apply_cors_headers()
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def create_server(config: ServerConfig) -> ThreadedHTTPServer:
    """Build a running server from a config."""

    # Build the pipeline
    from .hints import HINT_MODES as _HM
    from .parser import REASONING_FORMATS as _RF

    if config.hint_mode not in _HM:
        raise ConfigError(f"hint_mode must be one of {_HM}")
    if config.reasoning_format not in _RF:
        raise ConfigError(f"reasoning_format must be one of {_RF}")

    pipeline = ChatPipeline(
        template_path=config.template_path,
        template_source=config.template_source,
        base_url=config.backend_url,
        profile=config.profile,
        profile_file=config.profile_file,
        auto_profile=config.auto_profile,
        bos_token=config.bos_token,
        eos_token=config.eos_token,
        add_bos=config.add_bos,
        hint_mode=config.hint_mode,
        grammar_style=config.grammar_style,
        reasoning_format=config.reasoning_format,
        template_kwargs=config.template_kwargs or {},
        strict_template=config.strict_template,
        extra_stops=config.extra_stops,
        stop_at_call_end=config.stop_at_call_end,
        strip_generation_thinking=config.strip_generation_thinking,
        timeout=config.backend_timeout,
        debug=config.debug,
        client=CompletionClient(
            config.backend_url,
            timeout=config.backend_timeout,
            api_key=config.backend_api_key,
        ),
    )

    config.pipeline = pipeline  # type: ignore[attr-defined]

    server = ThreadedHTTPServer((config.host, config.port), ChatHandler)
    server.config = config

    return server


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_server(config: ServerConfig) -> None:
    """Start the daemon and block until interrupted."""

    server = create_server(config)

    desc = config.pipeline.describe()
    logger.info(
        "toolcaller daemon starting on %s:%d -> %s (profile: %s)",
        config.host,
        config.port,
        config.backend_url,
        desc.get("profile", "?"),
    )
    logger.info("endpoints: /v1/chat/completions, /health, /props, /v1/models")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()

