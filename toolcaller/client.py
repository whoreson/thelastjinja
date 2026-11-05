"""HTTP client for llama.cpp / ik_llama.cpp style ``/completion`` servers.

Two things make this more than a thin ``requests.post`` wrapper:

**Exact stopping.**  A chat prompt is a sequence of turns, e.g.
``<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n<think>\\n``.
The model may legitimately emit ``<|im_end|>`` (end of turn) - or, for
thinking models, ``</think>`` mid-answer.  Naive substring stopping either
truncates the answer or lets the model run into the next turn.  We therefore
use the server's ``/tokenize`` endpoint to turn every stop sequence into an
exact token list and stop on a token *boundary*.

**Raw text out.**  ``content`` in the ``/completion`` response already has
the stop word and the EOS token removed
(``common_sampler_print``/``server_slot::send_final_response``), so the raw
output has to be reconstructed: ``text + stopped_word + eos_token?``.

Streaming (``stream: true``) via server-sent events is supported as well and
handles both plain ``"content"`` chunks and llama.cpp's
``{"content": "..."}`` SSE format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import requests

from .errors import ServerError

#: Defaults for a llama.cpp-compatible server.
DEFAULT_TIMEOUT = 3600.0


@dataclass
class CompletionResult:
    """Raw model output plus server metadata."""

    text: str
    """Reconstructed raw text (stop word and EOS re-attached)."""

    content: str
    """``content`` exactly as returned by the server (stop/EOS stripped)."""

    stopped_eos: bool = False
    stopped_word: bool = False
    stopped_limit: bool = False
    stop_word: str = ""
    tokens_predicted: int = 0
    tokens_evaluated: int = 0
    model: str = ""
    timings: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def finish_reason(self) -> str:
        if self.stopped_eos or self.stopped_word:
            return "stop"
        return "length"


class CompletionClient:
    """Talks to ``POST /completion`` (+ ``/tokenize``, ``/props``, ``/health``)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8082",
        *,
        timeout: float = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
        api_key: Optional[str] = None,
        completion_path: str = "/completion",
        tokenize_path: str = "/tokenize",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.completion_url = self.base_url + completion_path
        self.tokenize_url = self.base_url + tokenize_path
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        self._vocab_cache: Dict[str, Optional[List[int]]] = {}
        self._tokenize_supported: Optional[bool] = None

    # -- discovery ---------------------------------------------------------
    def props(self) -> Dict[str, Any]:
        return self._get_json("/props")

    def health(self) -> Dict[str, Any]:
        return self._get_json("/health")

    def _get_json(self, path: str) -> Dict[str, Any]:
        try:
            response = self.session.get(self.base_url + path, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ServerError(f"GET {path} failed: {exc}") from exc
        if response.status_code >= 400:
            raise ServerError(f"GET {path} -> HTTP {response.status_code}: {response.text[:500]}")
        try:
            return response.json()
        except ValueError as exc:
            raise ServerError(f"GET {path}: invalid JSON: {exc}") from exc

    # -- tokenisation ------------------------------------------------------
    def tokenize(self, text: str, *, add_special: bool = False) -> Optional[List[int]]:
        """Token ids of ``text`` or ``None`` if the server cannot tokenise."""
        if self._tokenize_supported is False:
            return None
        try:
            response = self.session.post(
                self.tokenize_url,
                json={"content": text, "add_special": add_special, "with_pieces": False},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            self._tokenize_supported = False
            raise ServerError(f"POST /tokenize failed: {exc}") from exc
        if response.status_code == 404:
            self._tokenize_supported = False
            return None
        if response.status_code >= 400:
            self._tokenize_supported = False
            return None
        self._tokenize_supported = True
        try:
            payload = response.json()
        except ValueError:
            self._tokenize_supported = False
            return None
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            return None
        ids: List[int] = []
        for token in tokens:
            if isinstance(token, int):
                ids.append(token)
            elif isinstance(token, dict) and isinstance(token.get("id"), int):
                ids.append(token["id"])
            else:
                return None
        return ids

    def stop_token_sequences(self, stops: Sequence[str]) -> List[List[int]]:
        """Tokenise each stop string; unusable ones are dropped."""
        sequences: List[List[int]] = []
        for stop in stops:
            if not stop:
                continue
            cached = self._vocab_cache.get(stop, "miss")  # type: ignore[arg-type]
            if cached == "miss":
                cached = self.tokenize(stop)
                self._vocab_cache[stop] = cached
            if cached:
                sequences.append(list(cached))
        return sequences

    # -- completion (non-streaming) ---------------------------------------
    def complete(
        self,
        prompt: str,
        *,
        stops: Sequence[str] = (),
        n_predict: int = -1,
        grammar: Optional[str] = None,
        grammar_triggers: Optional[Sequence[Dict[str, Any]]] = None,
        grammar_lazy: bool = False,
        preserved_tokens: Optional[Sequence[str]] = None,
        sampling: Optional[Dict[str, Any]] = None,
        cache_prompt: bool = True,
        id_slot: int = -1,
        extra: Optional[Dict[str, Any]] = None,
    ) -> CompletionResult:
        body = self._build_body(
            prompt,
            stops=stops,
            n_predict=n_predict,
            grammar=grammar,
            grammar_triggers=grammar_triggers,
            grammar_lazy=grammar_lazy,
            preserved_tokens=preserved_tokens,
            sampling=sampling,
            cache_prompt=cache_prompt,
            id_slot=id_slot,
            stream=False,
            extra=extra,
        )
        payload = self._post_json(self.completion_url, body)
        return self._parse_response(payload)

    # -- completion (streaming) -------------------------------------------
    def complete_stream(
        self,
        prompt: str,
        *,
        on_chunk: Optional[Callable[[str], None]] = None,
        stops: Sequence[str] = (),
        n_predict: int = -1,
        grammar: Optional[str] = None,
        grammar_triggers: Optional[Sequence[Dict[str, Any]]] = None,
        grammar_lazy: bool = False,
        preserved_tokens: Optional[Sequence[str]] = None,
        sampling: Optional[Dict[str, Any]] = None,
        cache_prompt: bool = True,
        id_slot: int = -1,
        extra: Optional[Dict[str, Any]] = None,
    ) -> CompletionResult:
        body = self._build_body(
            prompt,
            stops=stops,
            n_predict=n_predict,
            grammar=grammar,
            grammar_triggers=grammar_triggers,
            grammar_lazy=grammar_lazy,
            preserved_tokens=preserved_tokens,
            sampling=sampling,
            cache_prompt=cache_prompt,
            id_slot=id_slot,
            stream=True,
            extra=extra,
        )
        pieces: List[str] = []
        final: Dict[str, Any] = {}
        streamed_text = ""
        try:
            response = self.session.post(
                self.completion_url,
                json=body,
                stream=True,
                timeout=self.timeout,
                headers={"Accept": "text/event-stream"},
            )
        except requests.RequestException as exc:
            raise ServerError(f"POST /completion failed: {exc}") from exc
        if response.status_code >= 400:
            raise ServerError(self._http_error_message(response))
        try:
            for event in self._iter_sse(response):
                if not isinstance(event, dict):
                    continue
                if "error" in event:
                    raise ServerError(str(event["error"]))
                piece = event.get("content")
                if isinstance(piece, str) and piece:
                    pieces.append(piece)
                    if on_chunk is not None:
                        on_chunk(piece)
                if event.get("stop"):
                    final = event
        except requests.RequestException as exc:
            raise ServerError(f"stream interrupted: {exc}") from exc
        finally:
            response.close()

        streamed_text = "".join(pieces)
        # The final event carries the authoritative `content` (stop word and
        # EOS already stripped).  Fall back to the concatenated chunks when the
        # server omits it (or sends an empty one).
        if not isinstance(final.get("content"), str) or not final.get("content"):
            final["content"] = streamed_text
        return self._parse_response(final, streamed_text=streamed_text)

    # -- helpers -----------------------------------------------------------
    def _build_body(
        self,
        prompt: str,
        *,
        stops: Sequence[str],
        n_predict: int,
        grammar: Optional[str],
        grammar_triggers: Optional[Sequence[Dict[str, Any]]],
        grammar_lazy: bool,
        preserved_tokens: Optional[Sequence[str]],
        sampling: Optional[Dict[str, Any]],
        cache_prompt: bool,
        id_slot: int,
        stream: bool,
        extra: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "prompt": prompt,
            "n_predict": n_predict,
            "stream": stream,
            "cache_prompt": cache_prompt,
        }
        stop_list = [s for s in stops if s]
        if stop_list:
            body["stop"] = stop_list
        if id_slot >= 0:
            body["id_slot"] = id_slot
        if grammar:
            body["grammar"] = grammar
            if grammar_lazy:
                # llama.cpp calls this `grammar_lazy` on the CLI only; the
                # server infers laziness from the presence of triggers.
                pass
            if grammar_triggers:
                body["grammar_triggers"] = list(grammar_triggers)
        if preserved_tokens:
            body["preserved_tokens"] = list(preserved_tokens)
        if sampling:
            body.update(sampling)
        if extra:
            body.update(extra)
        return body

    def _post_json(self, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.session.post(url, json=body, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ServerError(f"POST {url} failed: {exc}") from exc
        if response.status_code >= 400:
            raise ServerError(self._http_error_message(response))
        try:
            payload = response.json()
        except ValueError as exc:
            raise ServerError(f"POST {url}: invalid JSON response: {exc}") from exc
        if isinstance(payload, dict) and "error" in payload:
            error = payload["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise ServerError(f"server error: {message}")
        return payload

    @staticmethod
    def _http_error_message(response: "requests.Response") -> str:
        text = (response.text or "").strip()
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                error = payload.get("error")
                if isinstance(error, dict) and "message" in error:
                    return f"HTTP {response.status_code}: {error['message']}"
                if isinstance(error, str):
                    return f"HTTP {response.status_code}: {error}"
        except ValueError:
            pass
        return f"HTTP {response.status_code}: {text[:800]}"

    @staticmethod
    def _iter_sse(response: "requests.Response") -> Iterator[Any]:
        data_lines: List[str] = []
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.rstrip("\r")
            if line == "":
                if data_lines:
                    chunk = "\n".join(data_lines)
                    data_lines = []
                    yield from _decode_sse_payload(chunk)
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif line.startswith("event:") or line.startswith("id:"):
                continue
            else:
                # server that streams bare JSON lines
                data_lines.append(line)
        if data_lines:
            yield from _decode_sse_payload("\n".join(data_lines))

    def _parse_response(
        self, payload: Dict[str, Any], *, streamed_text: Optional[str] = None
    ) -> CompletionResult:
        content = payload.get("content")
        if not isinstance(content, str):
            generated = payload.get("generated_text")
            if isinstance(generated, str):
                content = generated
            elif streamed_text is not None:
                content = streamed_text
            else:
                content = ""

        warnings: List[str] = []
        stopped_eos = bool(payload.get("stopped_eos", False))
        stopped_word = bool(payload.get("stopped_word", False))
        stopped_limit = bool(payload.get("stopped_limit", False))
        stop_word = payload.get("stopping_word") or payload.get("stop_word") or ""

        # When streaming, llama.cpp reports the reconstructed content only in
        # the final event; if the final event is missing we cannot know which
        # stop word fired.
        if streamed_text is not None and not payload:
            warnings.append("stream ended without a final event")

        text = content
        if stopped_word and stop_word:
            text = text + stop_word
        elif stopped_eos:
            eos = payload.get("eos_token")
            if isinstance(eos, str) and eos:
                text = text + eos

        return CompletionResult(
            text=text,
            content=content,
            stopped_eos=stopped_eos,
            stopped_word=stopped_word,
            stopped_limit=stopped_limit,
            stop_word=stop_word,
            tokens_predicted=int(payload.get("tokens_predicted") or 0),
            tokens_evaluated=int(payload.get("tokens_evaluated") or 0),
            model=str(payload.get("model") or ""),
            timings=payload.get("timings") or {},
            raw=payload,
            warnings=warnings,
        )


def _decode_sse_payload(chunk: str) -> Iterator[Any]:
    """Decode one SSE ``data:`` payload (JSON, ``[DONE]`` or raw text)."""
    chunk = chunk.strip()
    if not chunk:
        return
    if chunk == "[DONE]":
        return
    try:
        yield json.loads(chunk)
    except ValueError:
        # Bare text stream (some servers stream plain tokens).
        yield {"content": chunk}


def format_timings(timings: Dict[str, Any]) -> str:
    """Human readable timings summary."""
    if not timings:
        return ""
    parts = []
    if timings.get("prompt_n"):
        parts.append(
            f"prompt {timings['prompt_n']} tok in {timings.get('prompt_ms', 0) / 1000:.2f}s "
            f"({timings.get('prompt_per_second', 0):.1f} tok/s)"
        )
    if timings.get("predicted_n"):
        parts.append(
            f"generated {timings['predicted_n']} tok in {timings.get('predicted_ms', 0) / 1000:.2f}s "
            f"({timings.get('predicted_per_second', 0):.2f} tok/s)"
        )
    return ", ".join(parts)
