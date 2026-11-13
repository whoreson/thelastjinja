"""Unit tests: completion client (mocked HTTP) and the end-to-end pipeline."""

from __future__ import annotations

import json
import os
import unittest
from typing import Any, Dict, List, Optional

from toolcaller.api import ChatPipeline, _append_hint, _strip_generation_prompt
from toolcaller.client import CompletionClient, _decode_sse_payload, format_timings
from toolcaller.errors import ServerError
from toolcaller.renderer import RenderResult

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_TEMPLATE = os.path.join(HERE, "qwen.jinja")


class FakeResponse:
    def __init__(self, payload: Any, status: int = 200, lines: Optional[List[str]] = None):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload) if not isinstance(payload, str) else payload
        self._lines = lines or []

    def json(self) -> Any:
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload

    @property
    def content(self) -> bytes:
        return self.text.encode("utf-8")

    def iter_content(self, chunk_size=8192):
        raw = self.content
        start = 0
        while start < len(raw):
            end = start + chunk_size
            yield raw[start:end]
            start = end

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._lines:
            yield line

    def close(self) -> None:
        pass


class FakeSession:
    """Records requests and returns canned responses."""

    def __init__(self, responses: List[FakeResponse]):
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []
        self.headers: Dict[str, str] = {}

    def _next(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"unexpected request: {method} {url}")
        return self.responses.pop(0)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._next("POST", url, **kwargs)

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._next("GET", url, **kwargs)


class TestCompletionClient(unittest.TestCase):
    def _client(self, responses, **kwargs):
        session = FakeSession(responses)
        return CompletionClient("http://x", session=session, **kwargs), session

    def test_complete_reconstructs_stop_word(self):
        client, session = self._client(
            [FakeResponse({"content": "<tool_call>", "stopped_word": True, "stopping_word": "</tool_call>"})]
        )
        result = client.complete("p", stops=["</tool_call>"])
        self.assertEqual(result.text, "<tool_call></tool_call>")
        self.assertTrue(result.stopped_word)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(session.calls[0]["json"]["prompt"], "p")
        self.assertEqual(session.calls[0]["json"]["stop"], ["</tool_call>"])

    def test_complete_appends_eos(self):
        client, _ = self._client(
            [FakeResponse({"content": "hi", "stopped_eos": True, "eos_token": "<|im_end|>"})]
        )
        self.assertEqual(client.complete("p").text, "hi<|im_end|>")

    def test_finish_reason_length(self):
        client, _ = self._client([FakeResponse({"content": "x", "stopped_limit": True})])
        result = client.complete("p", n_predict=5)
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.content, "x")

    def test_server_error_payload(self):
        client, _ = self._client([FakeResponse({"error": {"message": "boom"}})])
        with self.assertRaises(ServerError) as ctx:
            client.complete("p")
        self.assertIn("boom", str(ctx.exception))

    def test_http_error(self):
        client, _ = self._client([FakeResponse("nope", status=500)])
        with self.assertRaises(ServerError) as ctx:
            client.complete("p")
        self.assertIn("500", str(ctx.exception))

    def test_grammar_and_triggers_forwarded(self):
        client, session = self._client([FakeResponse({"content": ""})])
        client.complete(
            "p",
            grammar="root ::= x",
            grammar_triggers=[{"type": 1, "value": "<tool_call>"}],
            preserved_tokens=["<tool_call>"],
        )
        body = session.calls[0]["json"]
        self.assertEqual(body["grammar"], "root ::= x")
        self.assertEqual(body["grammar_triggers"], [{"type": 1, "value": "<tool_call>"}])
        self.assertEqual(body["preserved_tokens"], ["<tool_call>"])

    def test_sampling_params_forwarded(self):
        client, session = self._client([FakeResponse({"content": ""})])
        client.complete("p", sampling={"temperature": 0.0, "top_k": 1})
        body = session.calls[0]["json"]
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["top_k"], 1)

    def test_tokenize_usable(self):
        client, _ = self._client(
            [FakeResponse({"tokens": [1, 2, 3]}), FakeResponse({"tokens": [{"id": 4}]})]
        )
        self.assertEqual(client.tokenize("a"), [1, 2, 3])
        self.assertEqual(client.tokenize("b"), [4])

    def test_tokenize_unsupported_returns_none(self):
        client, _ = self._client([FakeResponse("not found", status=404)])
        self.assertIsNone(client.tokenize("a"))
        # and it is not retried
        self.assertIsNone(client.tokenize("b"))

    def test_stop_token_sequences_cached(self):
        client, session = self._client(
            [FakeResponse({"tokens": [1, 2]})]
        )
        self.assertEqual(client.stop_token_sequences(["</tool_call>"]), [[1, 2]])
        self.assertEqual(client.stop_token_sequences(["</tool_call>"]), [[1, 2]])
        self.assertEqual(len(session.calls), 1)

    def test_props(self):
        client, _ = self._client(
            [FakeResponse({"bos_token": "<s>", "eos_token": "</s>", "chat_template": "x"})]
        )
        props = client.props()
        self.assertEqual(props["eos_token"], "</s>")

    def test_stream_collects_chunks_and_final(self):
        lines = [
            'data: {"content": "Hello"}',
            "",
            'data: {"content": " world"}',
            "",
            'data: {"content": "", "stop": true, "stopped_word": true, "stopping_word": "<|im_end|>", "tokens_predicted": 2}',
            "",
        ]
        client, _ = self._client([FakeResponse({"content": ""}, lines=lines)])
        chunks: List[str] = []
        result = client.complete_stream("p", on_chunk=chunks.append)
        self.assertEqual(chunks, ["Hello", " world"])
        self.assertEqual(result.content, "Hello world")
        self.assertEqual(result.text, "Hello world<|im_end|>")

    def test_stream_with_done_marker(self):
        lines = ['data: {"content": "x"}', "", "data: [DONE]", ""]
        client, _ = self._client([FakeResponse({"content": ""}, lines=lines)])
        result = client.complete_stream("p")
        self.assertEqual(result.text, "x")

    def test_sse_plain_text_payload(self):
        self.assertEqual(list(_decode_sse_payload("raw text")), [{"content": "raw text"}])
        self.assertEqual(list(_decode_sse_payload("[DONE]")), [])

    def test_format_timings(self):
        text = format_timings(
            {
                "prompt_n": 10,
                "prompt_ms": 1000,
                "prompt_per_second": 10,
                "predicted_n": 5,
                "predicted_ms": 5000,
                "predicted_per_second": 1,
            }
        )
        self.assertIn("prompt 10 tok", text)
        self.assertIn("generated 5 tok", text)
        self.assertEqual(format_timings({}), "")


class PipelineFakeClient(CompletionClient):
    """Client that returns a canned raw completion."""

    def __init__(self, text: str, **kwargs: Any):
        super().__init__("http://x")
        self._text = text
        self.last_body: Dict[str, Any] = {}
        self.last_stops: List[str] = []

    def props(self):  # noqa: D102
        return {"bos_token": "<|endoftext|>", "eos_token": "<|im_end|>"}

    def complete(self, prompt: str, **kwargs: Any):
        from toolcaller.client import CompletionResult

        self.last_body = dict(kwargs, prompt=prompt)
        self.last_stops = list(kwargs.get("stops") or [])
        return CompletionResult(
            text=self._text,
            content=self._text,
            stopped_eos=True,
            tokens_predicted=len(self._text.split()),
            tokens_evaluated=10,
            model="fake",
        )


def make_pipeline(text: str, **kwargs: Any) -> ChatPipeline:
    kwargs.setdefault("hint_mode", "never")
    kwargs.setdefault("template_kwargs", {"enable_thinking": False})
    return ChatPipeline(
        template_path=QWEN_TEMPLATE,
        client=PipelineFakeClient(text),
        discover_tokens=True,
        **kwargs,
    )


TOOL_CALL = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nBudapest\n"
    "</parameter>\n</function>\n</tool_call>"
)


class TestPipeline(unittest.TestCase):
    def test_prompt_reaches_client(self):
        pipe = make_pipeline("hello")
        pipe.complete({"messages": [{"role": "user", "content": "hi"}]})
        prompt = pipe.client.last_body["prompt"]
        self.assertTrue(prompt.startswith("<|im_start|>user\nhi<|im_end|>"))
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_tool_call_parsed_into_openai_message(self):
        pipe = make_pipeline(TOOL_CALL)
        result = pipe.complete(
            {
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {"type": "object"}},
                    }
                ],
            }
        )
        self.assertEqual(result.finish_reason, "tool_calls")
        message = result.message
        self.assertIsNone(message["content"])
        call = message["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Budapest"})

    def test_eos_is_a_stop(self):
        pipe = make_pipeline("x")
        pipe.complete({"messages": [{"role": "user", "content": "hi"}]})
        self.assertIn("<|im_end|>", pipe.client.last_stops)

    def test_max_tokens_forwarded(self):
        pipe = make_pipeline("x")
        pipe.complete({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 42})
        self.assertEqual(pipe.client.last_body["n_predict"], 42)

    def test_unlimited_when_absent(self):
        pipe = make_pipeline("x")
        pipe.complete({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(pipe.client.last_body["n_predict"], -1)

    def test_openai_response_shape(self):
        pipe = make_pipeline("Just answer.")
        result = pipe.complete({"messages": [{"role": "user", "content": "hi"}]})
        response = result.to_response()
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"]["content"], "Just answer.")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertIn("usage", response)
        self.assertTrue(response["id"].startswith("chatcmpl-"))

    def test_stream_chunks(self):
        pipe = make_pipeline(TOOL_CALL)
        result = pipe.complete({"messages": [{"role": "user", "content": "hi"}]})
        chunks = result.stream_chunks()
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
        tool_chunks = [c for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
        self.assertEqual(len(tool_chunks), 1)

    def test_stream_events_from_completed_result(self):
        """A buffered (non-streamed) result can still be rendered as SSE."""
        from toolcaller.server import build_sse_stream

        pipe = make_pipeline(TOOL_CALL)
        result = pipe.complete({"messages": [{"role": "user", "content": "hi"}]})
        stream = build_sse_stream(
            result.stream_events(), finish_reason=result.finish_reason,
            request_id="chatcmpl-t", created=1,
        )
        self.assertEqual(stream[-1], "[DONE]")
        finish = stream[-2]["choices"][0]
        self.assertEqual(finish["finish_reason"], "tool_calls")

        deltas = [c["choices"][0]["delta"] for c in stream[:-1]]
        self.assertFalse(any("<tool_call>" in (d.get("content") or "") for d in deltas))
        tool_deltas = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
        self.assertEqual(tool_deltas[0]["function"]["name"], "get_weather")
        args = "".join(
            t.get("function", {}).get("arguments", "") for t in tool_deltas
        )
        self.assertEqual(json.loads(args), {"city": "Budapest"})

    def test_tool_choice_none_drops_calls(self):
        pipe = make_pipeline(TOOL_CALL)
        result = pipe.complete(
            {"messages": [{"role": "user", "content": "hi"}], "tool_choice": "none"}
        )
        self.assertNotIn("tool_calls", result.message)
        self.assertEqual(result.finish_reason, "stop")

    def test_per_request_profile_override(self):
        pipe = make_pipeline('<tool_call>{"name":"f","arguments":{"a":1}}</tool_call>')
        result = pipe.complete(
            {"messages": [{"role": "user", "content": "hi"}]}, profile="hermes"
        )
        self.assertEqual(result.profile.name, "hermes")
        self.assertEqual(result.parsed.tool_calls[0].name, "f")

    def test_reasoning_split(self):
        pipe = make_pipeline("<think>\nhmm\n</think>\n\nThe answer.")
        result = pipe.complete({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(result.parsed.reasoning_content, "hmm")
        self.assertEqual(result.message["content"], "The answer.")

    def test_describe(self):
        pipe = make_pipeline("x")
        info = pipe.describe()
        self.assertEqual(info["profile"], "qwen")
        self.assertEqual(info["eos_token"], "<|im_end|>")
        self.assertIn("caps", info)


class TestHintInjection(unittest.TestCase):
    def test_append_to_existing_system(self):
        request = {
            "messages": [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "u"},
            ]
        }
        out = _append_hint(request, "\nHINT")
        self.assertEqual(out["messages"][0]["content"], "SYS\nHINT")
        self.assertEqual(out["messages"][1]["content"], "u")
        # original request untouched
        self.assertEqual(request["messages"][0]["content"], "SYS")

    def test_append_to_content_parts(self):
        request = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "SYS"}]},
                {"role": "user", "content": "u"},
            ]
        }
        out = _append_hint(request, "HINT")
        self.assertEqual(out["messages"][0]["content"][-1], {"type": "text", "text": "HINT"})

    def test_insert_system_when_absent(self):
        request = {"messages": [{"role": "user", "content": "u"}]}
        out = _append_hint(request, "\nHINT")
        self.assertEqual(out["messages"][0]["role"], "system")
        self.assertEqual(out["messages"][1]["role"], "user")

    def test_hint_reaches_rendered_prompt(self):
        pipe = make_pipeline("x", hint_mode="always")
        pipe.complete(
            {
                "messages": [{"role": "system", "content": "SYS."}, {"role": "user", "content": "u"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            }
        )
        prompt = pipe.client.last_body["prompt"]
        self.assertIn("SYS.", prompt)
        self.assertIn("<tool_call>", prompt)


class TestStripGenerationPrompt(unittest.TestCase):
    def test_strips_echoed_prefix(self):
        render = RenderResult(
            prompt="abcGEN", generation_prompt="GEN", chat_messages=[], caps=None
        )
        self.assertEqual(_strip_generation_prompt("GENhello", render), "hello")

    def test_no_prefix(self):
        render = RenderResult(
            prompt="abcGEN", generation_prompt="GEN", chat_messages=[], caps=None
        )
        self.assertEqual(_strip_generation_prompt("hello", render), "hello")


if __name__ == "__main__":
    unittest.main()


class TestStripGenerationThinking(unittest.TestCase):
    def test_prompt_loses_think_marker(self):
        # thinking must be enabled for the template to pre-open the block
        pipe = make_pipeline("ok", strip_generation_thinking=True, template_kwargs={})
        pipe.complete({"messages": [{"role": "user", "content": "hi"}]})
        prompt = pipe.client.last_body["prompt"]
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", prompt)

    def test_prompt_keeps_think_by_default(self):
        pipe = make_pipeline("ok")
        pipe.complete({"messages": [{"role": "user", "content": "hi"}]})
        self.assertIn("<think>", pipe.client.last_body["prompt"])

    def test_reasoning_split_after_stripping(self):
        # the model now emits <think> itself
        text = "<think>\n7*6=42\n</think>\n\n42"
        pipe = make_pipeline(text, strip_generation_thinking=True)
        result = pipe.complete({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(result.parsed.reasoning_content, "7*6=42")
        self.assertEqual(result.message["content"], "42")


class TestCancellation(unittest.TestCase):
    def test_cancel_during_stream(self):
        """When cancel event fires mid-stream, client closes and returns cancelled."""
        import threading

        cancel = threading.Event()
        cancel.set()  # pre-set so cancel is detected on first iteration
        lines = [
            "data: {\"content\": \"hello\"}",
            "",
            "data: {\"content\": \" world\"}",
            "",
        ]
        resp = FakeResponse({"content": "hello world"}, lines=lines)
        session = FakeSession([resp])
        client = CompletionClient("http://x", session=session)
        result = client.complete_stream("p", cancel=cancel)
        self.assertTrue(result.cancelled)
        self.assertEqual(result.finish_reason, "stop")

    def test_cancel_during_nonstream(self):
        import threading
        cancel = threading.Event()
        cancel.set()  # pre-cancel
        resp = FakeResponse({"content": "hi", "stopped_eos": True})
        session = FakeSession([resp])
        client = CompletionClient("http://x", session=session)
        result = client.complete("p", cancel=cancel)
        self.assertTrue(result.cancelled)

    def test_cancel_iter_content(self):
        import threading
        cancel = threading.Event()
        cancel.set()  # fire immediately so iter_content bails
        resp = FakeResponse({"content": "big response", "stopped_eos": True})
        session = FakeSession([resp])
        client = CompletionClient("http://x", session=session)
        result = client.complete("p", cancel=cancel)
        self.assertTrue(result.cancelled)


class TestStopTypeParsing(unittest.TestCase):
    """PicoLM uses stop_type string instead of stopped_eos/word/limit booleans."""

    def _make_result(self, payload: dict) -> Any:
        from toolcaller.client import CompletionClient
        client = CompletionClient("http://x")
        client._tokenize_supported = True
        return client._parse_response(payload)

    def test_stop_type_eos(self):
        r = self._make_result({"content": "hi", "stop_type": "eos"})
        self.assertTrue(r.stopped_eos)
        self.assertEqual(r.finish_reason, "stop")

    def test_stop_type_word(self):
        r = self._make_result({"content": "hello", "stop_type": "word", "stopping_word": "END"})
        self.assertTrue(r.stopped_word)
        self.assertEqual(r.stop_word, "END")
        self.assertEqual(r.finish_reason, "stop")

    def test_stop_type_limit(self):
        r = self._make_result({"content": "hello world", "stop_type": "limit"})
        self.assertTrue(r.stopped_limit)
        self.assertEqual(r.finish_reason, "length")

    def test_stop_type_none_with_content(self):
        # PicoLM streaming final event can report "none" when hitting token limit.
        r = self._make_result({"content": "generated text", "stop_type": "none", "tokens_evaluated": 10})
        self.assertTrue(r.stopped_limit)
        self.assertEqual(r.finish_reason, "length")

    def test_stop_type_none_empty_content(self):
        # Empty content with "none" should not be treated as limit.
        r = self._make_result({"content": "", "stop_type": "none"})
        self.assertFalse(r.stopped_limit)
        self.assertEqual(r.finish_reason, "length")

    def test_legacy_booleans_still_work(self):
        # llama.cpp style booleans still take precedence.
        r = self._make_result({"content": "x", "stopped_eos": True})
        self.assertTrue(r.stopped_eos)

    def test_legacy_booleans_word(self):
        r = self._make_result({"content": "x", "stopped_word": True, "stopping_word": "STOP"})
        self.assertTrue(r.stopped_word)
        self.assertEqual(r.stop_word, "STOP")
