"""Unit tests for the daemon's request routing (``--no-stream`` handling)."""

from __future__ import annotations

import unittest

from toolcaller.server import (
    ServerConfig,
    build_sse_stream,
    stream_requested,
    upstream_stream_enabled,
    wants_stream_usage,
)
from toolcaller.streamer import IncrementalStreamParser
from toolcaller.profiles import get_profile


class TestStreamRequested(unittest.TestCase):
    """``stream_requested`` reflects what the *client* asked for."""

    def test_stream_false_by_default(self):
        self.assertFalse(stream_requested({}, ServerConfig()))

    def test_stream_true_is_honoured_by_default(self):
        self.assertTrue(stream_requested({"stream": True}, ServerConfig()))

    def test_client_stream_is_honoured_even_with_no_stream(self):
        # --no-stream affects the upstream call, not the response format: a
        # client that asked for SSE must still receive SSE (agentic harnesses
        # always parse the body as a stream and fail otherwise).
        config = ServerConfig(no_stream=True)
        self.assertTrue(stream_requested({"stream": True}, config))

    def test_falsy_stream_values(self):
        for value in (None, 0, "", False):
            self.assertFalse(stream_requested({"stream": value}, ServerConfig()))


class TestUpstreamStreaming(unittest.TestCase):
    """``--no-stream`` disables streaming *from the backend*."""

    def test_streaming_by_default(self):
        self.assertTrue(upstream_stream_enabled(ServerConfig()))

    def test_no_stream_disables_upstream_streaming(self):
        self.assertFalse(upstream_stream_enabled(ServerConfig(no_stream=True)))


class TestStreamUsage(unittest.TestCase):
    def test_usage_requested(self):
        self.assertTrue(
            wants_stream_usage({"stream_options": {"include_usage": True}})
        )

    def test_usage_not_requested(self):
        self.assertFalse(wants_stream_usage({}))
        self.assertFalse(wants_stream_usage({"stream_options": {}}))
        self.assertFalse(wants_stream_usage({"stream_options": {"include_usage": False}}))

    def test_usage_trailer_chunk(self):
        stream = build_sse_stream(
            [], finish_reason="stop", usage={"prompt_tokens": 1,
                                            "completion_tokens": 2,
                                            "total_tokens": 3},
        )
        trailer = stream[-2]
        self.assertEqual(trailer["choices"], [])
        self.assertEqual(trailer["usage"]["total_tokens"], 3)
        self.assertEqual(stream[-1], "[DONE]")

    def test_no_usage_trailer_by_default(self):
        stream = build_sse_stream([], finish_reason="stop")
        self.assertEqual(stream[-1], "[DONE]")
        self.assertTrue(all(c["choices"] for c in stream[:-1]))


class TestInBandSSE(unittest.TestCase):
    """The daemon must emit Pi's in-band reasoning/tool-call delta format."""

    def _events(self, text: str, profile: str = "spark", prefilled: bool = True):
        parser = IncrementalStreamParser(
            get_profile(profile), reasoning_prefilled=prefilled
        )
        events = []
        for index in range(0, len(text), 3):
            events.extend(parser.feed(text[index:index + 3]))
        events.extend(parser.finalize())
        return events

    def _stream(self, text: str, *, finish_reason: str, profile: str = "spark",
                prefilled: bool = True):
        return build_sse_stream(
            self._events(text, profile, prefilled=prefilled),
            finish_reason=finish_reason, request_id="chatcmpl-x", created=1,
        )

    def test_role_chunk_first(self):
        stream = self._stream("hello", finish_reason="stop")
        self.assertEqual(stream[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(stream[-1], "[DONE]")

    def test_reasoning_uses_reasoning_content(self):
        stream = self._stream("thoughts</think>answer", finish_reason="stop")
        deltas = [c["choices"][0]["delta"] for c in stream[:-1]]
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
        self.assertEqual(reasoning, "thoughts")
        self.assertEqual(content, "answer")

    def test_tool_call_is_in_band_with_incremental_arguments(self):
        text = (
            "hmm</think><tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Budapest</arg_value></tool_call>"
        )
        stream = self._stream(text, finish_reason="tool_calls")
        deltas = [c["choices"][0]["delta"] for c in stream[:-1]]

        # No markup leaks into content.
        content = "".join(d.get("content", "") for d in deltas)
        self.assertNotIn("<tool_call>", content)
        self.assertNotIn("</think>", content)

        tool_deltas = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
        self.assertGreaterEqual(len(tool_deltas), 2, "arguments must be incremental")
        first = tool_deltas[0]
        self.assertEqual(first["index"], 0)
        self.assertEqual(first["type"], "function")
        self.assertEqual(first["function"]["name"], "get_weather")
        self.assertTrue(first["id"])

        import json
        args = "".join(t.get("function", {}).get("arguments", "") for t in tool_deltas)
        self.assertEqual(json.loads(args), {"city": "Budapest"})

    def test_finish_reason_is_last(self):
        stream = self._stream("<tool_call>f</tool_call>", finish_reason="tool_calls")
        finish = stream[-2]["choices"][0]
        self.assertEqual(finish["finish_reason"], "tool_calls")
        self.assertEqual(finish["delta"], {})

    def test_parallel_tool_calls_keep_distinct_indices(self):
        text = (
            "<tool_call>a<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
            "<tool_call>b<arg_key>y</arg_key><arg_value>2</arg_value></tool_call>"
        )
        stream = self._stream(text, finish_reason="tool_calls")
        deltas = [c["choices"][0]["delta"] for c in stream[:-1]]
        indices = {
            d["tool_calls"][0]["index"] for d in deltas if "tool_calls" in d
        }
        self.assertEqual(indices, {0, 1})

    def test_plain_text_has_no_tool_calls(self):
        stream = self._stream("just an answer", finish_reason="stop", prefilled=False)
        deltas = [c["choices"][0]["delta"] for c in stream[:-1]]
        self.assertFalse(any("tool_calls" in d for d in deltas))
        self.assertEqual("".join(d.get("content", "") for d in deltas), "just an answer")


if __name__ == "__main__":
    unittest.main()
