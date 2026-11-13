"""Tests for the incremental (in-band) stream splitter."""

from __future__ import annotations

import json
import unittest

from toolcaller.profiles import get_profile
from toolcaller.streamer import (
    CONTENT,
    REASONING,
    TOOL_ARGS,
    TOOL_START,
    IncrementalStreamParser,
)


def run(text: str, profile: str, *, chunk: int = 3, prefilled: bool = False,
        reasoning_format: str = "auto"):
    """Feed ``text`` in fixed-size chunks and return all events."""
    parser = IncrementalStreamParser(
        get_profile(profile),
        reasoning_format=reasoning_format,
        reasoning_prefilled=prefilled,
    )
    events = []
    for index in range(0, len(text), chunk):
        events.extend(parser.feed(text[index:index + chunk]))
    events.extend(parser.finalize())
    return events


def join(events, kind: str) -> str:
    return "".join(payload for k, payload in events if k == kind)


def tool_starts(events):
    return [payload for k, payload in events if k == TOOL_START]


def tool_args(events):
    """Return ``{index: accumulated argument string}``."""
    out = {}
    for kind, payload in events:
        if kind == TOOL_ARGS:
            index, fragment = payload
            out[index] = out.get(index, "") + fragment
    return out


class TestPlainText(unittest.TestCase):
    def test_content_only(self):
        events = run("Hello there, this is plain text.", "spark")
        self.assertEqual(join(events, CONTENT), "Hello there, this is plain text.")
        self.assertEqual(join(events, REASONING), "")
        self.assertEqual(tool_starts(events), [])

    def test_reasoning_format_none_disables_split(self):
        events = run("a</think>b", "spark", reasoning_format="none")
        self.assertEqual(join(events, REASONING), "")
        self.assertIn("a</think>b", join(events, CONTENT))

    def test_explicit_think_block(self):
        events = run("<think>reasoning here</think>answer", "spark")
        self.assertEqual(join(events, REASONING), "reasoning here")
        self.assertEqual(join(events, CONTENT), "answer")

    def test_prefilled_reasoning(self):
        events = run("thinking...</think>the answer", "spark", prefilled=True)
        self.assertEqual(join(events, REASONING), "thinking...")
        self.assertEqual(join(events, CONTENT), "the answer")

    def test_leading_close_without_start_is_reasoning(self):
        # Models whose prompt prefilled the marker emit a bare ``</think>``.
        # Even when prefill detection fails, the text before it is reasoning.
        events = run("reasoning</think>answer", "spark", prefilled=False)
        self.assertEqual(join(events, REASONING), "reasoning")
        self.assertEqual(join(events, CONTENT), "answer")


class TestSingleToolCall(unittest.TestCase):
    def test_spark_call(self):
        text = (
            "Think.</think><tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Budapest</arg_value></tool_call>"
        )
        events = run(text, "spark", prefilled=True)
        self.assertEqual(join(events, REASONING), "Think.")
        self.assertEqual(join(events, CONTENT), "")
        starts = tool_starts(events)
        self.assertEqual(len(starts), 1)
        index, call_id, name = starts[0]
        self.assertEqual(index, 0)
        self.assertEqual(name, "get_weather")
        self.assertTrue(call_id)
        args = tool_args(events)
        self.assertEqual(json.loads(args[0]), {"city": "Budapest"})

    def test_hermes_call(self):
        text = (
            '<tool_call>\n{"name":"get_weather","arguments":'
            '{"city":"Budapest","days":2}}\n</tool_call>'
        )
        events = run(text, "hermes")
        starts = tool_starts(events)
        self.assertEqual([s[2] for s in starts], ["get_weather"])
        self.assertEqual(
            json.loads(tool_args(events)[0]), {"city": "Budapest", "days": 2}
        )

    def test_qwen_call(self):
        text = (
            "<tool_call>\n<function=get_weather>\n<parameter=city>\n"
            "Budapest\n</parameter>\n</function>\n</tool_call>"
        )
        events = run(text, "qwen")
        self.assertEqual([s[2] for s in tool_starts(events)], ["get_weather"])
        self.assertEqual(json.loads(tool_args(events)[0]), {"city": "Budapest"})

    def test_call_markup_never_leaks_into_content(self):
        text = "Sure.<tool_call>reset</tool_call>"
        events = run(text, "spark")
        self.assertEqual(join(events, CONTENT), "Sure.")
        self.assertNotIn("<tool_call>", join(events, CONTENT))

    def test_no_argument_call(self):
        text = "<tool_call>reset</tool_call>"
        events = run(text, "spark")
        self.assertEqual([s[2] for s in tool_starts(events)], ["reset"])
        self.assertEqual(json.loads(tool_args(events)[0]), {})


class TestParallelToolCalls(unittest.TestCase):
    def test_two_calls_get_distinct_indices(self):
        text = (
            "x</think>"
            "<tool_call>read_file<arg_key>path</arg_key><arg_value>/etc/x</arg_value></tool_call>"
            "<tool_call>bash<arg_key>command</arg_key><arg_value>date</arg_value></tool_call>"
        )
        events = run(text, "spark", prefilled=True)
        starts = tool_starts(events)
        self.assertEqual([(s[0], s[2]) for s in starts], [(0, "read_file"), (1, "bash")])
        args = tool_args(events)
        self.assertEqual(json.loads(args[0]), {"path": "/etc/x"})
        self.assertEqual(json.loads(args[1]), {"command": "date"})
        self.assertNotEqual(starts[0][1], starts[1][1])  # unique ids


class TestIncrementalArguments(unittest.TestCase):
    def test_fragments_concatenate_to_valid_json(self):
        text = (
            '<tool_call>{"name":"f","arguments":{"a":1,"b":"two",'
            '"c":[1,2,3]}}</tool_call>'
        )
        events = run(text, "hermes", chunk=1)
        fragments = [payload[1] for k, payload in events if k == TOOL_ARGS]
        self.assertGreater(len(fragments), 1, "arguments should arrive incrementally")
        self.assertEqual(
            json.loads("".join(fragments)), {"a": 1, "b": "two", "c": [1, 2, 3]}
        )

    def test_partial_json_is_parseable_prefix(self):
        text = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Budapest</arg_value>"
            "</tool_call>"
        )
        events = run(text, "spark", chunk=1)
        accumulated = ""
        checked = 0
        for kind, payload in events:
            if kind != TOOL_ARGS:
                continue
            accumulated += payload[1]
            try:
                json.loads(accumulated)
                checked += 1
            except ValueError:
                # A prefix may be mid-string; only assert it *grows* to valid.
                pass
        self.assertGreaterEqual(checked, 1)
        self.assertEqual(json.loads(accumulated), {"city": "Budapest"})

    def test_truncated_call_still_terminates(self):
        # The value was streamed as an open JSON string before the stream was
        # cut, so the client ends up with the string form "1" (deltas are
        # append-only and cannot be retyped to the number 1).
        text = "<tool_call>f<arg_key>a</arg_key><arg_value>1"
        events = run(text, "spark")
        self.assertEqual([s[2] for s in tool_starts(events)], ["f"])
        self.assertEqual(json.loads(tool_args(events)[0]), {"a": "1"})

    def test_arguments_stream_before_the_closing_tag(self):
        """Argument values reach the client while they are still being written."""
        parser = IncrementalStreamParser(
            get_profile("spark"), reasoning_prefilled=True
        )
        events = []
        for piece in [
            "<tool_call>", "get_weather", "<arg_key>city</arg_key>",
            "<arg_value>Bud", "apest", "</arg_value>", "</tool_call>",
        ]:
            events.extend(parser.feed(piece))
        events.extend(parser.finalize())

        args = tool_args(events)[0]
        self.assertIn("Bud", args, "partial value must be streamed")
        self.assertEqual(json.loads(args), {"city": "Budapest"})
        # The closing tag must not appear in the accumulated arguments.
        self.assertNotIn("arg_value", args)


class TestMarkerSplitAcrossChunks(unittest.TestCase):
    def test_delimiter_split_between_feeds(self):
        # "<tool_ca" + "ll>" must still be recognized as one marker.
        parser = IncrementalStreamParser(get_profile("spark"))
        events = []
        for piece in ["Hi<tool_ca", "ll>reset</tool_c", "all>"]:
            events.extend(parser.feed(piece))
        events.extend(parser.finalize())
        self.assertEqual(join(events, CONTENT), "Hi")
        self.assertEqual([s[2] for s in tool_starts(events)], ["reset"])

    def test_reasoning_end_split_across_chunks(self):
        parser = IncrementalStreamParser(get_profile("spark"), reasoning_prefilled=True)
        events = []
        for piece in ["think</th", "ink>answer"]:
            events.extend(parser.feed(piece))
        events.extend(parser.finalize())
        self.assertEqual(join(events, REASONING), "think")
        self.assertEqual(join(events, CONTENT), "answer")


if __name__ == "__main__":
    unittest.main()
