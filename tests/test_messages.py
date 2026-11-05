"""Unit tests: message normalisation (llama.cpp semantics)."""

from __future__ import annotations

import json
import unittest

from toolcaller.errors import RequestError
from toolcaller.messages import (
    dump_json,
    messages_to_json,
    normalize_content,
    normalize_message,
    normalize_messages,
    normalize_request,
)


class TestDumpJson(unittest.TestCase):
    def test_compact_and_ordered(self):
        # llama.cpp dumps compact JSON with preserved key order and no ASCII escaping
        self.assertEqual(dump_json({"b": 1, "a": "é"}), '{"b":1,"a":"é"}')

    def test_list(self):
        self.assertEqual(dump_json([1, {"x": None}]), '[1,{"x":null}]')


class TestNormalizeMessage(unittest.TestCase):
    def test_string_content(self):
        msg = normalize_message({"role": "user", "content": "hi"})
        self.assertEqual(msg, {"role": "user", "content": "hi"})

    def test_null_content_with_tool_calls(self):
        msg = normalize_message(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"a":1}'},
                    }
                ],
            }
        )
        self.assertIsNone(msg["content"])
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "f")
        # arguments stay a JSON string internally
        self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], '{"a":1}')

    def test_object_arguments_are_dumped(self):
        msg = normalize_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": "f", "arguments": {"a": 1}}}
                ],
            }
        )
        self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], '{"a":1}')

    def test_missing_call_id_is_none(self):
        msg = normalize_message(
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": ""}}],
            }
        )
        self.assertIsNone(msg["tool_calls"][0]["id"])

    def test_content_parts(self):
        msg = normalize_message(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        )
        self.assertEqual(msg["content"][0], {"type": "text", "text": "a"})
        self.assertEqual(msg["content"][1]["type"], "image_url")

    def test_unknown_part_passthrough(self):
        msg = normalize_message(
            {"role": "user", "content": [{"type": "custom", "payload": 1}]}
        )
        self.assertEqual(msg["content"][0]["type"], "custom")

    def test_reasoning_and_name_and_tool_call_id(self):
        msg = normalize_message(
            {
                "role": "tool",
                "content": "r",
                "tool_call_id": "c1",
                "name": "f",
                "reasoning_content": "why",
            }
        )
        self.assertEqual(msg["tool_call_id"], "c1")
        self.assertEqual(msg["name"], "f")
        self.assertEqual(msg["reasoning_content"], "why")

    def test_legacy_function_call(self):
        msg = normalize_message(
            {
                "role": "assistant",
                "function_call": {"name": "f", "arguments": '{"a":1}'},
            }
        )
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "f")

    def test_missing_role(self):
        with self.assertRaises(RequestError):
            normalize_message({"content": "x"})

    def test_missing_content_and_calls(self):
        with self.assertRaises(RequestError):
            normalize_message({"role": "user"})

    def test_bad_tool_call_type(self):
        with self.assertRaises(RequestError):
            normalize_message(
                {
                    "role": "assistant",
                    "tool_calls": [{"type": "other", "function": {"name": "f"}}],
                }
            )

    def test_bad_content_type(self):
        # scalars are coerced (llama.cpp is lenient here), mappings are not
        with self.assertRaises(RequestError):
            normalize_message({"role": "user", "content": {"unexpected": 1}})
        self.assertEqual(
            normalize_message({"role": "user", "content": 1.5})["content"], "1.5"
        )


class TestNormalizeMessages(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(RequestError):
            normalize_messages([])

    def test_not_a_list(self):
        with self.assertRaises(RequestError):
            normalize_messages({"role": "user"})

    def test_index_in_error(self):
        with self.assertRaises(RequestError) as ctx:
            normalize_messages([{"role": "user", "content": "ok"}, {"content": "x"}])
        self.assertIn("messages[1]", str(ctx.exception))


class TestMessagesToJson(unittest.TestCase):
    def _msgs(self):
        return normalize_messages(
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"city":"Budapest"}'},
                        }
                    ],
                },
            ]
        )

    def test_string_arguments_by_default(self):
        out = messages_to_json(self._msgs())
        self.assertEqual(out[1]["tool_calls"][0]["function"]["arguments"], '{"city":"Budapest"}')
        self.assertEqual(out[1]["content"], "")

    def test_object_arguments(self):
        out = messages_to_json(self._msgs(), object_arguments=True)
        self.assertEqual(
            out[1]["tool_calls"][0]["function"]["arguments"], {"city": "Budapest"}
        )

    def test_invalid_json_stays_string(self):
        msgs = normalize_messages(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "f", "arguments": "not json"}}
                    ],
                }
            ]
        )
        out = messages_to_json(msgs, object_arguments=True)
        self.assertEqual(out[0]["tool_calls"][0]["function"]["arguments"], "not json")

    def test_concat_typed_text(self):
        msgs = normalize_messages(
            [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
        )
        out = messages_to_json(msgs, concat_typed_text=True)
        self.assertEqual(out[0]["content"], "a\nb")

    def test_omits_absent_fields(self):
        out = messages_to_json(normalize_messages([{"role": "user", "content": "x"}]))
        self.assertEqual(set(out[0]), {"role", "content"})


class TestNormalizeRequest(unittest.TestCase):
    def test_messages_required(self):
        with self.assertRaises(RequestError):
            normalize_request({"model": "x"})

    def test_tools_must_be_list(self):
        with self.assertRaises(RequestError):
            normalize_request({"messages": [{"role": "user", "content": "x"}], "tools": {}})

    def test_keeps_extra_keys(self):
        req = normalize_request(
            {"messages": [{"role": "user", "content": "x"}], "temperature": 0.5}
        )
        self.assertEqual(req["temperature"], 0.5)


if __name__ == "__main__":
    unittest.main()
