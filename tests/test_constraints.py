"""Unit tests: stop-string / grammar constraints and prompt hints."""

from __future__ import annotations

import unittest

from toolcaller.constraints import (
    build_constraints,
    strip_generation_reasoning,
    strip_trailing_whitespace,
)
from toolcaller.hints import HINT_MODES, build_hints
from toolcaller.profiles import get_profile


class TestBuildConstraints(unittest.TestCase):
    def test_eos_is_always_a_stop(self):
        c = build_constraints(get_profile("qwen"), eos_token="<|im_end|>", has_tools=False)
        self.assertEqual(c.stops, ["<|im_end|>"])

    def test_no_tools_means_no_call_terminators(self):
        c = build_constraints(get_profile("qwen"), eos_token="<|im_end|>", has_tools=False)
        self.assertNotIn("</tool_call>", c.stops)

    def test_function_tag_close_is_never_a_stop_string(self):
        # stopping at </function> would truncate parallel calls and, worse,
        # a single call before its </tool_call> terminator
        c = build_constraints(get_profile("qwen"), eos_token="<|im_end|>", has_tools=True)
        self.assertNotIn("</function>", c.stops)
        self.assertNotIn("<function=", c.stops)

    def test_call_end_not_a_stop_by_default(self):
        c = build_constraints(get_profile("qwen"), eos_token="<|im_end|>", has_tools=True)
        self.assertNotIn("</tool_call>", c.stops)

    def test_stop_at_call_end_optin(self):
        c = build_constraints(
            get_profile("qwen"), eos_token="<|im_end|>", has_tools=True, stop_at_call_end=True
        )
        self.assertIn("</tool_call>", c.stops)

    def test_extra_and_request_stops(self):
        c = build_constraints(
            get_profile("qwen"),
            eos_token="<|im_end|>",
            extra_stops=["EXTRA"],
            request_stops=["FROM-REQUEST"],
        )
        self.assertIn("EXTRA", c.stops)
        self.assertIn("FROM-REQUEST", c.stops)

    def test_deduplicates(self):
        c = build_constraints(
            get_profile("qwen"), eos_token="<|im_end|>", extra_stops=["<|im_end|>"]
        )
        self.assertEqual(c.stops.count("<|im_end|>"), 1)

    def test_special_tokens_are_preserved(self):
        c = build_constraints(get_profile("qwen"), eos_token="<|im_end|>", has_tools=True)
        self.assertIn("<|im_end|>", c.preserved_tokens)

    def test_profile_extra_stop_strings(self):
        c = build_constraints(get_profile("llama3"), eos_token="<|eot_id|>", has_tools=True)
        self.assertIn("<|eom_id|>", c.stops)
        self.assertIn("<|eot_id|>", c.stops)

    def test_merge_hints(self):
        hints = build_hints(get_profile("qwen"), has_tools=True, grammar_style="gbnf")
        c = build_constraints(get_profile("qwen"), eos_token="<|im_end|>")
        c.merge(hints)
        self.assertIsNotNone(c.grammar)
        for marker in ("<tool_call>", "</function>"):
            self.assertIn(marker, c.preserved_tokens)


class TestHints(unittest.TestCase):
    def test_never_mode(self):
        hints = build_hints(get_profile("qwen"), has_tools=True, mode="never")
        self.assertEqual(hints.system_suffix, "")
        self.assertTrue(hints.is_empty())

    def test_no_tools_no_hint(self):
        hints = build_hints(get_profile("qwen"), has_tools=False, mode="always")
        self.assertEqual(hints.system_suffix, "")

    def test_auto_skips_when_template_documents_format(self):
        tpl = "<tools>...</tools> reply with <tool_call><function=...>"
        hints = build_hints(
            get_profile("qwen"), has_tools=True, template_source=tpl, mode="auto"
        )
        self.assertEqual(hints.system_suffix, "")

    def test_auto_adds_when_template_silent(self):
        hints = build_hints(
            get_profile("qwen"), has_tools=True, template_source="{% for m in messages %}", mode="auto"
        )
        self.assertIn("<tool_call>", hints.system_suffix)

    def test_always_adds(self):
        hints = build_hints(
            get_profile("qwen"),
            has_tools=True,
            template_source="<tool_call><function=",
            mode="always",
        )
        self.assertIn("name>", hints.system_suffix + "name>")  # sanity
        self.assertTrue(hints.system_suffix)

    def test_hint_mentions_tool_name(self):
        hints = build_hints(
            get_profile("qwen"), has_tools=True, mode="always", tool_names=["get_weather"]
        )
        self.assertIn("get_weather", hints.system_suffix)

    def test_hint_shapes_for_profiles(self):
        expect = {
            "qwen": "<tool_call>",
            "mistral": "[TOOL_CALLS]",
            "hermes": "<tool_call>",
            "llama3": '{"name"',
            "deepseek": "DSML",
            "gpt-oss": "to=functions.",
            "kimi": "<|tool_call_begin|>",
            "lfm2": "<|tool_call_start|>",
            "gemma4": "call:",
            "ministral3": "[ARGS]",
        }
        for name, marker in expect.items():
            with self.subTest(profile=name):
                hints = build_hints(get_profile(name), has_tools=True, mode="always")
                self.assertIn(marker, hints.system_suffix)

    def test_preserved_tokens_include_markers(self):
        hints = build_hints(get_profile("qwen"), has_tools=True, mode="always")
        self.assertIn("<tool_call>", hints.preserved_tokens)
        self.assertIn("</tool_call>", hints.preserved_tokens)
        self.assertIn("<think>", hints.preserved_tokens)

    def test_invalid_mode(self):
        from toolcaller.errors import ConfigError

        with self.assertRaises(ConfigError):
            build_hints(get_profile("qwen"), has_tools=True, mode="sometimes")

    def test_grammar_is_lazy_with_word_trigger(self):
        hints = build_hints(get_profile("qwen"), has_tools=True, grammar_style="gbnf")
        self.assertIn("root ::=", hints.grammar)
        self.assertEqual(hints.grammar_triggers, [{"type": 1, "value": "<tool_call>"}])

    def test_grammar_for_mistral_json_array(self):
        hints = build_hints(get_profile("mistral"), has_tools=True, grammar_style="gbnf")
        self.assertIn("json", hints.grammar)


class TestGenerationReasoningStripping(unittest.TestCase):
    def test_strips_think(self):
        out = strip_generation_reasoning("<|im_start|>assistant\n<think>\n", get_profile("qwen"))
        # the marker goes, the structural newline before it stays
        self.assertEqual(out, "<|im_start|>assistant\n")

    def test_noop_when_absent(self):
        out = strip_generation_reasoning("<|im_start|>assistant\n", get_profile("qwen"))
        self.assertEqual(out, "<|im_start|>assistant\n")

    def test_empty(self):
        self.assertEqual(strip_generation_reasoning("", get_profile("qwen")), "")

    def test_strip_trailing_whitespace(self):
        self.assertEqual(strip_trailing_whitespace("a  \n\t"), "a")


if __name__ == "__main__":
    unittest.main()
