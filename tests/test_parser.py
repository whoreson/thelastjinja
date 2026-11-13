"""Unit tests: model-specific tool-call parsing for every built-in profile."""

from __future__ import annotations

import json
import unittest

from toolcaller.errors import ParseError
from toolcaller.parser import (
    SyntaxParser,
    parse_completion,
    resolve_profile,
    split_reasoning,
    ParserSelection,
)
from toolcaller.profiles import get_profile
from toolcaller.syntax import (
    SyntaxProfile,
    balanced_json_span,
    clean_xml_value,
    decode_python_kwargs,
    decode_xml_parameters,
    strip_code_fence,
)


def calls(text: str, profile: str, **kwargs):
    parsed = parse_completion(text, profile=profile, **kwargs)
    return [(c.name, c.arguments) for c in parsed.tool_calls], parsed


class TestSyntaxHelpers(unittest.TestCase):
    def test_strip_code_fence(self):
        self.assertEqual(strip_code_fence("```json\n{}\n```"), "{}")
        self.assertEqual(strip_code_fence("{}"), "{}")

    def test_balanced_json_span_object(self):
        # the scanner starts at the first structural character
        self.assertIsNone(balanced_json_span('pre {"a": 1}'))
        span, end = balanced_json_span('{"a": {"b": "}{"}} post')
        self.assertEqual(span, '{"a": {"b": "}{"}}')
        self.assertEqual('{"a": {"b": "}{"}} post'[end:], " post")

    def test_balanced_json_span_array(self):
        span, _ = balanced_json_span('[1, [2, {"x": "]"}] ]')
        self.assertEqual(span, '[1, [2, {"x": "]"}] ]')

    def test_balanced_json_span_none(self):
        self.assertIsNone(balanced_json_span("no json here"))
        self.assertIsNone(balanced_json_span('{"unterminated": 1'))

    def test_clean_xml_value_structure(self):
        self.assertEqual(clean_xml_value("\nBudapest\n"), "Budapest")
        self.assertEqual(clean_xml_value("\nline1\nline2\n"), "line1\nline2")

    def test_clean_xml_value_types(self):
        self.assertEqual(clean_xml_value("\n3\n"), 3)
        self.assertEqual(clean_xml_value("\n1.5\n"), 1.5)
        self.assertIs(clean_xml_value("\ntrue\n"), True)
        self.assertIsNone(clean_xml_value("\nnull\n"))
        self.assertEqual(clean_xml_value('\n{"a": 1}\n'), {"a": 1})
        self.assertEqual(clean_xml_value("\n[1, 2]\n"), [1, 2])
        self.assertEqual(clean_xml_value("\ntext\n"), "text")

    def test_decode_xml_parameters(self):
        prof = get_profile("qwen")
        body = (
            "<function=f>\n<parameter=a>\n1\n</parameter>\n"
            "<parameter=b>\nhello\n</parameter>\n</function>"
        )
        self.assertEqual(decode_xml_parameters(body, prof), {"a": 1, "b": "hello"})

    def test_decode_python_kwargs(self):
        self.assertEqual(
            decode_python_kwargs('city="Budapest", days=3, flag=True'),
            {"city": "Budapest", "days": 3, "flag": True},
        )


class TestQwenProfile(unittest.TestCase):
    def test_single_call(self):
        text = (
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nBudapest\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        got, parsed = calls(text, "qwen")
        self.assertEqual(got, [("get_weather", {"city": "Budapest"})])
        self.assertEqual(parsed.finish_reason, "tool_calls")
        self.assertEqual(parsed.content, "")

    def test_parallel_calls_and_content(self):
        text = (
            "Let me check the weather.\n"
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nBudapest\n</parameter>\n"
            "</function>\n</tool_call>\n"
            "<tool_call>\n<function=get_time>\n<parameter=city>\nVienna\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        got, parsed = calls(text, "qwen")
        self.assertEqual(
            got, [("get_weather", {"city": "Budapest"}), ("get_time", {"city": "Vienna"})]
        )
        self.assertEqual(parsed.content, "Let me check the weather.")

    def test_multiline_and_typed_values(self):
        text = (
            "<tool_call>\n<function=create_note>\n"
            "<parameter=title>\nGroceries\n</parameter>\n"
            "<parameter=body>\nmilk\neggs\n</parameter>\n"
            "<parameter=urgent>\ntrue\n</parameter>\n"
            "<parameter=count>\n2\n</parameter>\n"
            "<parameter=meta>\n{\"tags\": [\"a\"]}\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        got, _ = calls(text, "qwen")
        self.assertEqual(
            got,
            [
                (
                    "create_note",
                    {
                        "title": "Groceries",
                        "body": "milk\neggs",
                        "urgent": True,
                        "count": 2,
                        "meta": {"tags": ["a"]},
                    },
                )
            ],
        )

    def test_json_arguments_fallback(self):
        # some Qwen fine-tunes emit JSON inside <tool_call>
        text = '<tool_call>\n<function=get_weather>\n{"city": "Budapest"}\n</function>\n</tool_call>'
        got, _ = calls(text, "qwen")
        self.assertEqual(got, [("get_weather", {"city": "Budapest"})])

    def test_truncated_call_is_recovered(self):
        # generation cut off before </tool_call>
        text = (
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nBudapest\n</parameter>\n</function>"
        )
        got, parsed = calls(text, "qwen")
        self.assertEqual(got, [("get_weather", {"city": "Budapest"})])
        self.assertEqual(parsed.content, "")

    def test_reasoning_extraction(self):
        text = (
            "<think>\nI should call the tool.\n</think>\n\n"
            "<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"
        )
        got, parsed = calls(text, "qwen", reasoning_format="auto")
        self.assertEqual(got, [("f", {"x": 1})])
        self.assertEqual(parsed.reasoning_content, "I should call the tool.")

    def test_reasoning_inside_prefilled_think(self):
        # the generation prompt already opened <think>, so output starts inside it
        text = (
            "reasoning here\n</think>\n\n<tool_call>\n<function=f>\n"
            "<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"
        )
        got, parsed = calls(text, "qwen", reasoning_format="auto")
        self.assertEqual(got, [("f", {"x": 1})])
        self.assertEqual(parsed.reasoning_content, "reasoning here")

    def test_reasoning_format_none_keeps_tags(self):
        text = "<think>r</think>answer"
        _, parsed = calls(text, "qwen", reasoning_format="none")
        self.assertEqual(parsed.reasoning_content, "")
        self.assertEqual(parsed.content, "<think>r</think>answer")

    def test_no_tool_call_is_plain_content(self):
        _, parsed = calls("Just a normal answer.", "qwen")
        self.assertEqual(parsed.content, "Just a normal answer.")
        self.assertEqual(parsed.tool_calls, [])
        self.assertEqual(parsed.finish_reason, "stop")


class TestOtherProfiles(unittest.TestCase):
    def test_hermes(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Budapest"}}</tool_call>'
        got, _ = calls(text, "hermes")
        self.assertEqual(got, [("get_weather", {"city": "Budapest"})])

    def test_mistral(self):
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Bp"}, "id": "abc123"}]'
        parsed = parse_completion(text, profile="mistral")
        self.assertEqual(len(parsed.tool_calls), 1)
        call = parsed.tool_calls[0]
        self.assertEqual(call.name, "get_weather")
        self.assertEqual(call.arguments, {"city": "Bp"})
        self.assertEqual(call.id, "abc123")

    def test_mistral_two_calls(self):
        text = (
            '[TOOL_CALLS] [{"name": "a", "arguments": {}}, '
            '{"name": "b", "arguments": {"x": 1}}]'
        )
        got, _ = calls(text, "mistral")
        self.assertEqual(got, [("a", {}), ("b", {"x": 1})])

    def test_llama3(self):
        text = '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Bp"}}<|eom_id|>'
        got, _ = calls(text, "llama3")
        self.assertEqual(got, [("get_weather", {"city": "Bp"})])

    def test_deepseek(self):
        text = (
            "<｜DSML｜function_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">Budapest</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜function_calls>"
        )
        got, _ = calls(text, "deepseek")
        self.assertEqual(got, [("get_weather", {"city": "Budapest", "days": 3})])

    def test_gpt_oss(self):
        text = (
            "<|channel|>analysis<|message|>need weather<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.get_weather"
            '<|constrain|>json<|message|>{"city":"Budapest"}<|call|>'
        )
        parsed = parse_completion(text, profile="gpt-oss", reasoning_format="auto")
        self.assertEqual(
            [(c.name, c.arguments) for c in parsed.tool_calls],
            [("get_weather", {"city": "Budapest"})],
        )
        self.assertIn("need weather", parsed.reasoning_content)

    def test_kimi(self):
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.get_weather:0"
            '<|tool_call_argument_begin|>{"city": "Budapest"}<|tool_call_end|>'
            "<|tool_call_begin|>functions.get_time:1"
            '<|tool_call_argument_begin|>{"city": "Vienna"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        got, _ = calls(text, "kimi")
        self.assertEqual(got, [("get_weather", {"city": "Budapest"}), ("get_time", {"city": "Vienna"})])

    def test_lfm2(self):
        text = '<|tool_call_start|>[get_weather(city="Budapest", days=3)]<|tool_call_end|>'
        got, _ = calls(text, "lfm2")
        self.assertEqual(got, [("get_weather", {"city": "Budapest", "days": 3})])

    def test_lfm2_parallel(self):
        text = (
            "<|tool_call_start|>[get_weather(city=\"Budapest\"), get_time(city=\"Vienna\")]"
            "<|tool_call_end|>"
        )
        got, _ = calls(text, "lfm2")
        self.assertEqual(
            got, [("get_weather", {"city": "Budapest"}), ("get_time", {"city": "Vienna"})]
        )

    def test_ministral3(self):
        text = '[THINK]hmm[/THINK][TOOL_CALLS]get_weather[ARGS]{"city": "Budapest"}'
        got, parsed = calls(text, "ministral3")
        self.assertEqual(got, [("get_weather", {"city": "Budapest"})])
        self.assertEqual(parsed.reasoning_content, "hmm")

    def test_gemma4(self):
        text = '<|tool_call>call:get_weather{city:"Budapest"}<tool_call|>'
        got, _ = calls(text, "gemma4")
        self.assertEqual(got, [("get_weather", {"city": "Budapest"})])

    def test_plain_json(self):
        got, _ = calls('{"name": "f", "arguments": {"a": 1}}', "plain-json")
        self.assertEqual(got, [("f", {"a": 1})])


class TestLenientRecovery(unittest.TestCase):
    def test_fenced_json_array_fallback(self):
        text = (
            "```json\n[\n"
            '  {"tool": "get_weather", "arguments": {"city": "Budapest"}},\n'
            '  {"tool": "get_weather", "arguments": {"city": "Vienna"}}\n'
            "]\n```"
        )
        got, parsed = calls(text, "qwen")
        self.assertEqual(
            got, [("get_weather", {"city": "Budapest"}), ("get_weather", {"city": "Vienna"})]
        )
        self.assertEqual(parsed.content, "")

    def test_unparseable_call_goes_to_unparsed(self):
        text = "<tool_call>\n<function=f>\n<parameter=x>no close tag"
        parsed = parse_completion(text, profile="qwen")
        # lenient: kept as content/warnings rather than crashing
        self.assertIsInstance(parsed.warnings, list)

    def test_empty_call_block_is_ignored(self):
        # a <tool_call> block without a function name carries no call
        parsed = parse_completion("<tool_call>\n</tool_call>", profile="qwen")
        self.assertEqual(parsed.tool_calls, [])

    def test_call_without_parameters(self):
        parsed = parse_completion(
            "<tool_call>\n<function=reset>\n</function>\n</tool_call>", profile="qwen"
        )
        self.assertEqual([(c.name, c.arguments) for c in parsed.tool_calls], [("reset", {})])

    def test_strict_mode_raises_on_bad_arguments(self):
        from toolcaller.parser import _RawCall

        parser = SyntaxParser(get_profile("plain-json"), strict=True)
        parsed = parser.parse('{"name": "f", "arguments": {}}')
        self.assertEqual([(c.name, c.arguments) for c in parsed.tool_calls], [("f", {})])
        # undecodable arguments are reported, and strict mode turns them into errors
        with self.assertRaises(ValueError) as ctx:
            parser._build_call(_RawCall(name="f", argument_text="not json", raw="x"), 0)
        self.assertIn("JSON", str(ctx.exception))

    def test_wrapped_arguments(self):
        text = '<tool_call>{"name": "f", "arguments": "{\\"a\\": 1}"}</tool_call>'
        got, _ = calls(text, "hermes")
        self.assertEqual(got, [("f", {"a": 1})])

    def test_parameters_wrapper(self):
        text = '<tool_call>{"name": "f", "parameters": {"a": 1}}</tool_call>'
        got, _ = calls(text, "hermes")
        self.assertEqual(got, [("f", {"a": 1})])

    def test_tool_calls_wrapper(self):
        text = '{"tool_calls": [{"name": "f", "arguments": {"a": 1}}]}'
        got, _ = calls(text, "plain-json")
        self.assertEqual(got, [("f", {"a": 1})])


class TestParsedOutputShape(unittest.TestCase):
    def test_openai_message(self):
        parsed = parse_completion(
            "<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
            profile="qwen",
        )
        message = parsed.to_openai_message()
        self.assertEqual(message["role"], "assistant")
        self.assertIsNone(message["content"])
        call = message["tool_calls"][0]
        self.assertEqual(call["type"], "function")
        self.assertTrue(call["id"].startswith("call_"))
        self.assertEqual(json.loads(call["function"]["arguments"]), {"x": 1})

    def test_content_message_has_no_null(self):
        parsed = parse_completion("hello", profile="qwen")
        message = parsed.to_openai_message()
        self.assertEqual(message["content"], "hello")
        self.assertNotIn("tool_calls", message)

    def test_ids_unique(self):
        text = (
            "<tool_call>\n<function=a>\n</function>\n</tool_call>"
            "<tool_call>\n<function=b>\n</function>\n</tool_call>"
        )
        parsed = parse_completion(text, profile="qwen")
        ids = [c.id for c in parsed.tool_calls]
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(set(ids)), 2)


class TestReasoningSplitting(unittest.TestCase):
    def test_auto_without_tags(self):
        reasoning, content = split_reasoning("plain", get_profile("qwen"), "auto")
        self.assertEqual((reasoning, content), ("", "plain"))

    def test_deepseek_forces_extraction(self):
        reasoning, content = split_reasoning("all thinking", get_profile("qwen"), "deepseek")
        self.assertEqual(reasoning, "all thinking")
        self.assertEqual(content, "")

    def test_legacy_keeps_tags(self):
        reasoning, content = split_reasoning(
            "<think>r</think>c", get_profile("qwen"), "deepseek-legacy"
        )
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "<think>r</think>c")

    def test_profile_without_tags(self):
        reasoning, content = split_reasoning("<think>r</think>c", get_profile("mistral"), "auto")
        self.assertEqual((reasoning, content), ("", "<think>r</think>c"))

    def test_invalid_format(self):
        with self.assertRaises(ParseError):
            parse_completion("x", profile="qwen", reasoning_format="nonsense")


class TestProfileResolution(unittest.TestCase):
    def test_builtin(self):
        profile, _ = resolve_profile(ParserSelection(profile="qwen"))
        self.assertEqual(profile.name, "qwen")

    def test_alias(self):
        profile, _ = resolve_profile(ParserSelection(profile="qwen3"))
        self.assertEqual(profile.name, "qwen")

    def test_auto_from_template(self):
        profile, reason = resolve_profile(
            ParserSelection(auto_profile=True), "<tool_call><function=x><parameter=y>"
        )
        self.assertEqual(profile.name, "qwen")
        self.assertIn("auto", reason)

    def test_default(self):
        profile, _ = resolve_profile(ParserSelection(), None)
        self.assertEqual(profile.name, "qwen")

    def test_unknown_profile(self):
        from toolcaller.errors import ConfigError

        with self.assertRaises(ConfigError):
            resolve_profile(ParserSelection(profile="nope"))


class TestCustomProfile(unittest.TestCase):
    def test_custom_profile_roundtrip(self):
        profile = SyntaxProfile(
            name="mine",
            call_start="<<CALL>>",
            call_end="<</CALL>>",
            function_tag_open="NAME=",
            function_name_terminator="|",
            function_tag_close="",
            argument_style="xml_params",
            parameter_tag_open="ARG=",
            reasoning_start="<REASON>",
            reasoning_end="</REASON>",
        )
        text = "<<CALL>>\nNAME=get_x|<ARG=a>\n1\n</parameter>\n<</CALL>>"
        parsed = parse_completion(text, profile=profile)
        self.assertEqual(
            [(c.name, c.arguments) for c in parsed.tool_calls], [("get_x", {"a": 1})]
        )

    def test_custom_profile_json_roundtrip(self):
        profile = get_profile("qwen")
        clone = SyntaxProfile.from_dict(profile.to_dict())
        self.assertEqual(clone.name, profile.name)
        self.assertEqual(clone.argument_style, profile.argument_style)
        self.assertEqual(list(clone.extra_stop_strings), list(profile.extra_stop_strings))

    def test_invalid_argument_style(self):
        with self.assertRaises(ValueError):
            SyntaxProfile(name="bad", argument_style="nope")

    def test_unknown_keys_rejected(self):
        with self.assertRaises(ValueError):
            SyntaxProfile.from_dict({"name": "x", "bogus": 1})


if __name__ == "__main__":
    unittest.main()


class TestSparkProfile(unittest.TestCase):
    """Spark X2.6: <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>"""

    @classmethod
    def setUpClass(cls):
        cls.parser = SyntaxParser(get_profile("spark"), strict=False)

    def _parse(self, text):
        return self.parser.parse(text).tool_calls

    def test_single_arg(self):
        calls = self._parse("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Budapest</arg_value></tool_call>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].arguments, {"city": "Budapest"})

    def test_multiple_args(self):
        calls = self._parse(
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Budapest</arg_value>"
            "<arg_key>unit</arg_key><arg_value>celsius</arg_value></tool_call>"
        )
        self.assertEqual(calls[0].arguments, {"city": "Budapest", "unit": "celsius"})

    def test_typed_values(self):
        calls = self._parse(
            "<tool_call>search<arg_key>q</arg_key><arg_value>hello</arg_value>"
            "<arg_key>count</arg_key><arg_value>10</arg_value>"
            "<arg_key>debug</arg_key><arg_value>true</arg_value></tool_call>"
        )
        self.assertEqual(calls[0].arguments["count"], 10)
        self.assertEqual(calls[0].arguments["debug"], True)

    def test_parallel_calls(self):
        calls = self._parse(
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Budapest</arg_value></tool_call>"
            "<tool_call>get_time<arg_key>city</arg_key><arg_value>Vienna</arg_value></tool_call>"
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[1].name, "get_time")

    def test_truncated_call(self):
        calls = self._parse("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Budapest</arg_value>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments, {"city": "Budapest"})

    def test_with_preamble(self):
        calls = self._parse(
            "Here is the call:\n"
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Budapest</arg_value></tool_call>"
        )
        self.assertEqual(len(calls), 1)

    def test_profile_auto_detect(self):
        from toolcaller.profiles import detect_profile_from_template
        self.assertEqual(
            detect_profile_from_template("<arg_key>city</arg_key><arg_value>"),
            "spark",
        )


class TestGlmProfile(unittest.TestCase):
    """GLM shares Spark's tool syntax but has its own turn delimiters."""

    def setUp(self):
        self.parser = SyntaxParser(get_profile("glm"), strict=False)

    def test_tool_call_syntax_is_shared_with_spark(self):
        calls = self.parser.parse(
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>Budapest</arg_value></tool_call>"
        ).tool_calls
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].arguments, {"city": "Budapest"})

    def test_leading_close_reasoning(self):
        # The template prefills "<think>", so the model emits "</think>" first.
        parsed = self.parser.parse("</think>Hi there!", reasoning_format="auto")
        self.assertEqual(parsed.reasoning_content, "")
        self.assertEqual(parsed.content, "Hi there!")

    def test_turn_delimiter_never_leaks_into_content(self):
        parsed = self.parser.parse(
            "reasoning</think>Hi there!<|user|>", reasoning_format="auto"
        )
        self.assertEqual(parsed.reasoning_content, "reasoning")
        self.assertEqual(parsed.content, "Hi there!")

    def test_detected_from_template(self):
        from toolcaller.profiles import detect_profile_from_template

        self.assertEqual(detect_profile_from_template("[gMASK]<sop>..."), "glm")
        # Spark templates must still be detected as spark.
        self.assertEqual(
            detect_profile_from_template("<arg_key>k</arg_key><arg_value>"),
            "spark",
        )


class TestTurnStops(unittest.TestCase):
    """Turn delimiters are stops even without tools."""

    def test_glm_turn_stops_without_tools(self):
        from toolcaller.constraints import build_constraints

        constraints = build_constraints(
            get_profile("glm"), eos_token="<|endoftext|>", has_tools=False
        )
        self.assertIn("<|user|>", constraints.stops)
        self.assertIn("<|observation|>", constraints.stops)

    def test_glm_turn_stops_with_tools(self):
        from toolcaller.constraints import build_constraints

        constraints = build_constraints(
            get_profile("glm"), eos_token="<|endoftext|>", has_tools=True
        )
        for stop in ("<|user|>", "<|observation|>", "<|system|>"):
            self.assertIn(stop, constraints.stops)

    def test_profiles_without_turn_stops_are_unchanged(self):
        from toolcaller.constraints import build_constraints

        constraints = build_constraints(
            get_profile("spark"), eos_token="<|endoftext|>", has_tools=False
        )
        self.assertEqual(constraints.stops, ["<|endoftext|>"])
