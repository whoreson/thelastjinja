"""Unit tests: prompt rendering / llama.cpp workarounds / generation prompt."""

from __future__ import annotations

import os
import unittest

from toolcaller.environment import TemplateEnvironment
from toolcaller.errors import RequestError
from toolcaller.messages import normalize_request
from toolcaller.renderer import (
    PromptRenderer,
    _split_generation_prompt,
    map_developer_role_to_system,
    system_message_not_supported,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_TEMPLATE = os.path.join(HERE, "chat_template.jinja")

CHATML = (
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
)
CHATML_TOOLS = (
    "{%- if tools -%}{{- '<TOOLS>' -}}"
    "{%- for t in tools -%}{{- t.function.name -}}{%- endfor -%}"
    "{{- '</TOOLS>' -}}{%- endif -%}"
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message.role + '\n' + (message.content or '') -}}"
    "{%- if message.tool_calls -%}"
    "{%- for tc in message.tool_calls -%}"
    "{{- '<C>' + tc.function.name + ':' + tc.function.arguments|string + '</C>' -}}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
)


def renderer(source: str, **kwargs) -> PromptRenderer:
    return PromptRenderer(TemplateEnvironment(source), **kwargs)


def req(*messages, **extra):
    payload = {"messages": list(messages)}
    payload.update(extra)
    return normalize_request(payload)


class TestBasicRender(unittest.TestCase):
    def test_chatml(self):
        r = renderer(CHATML, eos_token="<|im_end|>")
        result = r.render(req({"role": "user", "content": "hi"}))
        self.assertEqual(
            result.prompt,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
        )
        self.assertEqual(result.generation_prompt, "<|im_start|>assistant\n")

    def test_generation_prompt_is_suffix(self):
        r = renderer(CHATML)
        result = r.render(req({"role": "user", "content": "hi"}))
        self.assertTrue(result.prompt.endswith(result.generation_prompt))
        self.assertEqual(result.prompt_for_parser + result.generation_prompt, result.prompt)

    def test_no_bos_duplication(self):
        r = renderer(CHATML, bos_token="<s>", add_bos=True)
        result = r.render(req({"role": "user", "content": "hi"}))
        self.assertEqual(result.prompt.count("<s>"), 1)
        self.assertTrue(result.prompt.startswith("<s>"))

    def test_bos_added_when_missing(self):
        r = renderer(CHATML, bos_token="<s>", add_bos=True)
        r2 = renderer("<s>" + CHATML, bos_token="<s>", add_bos=True)
        self.assertEqual(
            r.render(req({"role": "user", "content": "hi"})).prompt,
            r2.render(req({"role": "user", "content": "hi"})).prompt,
        )

    def test_bos_not_added_when_disabled(self):
        r = renderer(CHATML, bos_token="<s>", add_bos=False)
        self.assertFalse(r.render(req({"role": "user", "content": "hi"})).prompt.startswith("<s>"))

    def test_empty_messages_rejected(self):
        r = renderer(CHATML)
        with self.assertRaises(RequestError):
            r.render({"messages": []})

    def test_template_kwargs(self):
        tpl = "{% if enable_thinking %}THINK{% else %}NOTHINK{% endif %}{{ messages[0].content }}"
        r = renderer(tpl, default_template_kwargs={"enable_thinking": False})
        self.assertEqual(r.render(req({"role": "user", "content": "x"})).prompt, "NOTHINKx")


class TestWorkarounds(unittest.TestCase):
    def test_developer_mapped_to_system(self):
        out = map_developer_role_to_system([{"role": "developer", "content": "d"}])
        self.assertEqual(out[0]["role"], "system")

    def test_system_folded_into_user(self):
        msgs = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
        ]
        out = system_message_not_supported(msgs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "user")
        self.assertIn("SYS", out[0]["content"])
        self.assertIn("hello", out[0]["content"])

    def test_system_folded_into_parts(self):
        msgs = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        out = system_message_not_supported(msgs)
        self.assertEqual(out[0]["content"][0], {"type": "text", "text": "SYS"})
        self.assertEqual(out[0]["content"][1]["text"], "hello")

    def test_render_maps_developer(self):
        r = renderer(CHATML)
        out = r.render(req({"role": "developer", "content": "dev"}, {"role": "user", "content": "u"}))
        self.assertIn("<|im_start|>system\ndev", out.prompt)

    def test_null_content_becomes_empty_string(self):
        r = renderer(CHATML_TOOLS)
        out = r.render(
            req(
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"type": "function", "function": {"name": "f", "arguments": "{}"}}
                    ],
                },
            )
        )
        self.assertIn("<|im_start|>assistant\n<C>f:{}</C>", out.prompt)
        self.assertNotIn("None", out.prompt)

    def test_arguments_as_object_when_template_iterates(self):
        tpl = (
            "{%- for m in messages -%}{%- if m.tool_calls -%}"
            "{%- for tc in m.tool_calls -%}"
            "{%- for k, v in tc.function.arguments|items -%}{{ k }}={{ v }};{%- endfor -%}"
            "{%- endfor -%}{%- endif -%}{%- endfor -%}"
        )
        r = renderer(tpl)
        out = r.render(
            req(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"city":"Budapest"}'},
                        }
                    ],
                }
            )
        )
        self.assertEqual(out.prompt, "city=Budapest;")

    def test_tools_rendered(self):
        r = renderer(CHATML_TOOLS)
        out = r.render(
            req(
                {"role": "user", "content": "q"},
                tools=[
                    {"type": "function", "function": {"name": "alpha", "parameters": {}}},
                    {"type": "function", "function": {"name": "beta", "parameters": {}}},
                ],
            )
        )
        self.assertIn("<TOOLS>alphabeta</TOOLS>", out.prompt)

    def test_tools_rejected_when_template_ignores_them(self):
        r = renderer(CHATML)
        with self.assertRaises(RequestError):
            r.render(
                req(
                    {"role": "user", "content": "q"},
                    tools=[{"type": "function", "function": {"name": "f"}}],
                )
            )

    def test_tools_dropped_when_not_strict(self):
        r = PromptRenderer(TemplateEnvironment(CHATML), strict=False)
        out = r.render(
            req(
                {"role": "user", "content": "q"},
                tools=[{"type": "function", "function": {"name": "f"}}],
            )
        )
        self.assertNotIn("f", out.prompt)

    def test_tool_parameters_default_to_object_schema(self):
        tpl = "{% for t in tools %}{{ t.function.parameters|tojson }}{% endfor %}"
        r = renderer(tpl)
        out = r.render(
            req(
                {"role": "user", "content": "q"},
                tools=[{"type": "function", "function": {"name": "f"}}],
            )
        )
        self.assertEqual(out.prompt, '{"type": "object", "properties": {}}')

    def test_bad_tool_type(self):
        r = renderer(CHATML_TOOLS)
        with self.assertRaises(RequestError):
            r.render(
                req(
                    {"role": "user", "content": "q"},
                    tools=[{"type": "web_search", "function": {"name": "f"}}],
                )
            )


class TestGenerationPromptDiff(unittest.TestCase):
    def test_split_suffix(self):
        self.assertEqual(_split_generation_prompt("abcXYZ", "abc"), "XYZ")

    def test_split_identical(self):
        self.assertEqual(_split_generation_prompt("abc", "abc"), "")

    def test_split_fallback_common_prefix(self):
        # not a suffix and not a common-prefix extension -> no generation prompt
        from toolcaller.renderer import GenerationPromptError

        with self.assertRaises(GenerationPromptError):
            _split_generation_prompt("prefixA", "prefixB")

    def test_split_common_prefix_extension(self):
        self.assertEqual(_split_generation_prompt("abcX", "abc"), "X")

    def test_varying_template_raises(self):
        # a template whose add_generation_prompt output diverges in the middle
        tpl = "{% if add_generation_prompt %}A{% endif %}x{% if not add_generation_prompt %}B{% endif %}"
        r = renderer(tpl)
        out = r.render(req({"role": "user", "content": "q"}))
        self.assertEqual(out.generation_prompt, "")
        self.assertTrue(out.warnings)

    def test_varying_template_tolerated_when_not_strict(self):
        tpl = "{% if add_generation_prompt %}A{% endif %}x{% if not add_generation_prompt %}B{% endif %}"
        r = PromptRenderer(TemplateEnvironment(tpl), strict=False)
        out = r.render(req({"role": "user", "content": "q"}))
        self.assertEqual(out.generation_prompt, "")
        self.assertTrue(out.warnings)

    def test_no_diff_when_disabled(self):
        r = renderer(CHATML)
        out = r.render(req({"role": "user", "content": "q"}), generation_prompt_diff=False)
        self.assertEqual(out.generation_prompt, "")


class TestQwenTemplate(unittest.TestCase):
    def setUp(self):
        with open(QWEN_TEMPLATE, encoding="utf-8") as fh:
            self.renderer = PromptRenderer(
                TemplateEnvironment(fh.read()),
                bos_token="<|endoftext|>",
                eos_token="<|im_end|>",
            )

    def test_generation_prompt_has_think(self):
        out = self.renderer.render(req({"role": "user", "content": "hi"}))
        self.assertEqual(out.generation_prompt, "<|im_start|>assistant\n<think>\n")

    def test_system_prompt(self):
        out = self.renderer.render(
            req({"role": "system", "content": "S"}, {"role": "user", "content": "hi"})
        )
        self.assertTrue(out.prompt.startswith("<|im_start|>system\nS<|im_end|>\n"))

    def test_tools_block(self):
        out = self.renderer.render(
            req(
                {"role": "user", "content": "hi"},
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "d",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )
        )
        self.assertIn("# Tools", out.prompt)
        self.assertIn('"name": "get_weather"', out.prompt)
        self.assertIn("<tools>", out.prompt)

    def test_past_tool_call_rendering(self):
        out = self.renderer.render(
            req(
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Budapest"}'},
                        }
                    ],
                },
                {"role": "tool", "content": "sunny", "tool_call_id": "c1"},
            )
        )
        self.assertIn("<function=get_weather>", out.prompt)
        self.assertIn("<parameter=city>", out.prompt)
        self.assertIn("<tool_response>\nsunny\n</tool_response>", out.prompt)

    def test_thinking_disabled(self):
        renderer = PromptRenderer(
            self.renderer.env,
            bos_token="<|endoftext|>",
            eos_token="<|im_end|>",
            default_template_kwargs={"enable_thinking": False},
        )
        out = renderer.render(req({"role": "user", "content": "hi"}))
        self.assertEqual(
            out.generation_prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def test_scalar_arguments_render_as_string(self):
        out = self.renderer.render(
            req(
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "f",
                                "arguments": '{"n": 3, "flag": true, "nested": {"a": [1, 2]}}',
                            },
                        }
                    ],
                },
            )
        )
        # scalars are emitted bare, objects/arrays as JSON
        self.assertIn("<parameter=n>\n3\n</parameter>", out.prompt)
        self.assertIn("<parameter=flag>\ntrue\n</parameter>", out.prompt)
        self.assertIn('<parameter=nested>\n{"a": [1, 2]}\n</parameter>', out.prompt)


if __name__ == "__main__":
    unittest.main()
