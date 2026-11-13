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
QWEN_TEMPLATE = os.path.join(HERE, "qwen.jinja")
SPARK_TEMPLATE = os.path.join(HERE, "spark.jinja")
GLM_TEMPLATE = os.path.join(HERE, "glm.jinja")

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


class TestReasoningEffort(unittest.TestCase):
    """reasoning_effort in request maps to enable_thinking / reasoning_effort context."""

    def _make_renderer(self):
        from toolcaller.renderer import PromptRenderer
        from toolcaller.environment import load_template_file

        env = load_template_file(SPARK_TEMPLATE)
        return PromptRenderer(
            env=env,
            bos_token="<｜start▁of▁sentence｜>",
            eos_token="<｜end▁of▁sentence｜>",
            strict=True,
        )

    def test_default_thinking_on(self):
        r = self._make_renderer()
        ctx = r._build_context([], [], request={"messages": []}, template_kwargs=None, now=None)
        self.assertTrue(ctx["enable_thinking"])

    def test_reasoning_effort_none(self):
        r = self._make_renderer()
        ctx = r._build_context([], [], request={"messages": [], "reasoning_effort": "none"}, template_kwargs=None, now=None)
        self.assertFalse(ctx["enable_thinking"])

    def test_reasoning_effort_low(self):
        r = self._make_renderer()
        ctx = r._build_context([], [], request={"messages": [], "reasoning_effort": "low"}, template_kwargs=None, now=None)
        self.assertTrue(ctx["enable_thinking"])
        self.assertEqual(ctx["reasoning_effort"], "low")

    def test_reasoning_effort_high(self):
        r = self._make_renderer()
        ctx = r._build_context([], [], request={"messages": [], "reasoning_effort": "high"}, template_kwargs=None, now=None)
        self.assertTrue(ctx["enable_thinking"])
        self.assertEqual(ctx["reasoning_effort"], "high")

    def test_template_kwargs_override(self):
        r = self._make_renderer()
        # template_kwargs can force enable_thinking off even with default reasoning_effort
        ctx = r._build_context(
            [], [],
            request={"messages": []},
            template_kwargs={"enable_thinking": False},
            now=None,
        )
        self.assertFalse(ctx["enable_thinking"])


class TestResolveThinking(unittest.TestCase):
    """Every client convention that disables thinking must be honoured.

    Pi sends nothing at all for the plain OpenAI-compatible format, and sends
    ``reasoning_effort`` / ``enable_thinking`` / ``chat_template_kwargs`` for
    the qwen formats - so all of these have to map to the template variable.
    """

    def test_absent_means_default_on(self):
        from toolcaller.renderer import resolve_thinking

        self.assertEqual(resolve_thinking({}), (True, None))
        self.assertEqual(resolve_thinking(None), (True, None))
        self.assertEqual(resolve_thinking({"messages": []}), (True, None))

    def test_reasoning_effort_values(self):
        from toolcaller.renderer import resolve_thinking

        self.assertEqual(resolve_thinking({"reasoning_effort": "none"}), (False, None))
        self.assertEqual(resolve_thinking({"reasoning_effort": "off"}), (False, None))
        self.assertEqual(resolve_thinking({"reasoning_effort": "minimal"}), (False, None))
        self.assertEqual(resolve_thinking({"reasoning_effort": None}), (False, None))
        self.assertEqual(resolve_thinking({"reasoning_effort": "low"}), (True, "low"))
        self.assertEqual(resolve_thinking({"reasoning_effort": "medium"}), (True, "high"))
        self.assertEqual(resolve_thinking({"reasoning_effort": "high"}), (True, "high"))

    def test_enable_thinking_flag(self):
        from toolcaller.renderer import resolve_thinking

        self.assertEqual(resolve_thinking({"enable_thinking": False}), (False, None))
        self.assertEqual(resolve_thinking({"enable_thinking": True}), (True, None))
        self.assertEqual(
            resolve_thinking({"enable_thinking": "false"}), (False, None)
        )

    def test_chat_template_kwargs(self):
        from toolcaller.renderer import resolve_thinking

        self.assertEqual(
            resolve_thinking({"chat_template_kwargs": {"enable_thinking": False}}),
            (False, None),
        )

    def test_nested_thinking_objects(self):
        from toolcaller.renderer import resolve_thinking

        self.assertEqual(
            resolve_thinking({"thinking": {"type": "disabled"}}), (False, None)
        )
        self.assertEqual(
            resolve_thinking({"thinking": {"type": "enabled"}}), (True, None)
        )
        self.assertEqual(
            resolve_thinking({"reasoning": {"enabled": False}}), (False, None)
        )
        self.assertEqual(
            resolve_thinking({"reasoning": {"max_tokens": 0}}), (False, None)
        )
        self.assertEqual(resolve_thinking({"reasoning_enabled": False}), (False, None))

    def test_off_signal_changes_rendered_prompt(self):
        from toolcaller.renderer import PromptRenderer
        from toolcaller.environment import load_template_file

        env = load_template_file(SPARK_TEMPLATE)
        r = PromptRenderer(
            env=env,
            bos_token="<｜start▁of▁sentence｜>",
            eos_token="<｜end▁of▁sentence｜>",
            strict=True,
        )
        on = r.render(
            {"messages": [{"role": "user", "content": "hi"}]},
            template_kwargs=None,
        )
        off = r.render(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "none",
            },
            template_kwargs=None,
        )
        self.assertTrue(on.prompt.rstrip().endswith("<think>"))
        self.assertFalse(off.prompt.rstrip().endswith("<think>"))


class TestReasoningEffortWithoutEnableThinking(unittest.TestCase):
    """GLM has no enable_thinking switch but honours reasoning_effort levels.

    The effort must be forwarded even though the template cannot disable
    thinking, otherwise every request falls back to the template default.
    """

    def _renderer(self):
        from toolcaller.renderer import PromptRenderer
        from toolcaller.environment import load_template_file

        env = load_template_file(GLM_TEMPLATE)
        return PromptRenderer(
            env=env,
            bos_token="[gMASK]",
            eos_token="<|endoftext|>",
            strict=True,
        )

    def _effort_line(self, request):
        render = self._renderer().render(request, template_kwargs=None)
        import re

        match = re.search(r"Reasoning Effort: (\w+)", render.prompt)
        return match.group(1) if match else None

    def test_low_is_forwarded(self):
        self.assertEqual(
            self._effort_line(
                {"messages": [{"role": "user", "content": "hi"}],
                 "reasoning_effort": "low"}
            ),
            "Low",
        )

    def test_high_is_forwarded(self):
        self.assertEqual(
            self._effort_line(
                {"messages": [{"role": "user", "content": "hi"}],
                 "reasoning_effort": "high"}
            ),
            "High",
        )

    def test_default_when_absent(self):
        self.assertEqual(
            self._effort_line({"messages": [{"role": "user", "content": "hi"}]}),
            "Max",
        )
