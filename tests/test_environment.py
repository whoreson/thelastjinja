"""Unit tests: template environment and capability detection."""

from __future__ import annotations

import os
import unittest

from toolcaller.caps import analyze_caps
from toolcaller.environment import TemplateEnvironment
from toolcaller.errors import TemplateError

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_TEMPLATE = os.path.join(HERE, "qwen.jinja")

# A tiny template that has tools + system + tool calls, ChatML style.
TOOL_TEMPLATE = """
{%- for message in messages -%}
  {%- if message.role == 'system' -%}
    {{- '[SYS]' + message.content + '[/SYS]' -}}
  {%- elif message.role == 'user' -%}
    {{- '[USER]' + message.content + '[/USER]' -}}
  {%- elif message.role == 'assistant' -%}
    {{- '[ASSISTANT]' + (message.content or '') -}}
    {%- if message.tool_calls -%}
      {%- for tc in message.tool_calls -%}
        {{- '[CALL]' + tc.function.name + '|' + tc.function.arguments|string + '[/CALL]' -}}
      {%- endfor -%}
    {%- endif -%}
  {%- elif message.role == 'tool' -%}
    {{- '[TOOL]' + message.content + '[/TOOL]' -}}
  {%- endif -%}
{%- endfor -%}
{%- if tools -%}
  {%- for tool in tools -%}
    {{- '[TOOLDEF]' + tool.function.name + '[/TOOLDEF]' -}}
  {%- endfor -%}
{%- endif -%}
{%- if add_generation_prompt -%}{{- '[ASSISTANT]' -}}{%- endif -%}
"""

# Object-arguments template (iterates arguments).
OBJ_ARGS_TEMPLATE = """
{%- for message in messages -%}
{{- message.role }}:{{ message.content -}}
{%- if message.tool_calls -%}
{%- for tc in message.tool_calls -%}
|{{ tc.function.name }}({% for k, v in tc.function.arguments|items %}{{ k }}={{ v }};{% endfor %})
{%- endfor -%}
{%- endif -%}
{%- endfor -%}
"""

NO_SYSTEM_TEMPLATE = """
{%- for message in messages -%}
  {%- if message.role != 'system' -%}
    {{- message.role + ':' + message.content + '\\n' -}}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}assistant:{%- endif -%}
"""


def env(source: str) -> TemplateEnvironment:
    return TemplateEnvironment(source)


class TestEnvironmentBasics(unittest.TestCase):
    def test_raise_exception_global(self):
        e = env("{% if true %}{{ raise_exception('boom') }}{% endif %}")
        with self.assertRaises(TemplateError) as ctx:
            e.render({})
        self.assertIn("boom", str(ctx.exception))

    def test_tojson_is_compact_and_unescaped(self):
        e = env("{{ v|tojson }}")
        out = e.render({"v": {"a": "x<y>", "b": 1}})
        # nlohmann::json::dump() style: ", " and ": ", no HTML escaping
        self.assertEqual(out, '{"a": "x<y>", "b": 1}')

    def test_undefined_renders_empty(self):
        e = env("a{{ missing }}b")
        self.assertEqual(e.render({}), "ab")

    def test_default_filter_works(self):
        e = env("{{ missing|default('x') }}")
        self.assertEqual(e.render({}), "x")

    def test_no_trim_blocks_by_default(self):
        e = env("{% if true %}\nX\n{% endif %}\n")
        self.assertIn("\nX\n", e.render({}))

    def test_strftime_now(self):
        e = env('{{ strftime_now("%Y") }}')
        self.assertEqual(len(e.render({})), 4)

    def test_namespace(self):
        e = env("{% set ns = namespace(n=0) %}{% for i in range(3) %}{% set ns.n = ns.n + i %}{% endfor %}{{ ns.n }}")
        self.assertEqual(e.render({}), "3")

    def test_syntax_error_reported(self):
        with self.assertRaises(TemplateError):
            env("{% if x %}")

    def test_empty_template_rejected(self):
        with self.assertRaises(TemplateError):
            env("   ")

    def test_do_extension(self):
        e = env("{% set xs = [] %}{% do xs.append(1) %}{{ xs }}")
        self.assertEqual(e.render({}), "[1]")


class TestGlobalUse(unittest.TestCase):
    def test_detects_globals(self):
        used = env(TOOL_TEMPLATE).used_globals()
        self.assertIn("messages", used)
        self.assertIn("tools", used)
        self.assertIn("add_generation_prompt", used)

    def test_assigned_names_are_not_globals(self):
        e = env("{% set tmp = 1 %}{{ tmp }}")
        self.assertNotIn("tmp", e.used_globals())

    def test_loop_vars_excluded(self):
        e = env("{% for x in messages %}{{ x.role }}{% endfor %}")
        self.assertNotIn("x", e.used_globals())

    def test_attribute_name_not_counted(self):
        e = env("{% for t in tools %}{{ t.function }}{% endfor %}")
        used = e.used_globals()
        # `tools` is a global; `t` is a loop target and `function` an attribute
        self.assertIn("tools", used)
        self.assertNotIn("t", used)
        self.assertNotIn("function", used)

    def test_qwen_template_globals(self):
        with open(QWEN_TEMPLATE, encoding="utf-8") as fh:
            used = TemplateEnvironment(fh.read()).used_globals()
        for name in ("messages", "tools", "enable_thinking", "raise_exception"):
            self.assertIn(name, used)


class TestCaps(unittest.TestCase):
    def test_tool_template_caps(self):
        caps = analyze_caps(env(TOOL_TEMPLATE))
        self.assertTrue(caps.supports_tools)
        self.assertTrue(caps.supports_tool_calls)
        self.assertTrue(caps.supports_system_role)
        # template prints arguments as a string
        self.assertFalse(caps.supports_object_arguments)

    def test_object_arguments_detected(self):
        caps = analyze_caps(env(OBJ_ARGS_TEMPLATE))
        self.assertTrue(caps.supports_object_arguments)

    def test_system_role_not_supported(self):
        caps = analyze_caps(env(NO_SYSTEM_TEMPLATE))
        self.assertFalse(caps.supports_system_role)

    def test_template_without_tools(self):
        caps = analyze_caps(env("{% for m in messages %}{{ m.content }}{% endfor %}"))
        self.assertFalse(caps.supports_tools)
        self.assertFalse(caps.supports_tool_calls)

    def test_qwen_caps(self):
        with open(QWEN_TEMPLATE, encoding="utf-8") as fh:
            caps = analyze_caps(TemplateEnvironment(fh.read()))
        self.assertTrue(caps.supports_tools)
        self.assertTrue(caps.supports_tool_calls)
        self.assertTrue(caps.supports_parallel_tool_calls)
        self.assertTrue(caps.supports_system_role)
        self.assertTrue(caps.supports_object_arguments)
        self.assertTrue(caps.supports_enable_thinking)

    def test_caps_cached(self):
        e = env(TOOL_TEMPLATE)
        self.assertIs(analyze_caps(e), analyze_caps(e))

    def test_caps_to_map_keys(self):
        caps = analyze_caps(env(TOOL_TEMPLATE))
        self.assertEqual(
            set(caps.to_map()),
            {
                "supports_string_content",
                "supports_typed_content",
                "supports_tools",
                "supports_tool_calls",
                "supports_parallel_tool_calls",
                "supports_system_role",
                "supports_preserve_reasoning",
                "supports_object_arguments",
            },
        )


class TestBakes(unittest.TestCase):
    def test_baked_variable(self):
        e = env("{% set enable_thinking = true %}{{ enable_thinking }}")
        self.assertTrue(e.bakes("enable_thinking"))

    def test_not_baked(self):
        e = env("{% if enable_thinking %}x{% endif %}")
        self.assertFalse(e.bakes("enable_thinking"))


if __name__ == "__main__":
    unittest.main()
