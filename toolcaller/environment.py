"""Jinja2 environment hardened to behave like ``ik_llama.cpp``'s Minja.

Differences from a stock ``jinja2.Environment`` that matter for chat
templates:

* ``raise_exception(msg)`` is available as a *global* (Minja built-in,
  HuggingFace transformers too). Stock jinja2 only has the ``|raise_exception``
  filter, which renders to nothing.
* ``strftime_now(fmt)`` is available as a global (transformers extension).
* ``tojson`` mirrors ``nlohmann::json::dump()``: compact separators,
  ``ensure_ascii=False``.  Minja's ``tojson`` does **not** escape ``<``,
  ``>``, ``&`` or ``'`` (stock jinja2 escapes ``<``/``>``/``&``/``'`` into
  ``\\u003c`` etc.).  Producing HTML-escaped JSON inside a prompt changes the
  bytes the model sees, so we override it.
* ``trim_blocks``/``lstrip_blocks``/``keep_trailing_newline`` are disabled:
  HuggingFace renders templates exactly as written.
* Unknown globals used by exotic templates can be supplied by the caller.

We deliberately do **not** use ``jinja2.sandbox``: the template is trusted
local configuration, and sandboxing would break ``namespace()`` objects and
several macros used by real chat templates.
"""

from __future__ import annotations

import datetime as _datetime
import json
from typing import Any, Dict, Mapping, Optional

import jinja2
from jinja2 import nodes

from .errors import TemplateError


class _RaisedJinjaError(TemplateError):
    """Raised by the ``raise_exception()`` global.

    Subclassing ``TemplateError`` lets callers catch template problems with a
    single ``except``; the original message is kept verbatim.
    """


def _tojson_filter(value: Any, indent: Optional[int] = None,
                   ensure_ascii: bool = False, **kwargs: Any) -> str:
    """``nlohmann::json::dump()``-compatible ``|tojson``."""
    if indent is not None and indent >= 0:
        text = json.dumps(value, indent=indent, ensure_ascii=ensure_ascii)
        # nlohmann pretty-prints object *and* array separators with ", " / ": "
        return text
    return json.dumps(value, separators=(", ", ": "), ensure_ascii=ensure_ascii)


def _tojson_global(value: Any = None, **kwargs: Any) -> str:
    return _tojson_filter(value, **kwargs)


def _raise_exception(message: Any = "") -> str:
    raise _RaisedJinjaError(str(message))


def _strftime_now(fmt: str) -> str:
    return _datetime.datetime.now().strftime(fmt)


class GlobalUseAnalyzer:
    """Find which *global* variables a template reads.

    Used by :mod:`toolcaller.caps` to decide whether a template supports
    tools, object arguments, system roles, ... exactly like Minja's
    ``stats.used`` bookkeeping.

    A global is considered "used" when a ``{{ name }}`` / ``{{ name.attr }}``
    / ``{{ name | filter }}`` expression loads it, or when it is passed to
    ``namespace()``/``dict()``-style calls, or compared with ``is defined``.
    """

    #: Names that are always considered used (loop/builtin context).
    ALWAYS_USED = frozenset({"loop", "true", "false", "none", "True", "False", "None"})

    def __init__(self, environment: jinja2.Environment) -> None:
        self.environment = environment
        self._cache: Dict[int, frozenset] = {}

    def used_names(self, source: str) -> frozenset:
        key = hash(source)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            ast = self.environment.parse(source)
        except jinja2.TemplateSyntaxError as exc:
            raise TemplateError(f"template syntax error: {exc.message} (line {exc.lineno})") from exc

        nodes_all = list(_walk(ast))

        # `store` names (set/for/macro targets) are local, never globals.
        assigned: set = set()
        for node in nodes_all:
            if isinstance(node, nodes.Name) and getattr(node, "ctx", None) == "store":
                assigned.add(node.name)
            elif isinstance(node, nodes.Macro):
                assigned.add(node.name)

        used: set = set()
        for node in nodes_all:
            if isinstance(node, nodes.Name):
                if getattr(node, "ctx", None) == "load":
                    used.add(node.name)
            elif isinstance(node, nodes.Call):
                if isinstance(node.node, nodes.Name):
                    used.add(node.node.name)

        used -= assigned
        result = frozenset(used)
        self._cache[key] = result
        return result


def _walk(root: nodes.Node):
    """Depth-first iteration over ``root`` and all its descendants.

    ``jinja2.nodes.Node`` only offers ``find``/``find_all`` (type filtered), so
    an explicit walker is the simplest way to visit everything once.
    """
    stack = [root]
    seen = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        for field_name, value in node.iter_fields():
            if isinstance(value, nodes.Node):
                stack.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, nodes.Node):
                        stack.append(item)


class TemplateEnvironment:
    """Loads one template and renders it with a Minja-flavoured environment."""

    def __init__(self, source: str, *, extra_globals: Optional[Mapping[str, Any]] = None,
                 name: str = "chat_template.jinja") -> None:
        if not isinstance(source, str) or not source.strip():
            raise TemplateError(f"{name}: template is empty")
        self.source = source
        self.name = name

        env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            undefined=jinja2.Undefined,          # Minja: undefined renders as ""
            autoescape=False,                    # prompts are plain text
            trim_blocks=False,
            lstrip_blocks=False,
            keep_trailing_newline=True,
            # `{% do %}` and `{% break %}`/`{% continue %}` appear in some
            # vendor templates; keep them working.
            extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
            cache_size=400,
        )
        env.globals["raise_exception"] = _raise_exception
        env.globals["strftime_now"] = _strftime_now
        env.globals["tojson"] = _tojson_global
        env.filters["tojson"] = _tojson_filter
        if extra_globals:
            for key, value in extra_globals.items():
                env.globals[key] = value
        self.env = env

        try:
            self.template = env.from_string(source)
        except jinja2.TemplateSyntaxError as exc:
            raise TemplateError(
                f"{name}:{exc.lineno}: template syntax error: {exc.message}"
            ) from exc

        self._analyzer = GlobalUseAnalyzer(env)

    # -- introspection -----------------------------------------------------
    def used_globals(self) -> frozenset:
        """Global variable/filters names referenced by the template."""
        return self._analyzer.used_names(self.source)

    def bakes(self, variable: str) -> bool:
        """True if the template turns ``{{ variable is defined }}`` into a constant.

        Some templates do ``{% if tools %}`` (fine) but a few do
        ``{% set tools = tools if tools is defined else none %}`` (also fine).
        What breaks capability detection is a template that *assigns* the
        variable unconditionally, e.g. ``{% set enable_thinking = true %}``:
        in that case the caller's value can never reach the template and we
        must not report it as a knob.

        ``{% set x = x | default(val) %}`` is NOT a bake — the caller's
        value reaches through the default filter.
        """
        try:
            ast = self.env.parse(self.source)
        except jinja2.TemplateSyntaxError:
            return False
        for node in ast.find_all(nodes.Assign):
            if isinstance(node.target, nodes.Name) and node.target.name == variable:
                # {% set x = x | default(...) %} — not a bake, value passes through
                rhs = getattr(node, "node", None) or getattr(node, "value", None)
                if isinstance(rhs, nodes.Filter) and rhs.name == "default":
                    continue
                return True
        for node in ast.find_all(nodes.AssignBlock):
            if isinstance(node.target, nodes.Name) and node.target.name == variable:
                return True
        return False

    def renders_with(self, **context: Any) -> bool:
        """Render with a probe context; return False if the template raises."""
        try:
            self.template.render(**context)
            return True
        except Exception:  # noqa: BLE001 - any template failure counts as "no"
            return False

    def render(self, context: Mapping[str, Any]) -> str:
        try:
            return self.template.render(**dict(context))
        except _RaisedJinjaError:
            raise
        except TemplateError:
            raise
        except jinja2.UndefinedError as exc:
            raise TemplateError(f"{self.name}: {exc}") from exc
        except jinja2.TemplateError as exc:
            raise TemplateError(f"{self.name}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise TemplateError(f"{self.name}: {type(exc).__name__}: {exc}") from exc

    def __repr__(self) -> str:  # pragma: no cover
        return f"<TemplateEnvironment {self.name} {len(self.source)} chars>"


def make_environment(source: str, **kwargs: Any) -> TemplateEnvironment:
    """Convenience factory mirroring the public API."""
    return TemplateEnvironment(source, **kwargs)


def load_template_file(path: str, *, encoding: str = "utf-8") -> TemplateEnvironment:
    try:
        with open(path, "r", encoding=encoding) as fh:
            source = fh.read()
    except OSError as exc:
        raise TemplateError(f"cannot read template {path!r}: {exc}") from exc
    return TemplateEnvironment(source, name=path)
