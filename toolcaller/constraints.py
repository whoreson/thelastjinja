"""Decide which stop strings / grammar to send with the raw completion.

Why this is not trivial
-----------------------
The rendered prompt already contains ``<|im_end|>`` (end of every previous
turn) and, for thinking models, a trailing ``<think>``.  During generation
we want the model to stop at:

* the template's end-of-turn token (``<|im_end|>``) - or the server's EOS
* the profile's call terminator (``</tool_call>``, ``<|call|>``, ...)

but *not* at text the model is allowed to repeat (e.g. ``</think>`` is
legitimate in the middle of an answer).

We compute the stop list from the **rendered prompt**, not from a static
config, so the same wrapper works for any template:

1. always stop at the template's ``eos_token`` (if known)
2. add every terminator of the selected profile
3. optionally (``--stop-at-generation-prompt``) also stop at the
   generation *prefix* without its trailing ``<think>`` so thinking can be
   disabled - off by default because it only helps some models.

The server is asked to match these stops on **token boundaries** when
``/tokenize`` is available (see :mod:`toolcaller.client`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .syntax import SyntaxProfile


@dataclass
class GenerationConstraints:
    """Stop strings + optional grammar for one completion request."""

    stops: List[str] = field(default_factory=list)
    grammar: Optional[str] = None
    grammar_triggers: List[dict] = field(default_factory=list)
    preserved_tokens: List[str] = field(default_factory=list)

    def merge(self, hints: "object") -> "GenerationConstraints":
        """Merge a :class:`toolcaller.hints.TokenHints` instance."""
        grammar = getattr(hints, "grammar", None)
        if grammar:
            self.grammar = grammar
            self.grammar_triggers = list(getattr(hints, "grammar_triggers", []) or [])
        for token in getattr(hints, "preserved_tokens", []) or []:
            if token and token not in self.preserved_tokens:
                self.preserved_tokens.append(token)
        return self

    def to_dict(self) -> dict:
        return {
            "stops": list(self.stops),
            "grammar": self.grammar,
            "grammar_triggers": list(self.grammar_triggers),
            "preserved_tokens": list(self.preserved_tokens),
        }


def build_constraints(
    profile: SyntaxProfile,
    *,
    eos_token: str = "",
    extra_stops: Sequence[str] = (),
    template_stops: Sequence[str] = (),
    has_tools: bool = True,
    request_stops: Sequence[str] = (),
    stop_at_call_end: bool = False,
) -> GenerationConstraints:
    """Assemble the stop list for one request.

    Parameters
    ----------
    eos_token:
        The template's ``eos_token`` (or the model's EOS supplied by the
        caller).  Always a stop.
    extra_stops:
        User supplied ``--stop`` values.
    template_stops:
        Extra stops that the *template* implies (rare, e.g. harmony).
    has_tools:
        Without tools, call terminators are pointless but harmless - they are
        only added when tools are present to avoid cutting off content that
        legitimately mentions e.g. ``</tool_call>``.
    request_stops:
        OpenAI ``stop`` parameter values.
    stop_at_call_end:
        Also stop at the profile's ``call_end``.  Faster (no end-of-turn
        tail) but truncates parallel tool calls, so off by default.
    """
    stops: List[str] = []
    seen = set()

    def add(value: Optional[str]) -> None:
        if not value:
            return
        if value in seen:
            return
        seen.add(value)
        stops.append(value)

    add(eos_token)
    for value in template_stops:
        add(value)
    for value in request_stops:
        add(value)

    if has_tools:
        for terminator in profile.stop_strings(stop_at_call_end=stop_at_call_end):
            add(terminator)
    for value in extra_stops:
        add(value)

    preserved = [s for s in stops if _looks_like_special_token(s)]
    return GenerationConstraints(stops=stops, preserved_tokens=preserved)


def _looks_like_special_token(text: str) -> bool:
    """Heuristic: ``<|im_end|>``-style control tokens must be preserved."""
    if not text:
        return False
    if text.startswith("<|") and text.endswith("|>"):
        return True
    if text in ("<s>", "</s>", "<|endoftext|>"):
        return True
    return False


def strip_generation_reasoning(generation_prompt: str, profile: SyntaxProfile) -> str:
    """Remove the reasoning prefill from the generation prompt.

    Two shapes exist in the wild:

    * ``<|im_start|>assistant\\n<think>\\n`` - the model is placed *inside* a
      reasoning block, so it never emits ``<think>``/``</think>`` itself.
    * ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`` - thinking disabled:
      the block is pre-closed and nothing needs to be stripped.

    Only the first shape is modified (the trailing marker is dropped) so the
    model has to emit ``<think>`` itself and reasoning can be split from
    content.  The second shape is returned unchanged.
    """
    start = profile.reasoning_start
    end = profile.reasoning_end
    if not generation_prompt or not start:
        return generation_prompt

    stripped = strip_trailing_whitespace(generation_prompt)
    # Pre-closed block (thinking disabled) -> leave it alone.
    if end and stripped.endswith(end):
        return generation_prompt
    if stripped.endswith(start):
        return stripped[: -len(start)]
    return generation_prompt


def strip_trailing_whitespace(text: str) -> str:
    return text.rstrip(" \t\r\n")
