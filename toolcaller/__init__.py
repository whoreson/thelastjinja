"""toolcaller - turn OpenAI chat-completion JSON into raw text completions.

Motivation
----------
Some inference servers (ik_llama.cpp / llama.cpp built or run without
``--jinja``, many "text completion only" backends, vendor-specific servers...)
expose *only*:

* ``POST /completion``  - raw prompt in, raw text out
* ``POST /tokenize``    - optional, lets us match stop strings exactly

and provide **no** chat-completions endpoint, **no** jinja rendering and
**no** tool-call parsing.  ``toolcaller`` fills that gap:

    OpenAI chat completion JSON  --(jinja template)-->  raw prompt
                                      |
                                      v
                              POST /completion
                                      |
                                      v
        raw text  --(model-specific tool-call parser)-->  OpenAI JSON

Everything model specific lives in small, declarative "profiles"
(``toolcaller.profiles``) so that supporting a new model family means adding
one profile - no changes to the engine.

Quick start::

    python -m toolcaller --template chat_template.jinja \\
        --url http://localhost:8082 --chat request.json

    # or from python
    from toolcaller import ChatPipeline, load_chat_request
    pipe = ChatPipeline(template_path="chat_template.jinja",
                        base_url="http://localhost:8082")
    result = pipe.complete(load_chat_request("request.json"))
"""

from .errors import ToolcallerError, TemplateError, ServerError, ParseError, ConfigError
from .messages import (
    load_chat_request,
    normalize_request,
    normalize_messages,
    messages_to_json,
)
from .caps import TemplateCaps, analyze_caps
from .environment import TemplateEnvironment, make_environment, load_template_file
from .renderer import PromptRenderer, RenderResult
from .client import CompletionClient, CompletionResult
from .constraints import GenerationConstraints, build_constraints
from .server import ServerConfig, create_server, run_server
from .parser import parse_completion, ParserSelection, ParsedMessage
from .hints import build_hints, TokenHints
from .api import ChatPipeline, ChatResult
from .profiles import (
    SyntaxProfile,
    get_profile,
    register_profile,
    available_profiles,
)

__version__ = "1.0.0"

__all__ = [
    "ToolcallerError",
    "TemplateError",
    "ServerError",
    "ParseError",
    "ConfigError",
    "load_chat_request",
    "normalize_request",
    "normalize_messages",
    "messages_to_json",
    "TemplateCaps",
    "analyze_caps",
    "TemplateEnvironment",
    "make_environment",
    "load_template_file",
    "PromptRenderer",
    "RenderResult",
    "CompletionClient",
    "CompletionResult",
    "GenerationConstraints",
    "build_constraints",
    "parse_completion",
    "ParserSelection",
    "ParsedMessage",
    "build_hints",
    "TokenHints",
    "ChatPipeline",
    "ChatResult",
    "SyntaxProfile",
    "get_profile",
    "register_profile",
    "available_profiles",
]
