# toolcaller

Turn **OpenAI chat-completion JSON** into a **raw text completion** on servers
that only speak `/completion` — and get proper **tool calls** back.

```
OpenAI chat JSON ──► jinja template ──► prompt ──► POST /completion ──► raw text
                                                                          │
                                            ◄── model-specific parser ◄───┘
                                     OpenAI chat.completion JSON
```

Built for the exact situation this was written in: `ik_llama.cpp` /
`llama.cpp` running **without `--jinja`**, no chat-completions endpoint, no
tool-call parsing — just `POST /completion` and a GGUF with an embedded chat
template (tested against Qwen 3.6 27B on CPU at ~2 tok/s).

---

## Why this is not just "render and POST"

| Problem | What a naive wrapper does | What this does |
|---|---|---|
| **Tool-call format** | model emits free-form text, client gets nothing | 12 declarative syntax **profiles** (Qwen XML, Spark XML, Mistral `[TOOL_CALLS]`, DeepSeek DSML, GPT-OSS harmony, Kimi, LFM2, Gemma, Llama 3, Hermes, …) + JSON recovery fallback |
| **Stopping** | stops on `</tool_call>`, breaking parallel calls; or never stops | stop list derived from the template + profile; `</function>` and `</tool_call>` are *never* stop strings by default |
| **EOS handling** | `content` from `/completion` already excludes EOS | raw text reconstructed (`content + stop_word + eos`) so reasoning/tool markers survive |
| **Thinking models** | prompt ends with `<think>` so `reasoning_content` is always empty | generation prompt diffed out, pre-filled `<think>` handled, `reasoning_content` split correctly |
| **Template dialects** | plain `jinja2` breaks (`raise_exception`, `tojson` escaping, `trim_blocks`) | Minja-compatible environment (`raise_exception`, `strftime_now`, compact unescaped `tojson`, no block trimming) |
| **Not every template is the same** | passing `tools` to a template that ignores them silently loses them | capability probes (port of `common/jinja/caps.cpp`); fails loudly instead of dropping data |
| **Model won't follow the format** | — | optional profile-derived system-prompt hint (`--hint-mode`) and lazy GBNF grammar (`--grammar-style gbnf`) |

---

## Install / run

Python 3.10+ with `jinja2` and `requests` (both already present in
`~/python-3.10.7`). No other dependencies.

```bash
cd ~/jin
~/python-3.10.7/bin/python3 -m toolcaller --help
```

### Inspect the server first

```bash
python3 -m toolcaller --check-server --url http://localhost:8082
python3 -m toolcaller --inspect-template -t chat_template.jinja
python3 -m toolcaller --list-profiles
```

### Render only (see exactly what the model will read)

```bash
python3 -m toolcaller -t chat_template.jinja \
    --chat examples/request_multi_turn_tool_result.json --render-only
```

### A real tool-calling round trip

```bash
python3 -m toolcaller -t chat_template.jinja \
    --url http://localhost:8082 \
    --chat examples/request_tool_call.json \
    --template-kwarg enable_thinking=false \
    --max-tokens 300
```

```json
{
  "id": "chatcmpl-…",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_…",
            "type": "function",
            "function": { "name": "get_weather", "arguments": "{\"city\":\"Budapest\"}" }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": { "prompt_tokens": 291, "completion_tokens": 54, "total_tokens": 345 }
}
```

### Parse text you already have (no server)

```bash
echo '<tool_call><function=reset></function></tool_call>' > out.txt
python3 -m toolcaller --parse-file out.txt --profile qwen
```

### Use it as a library

```python
from toolcaller import ChatPipeline

pipe = ChatPipeline(
    template_path="chat_template.jinja",
    base_url="http://localhost:8082",
    template_kwargs={"enable_thinking": False},
)

result = pipe.complete({
    "messages": [{"role": "user", "content": "Weather in Budapest?"}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]},
        },
    }],
})

print(result.message)          # OpenAI assistant message (dict)
print(result.to_response())    # full chat.completion object
print(result.render.prompt)    # exact prompt that was sent
```

---

## Adding a new model family

Two ways — neither requires touching the engine.

### 1. A profile file (no code)

```bash
python3 -m toolcaller --write-example-profile my_model.json
# edit my_model.json …
python3 -m toolcaller -t my_template.jinja --chat req.json --profile-file my_model.json
```

```jsonc
{
  "name": "my-model",
  "call_start": "<tool_call>",          // opens a call
  "call_end":   "</tool_call>",         // closes it (safe stop string)
  "function_tag_open": "<function=",    // name starts after this…
  "function_name_terminator": ">",      // …and ends here
  "function_tag_close": "</function>",
  "argument_style": "xml_params",       // json | json_array | xml_params | python_kwargs
  "parameter_tag_open": "<parameter=",
  "parameter_tag_close": "</parameter>",  // optional, derived by default
  "parameter_name_terminator": ">",       // optional
  "reasoning_start": "<think>",
  "reasoning_end": "</think>",
  "extra_stop_strings": [],              // extra safe terminators
  "options": {}
}
```

### 2. A code profile

```python
from toolcaller import SyntaxProfile, register_profile

register_profile(SyntaxProfile(
    name="my-model",
    call_start="<<tool>>", call_end="<</tool>>",
    function_tag_open="fn:", function_name_terminator=";",
    argument_style="json",
))
```

Then `--profile my-model`, or let `--auto-profile` (default) detect it from the
template's markers.

### What a profile controls

* where one tool call starts/ends, and where the function name sits
* how the arguments are encoded (`xml_params`, `json`, `json_array`, `python_kwargs`)
* reasoning markers
* safe stop strings
* the exact text of the prompt-side format hint

The parser itself handles: parallel calls, preamble text before calls,
multi-line and typed argument values, truncated calls (`--max-tokens` cut off),
JSON-fenced fallbacks, `{"name","arguments"}`/`{"name","parameters"}` wrappers,
string-encoded arguments, unquoted keys (Gemma) and call-id extraction.

---

## Reference

### `ChatPipeline`

| Argument | Meaning |
|---|---|
| `template_path` / `template_source` | the jinja chat template |
| `base_url` | llama.cpp-style server (`/completion`, `/tokenize`, `/props`) |
| `profile` / `profile_file` / `auto_profile` | tool-call syntax selection |
| `bos_token` / `eos_token` / `add_bos` / `add_eos` | template tokens (auto-discovered from `/props`) |
| `hint_mode` | `auto` (default), `always`, `never` |
| `grammar_style` | `none` (default) or `gbnf` (lazy grammar + trigger) |
| `reasoning_format` | `auto`, `none`, `deepseek`, `deepseek-legacy`, `extract` |
| `template_kwargs` | extra template variables, e.g. `{"enable_thinking": False}` |
| `strict_template` | fail when the request needs features the template lacks |
| `extra_stops` / `stop_at_call_end` | additional sanitised stop strings |
| `strip_generation_thinking` | drop a pre-filled `<think>` so the model emits it itself |
| `client` / `timeout` / `discover_tokens` | transport details |

`pipeline.complete(request, ...)` returns a `ChatResult` with `message`,
`finish_reason`, `parsed` (reasoning/content/tool calls), `render` (prompt +
generation prompt), `completion` (raw server data), `warnings` and `to_response()`.

### CLI highlights

```
--chat FILE | --messages JSON | --stdin     input
--template FILE                             jinja template (required to render)
--render-only                               print prompt + caps, never call the server
--parse-file FILE                           parse raw text (no server)
--profile NAME | --profile-file FILE        tool-call syntax
--reasoning-format {auto,none,deepseek,…}   reasoning handling
--hint-mode {auto,always,never}             prompt-side format reminder
--grammar-style {none,gbnf}                 lazy grammar constraining
--stop-at-call-end                          faster stops, breaks parallel calls
--strip-generation-thinking                 make the model emit <think> itself
--response-format {openai,simple,raw,both}  output shape
--stream                                    stream from the server
--no-stream                                 fetch the whole completion in one /completion call
--dump                                      print the received JSON request to stderr
--serve                                     run the OpenAI-compatible daemon
--inspect-template / --check-server / --list-profiles / --dump-profile NAME
```

`--no-stream` controls whether the proxy **streams from the backend**, not the
format of the client-facing response. On the CLI it fetches the whole
completion in a single `/completion` call (upstream receives `"stream": false`).
With `--serve` the same applies per request, but the response format still
follows the client's `"stream"` field:

| Client `stream` | `--no-stream` | Upstream call | Response body |
|---|---|---|---|
| `false` | either | blocking or streamed | `chat.completion` JSON |
| `true` | off | streamed `/completion` | SSE as tokens arrive |
| `true` | on | one blocking `/completion` | SSE, emitted in one pass at the end |

A client that asked for SSE always gets SSE (OpenAI SDK clients - including
agentic harnesses such as Pi - iterate the body as a stream and fail with
"Stream ended without finish_reason" if handed a plain JSON object).

### Inspecting the request

`--dump` prints the JSON request as received, verbatim, to **stderr**, before
any processing.  stderr keeps stdout machine-readable, so piping the result
still works:

```bash
python3 -m toolcaller -t qwen.jinja --chat examples/request_tool_call.json \
    --dump --render-only | jq .profile
```

It applies to the daemon too, where each `POST /v1/chat/completions` body is
written to the server's stderr (never into the SSE/JSON response).

### Streaming format (in-band tool calls)

With SSE enabled the daemon emits the deltas agentic clients expect, in order:

| Delta | Meaning |
|---|---|
| `{"role":"assistant"}` | first chunk |
| `{"reasoning_content":"..."}` | chain of thought |
| `{"content":"..."}` | answer text |
| `{"tool_calls":[{"index":0,"id":"call_…","type":"function","function":{"name":"…","arguments":""}}]}` | tool call begins |
| `{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\":\\"Bud…"}}]}` | argument fragments |
| `{"delta":{},"finish_reason":"tool_calls"}` then `[DONE]` | end of stream |

Argument fragments are streamed incrementally (never replayed at the end) and
concatenate to a JSON object. Tool-call markup (`<tool_call>`, `<arg_key>`, …)
and reasoning markers never leak into `content`. `toolcaller/streamer.py` does
the splitting while tokens arrive, so non-streaming and streaming agree.

### Known limitations

* **Thinking is pre-filled by many templates.** Qwen's generation prompt ends
  with `<think>`, so the model never emits `<think>`/`</think>` itself and
  `reasoning_content` can only be split after `</think>` appears. Three
  options:

  | Goal | Flag |
  |---|---|
  | fastest tool calls | `--template-kwarg enable_thinking=false` |
  | proper `reasoning_content` always | `--strip-generation-thinking` |
  | treat pre-`</think>` text as reasoning | `--reasoning-format deepseek` |

  `--strip-generation-thinking` is the most faithful: the prompt ends at
  `<|im_start|>assistant` and the model produces its own `<think>…</think>`
  block, which is then split into `reasoning_content` and `content`.
* **`stop_at_call_end` off by default** — stopping at `</tool_call>` truncates
  parallel calls, so the wrapper prefers end-of-turn stopping.
* **No grammar constraining by default.** `--grammar-style gbnf` uses a
  deliberately loose lazy grammar; use a `--jinja` server for exact schemas.
* `logprobs`, `n>1`, images/audio and `response_format: json_schema` are not
  forwarded (raw completions have no place for them).

---

## Layout

```
toolcaller/
  messages.py     OpenAI request normalisation (llama.cpp-compatible)
  environment.py  Minja-compatible jinja2 environment + global usage analysis
  caps.py         template capability probing (port of common/jinja/caps.cpp)
  renderer.py     prompt rendering + llama.cpp message workarounds
  client.py       /completion (+ /tokenize, /props), SSE streaming
  constraints.py  stop strings / grammar assembly
  hints.py        profile-derived prompt hints + lazy GBNF
  syntax.py       SyntaxProfile + argument decoders
  profiles.py     built-in profiles + aliases + profile files
  parser.py       generic profile-driven tool-call parser
  streamer.py     incremental reasoning/content/tool-call stream splitter
  api.py          ChatPipeline / ChatResult
  __main__.py     CLI

examples/         ready-to-run chat requests
tests/            243 unit tests (`python3 -m unittest discover -s tests`)
```

## Tests

```bash
cd ~/jin && ~/python-3.10.7/bin/python3 -m unittest discover -s tests -v
```
