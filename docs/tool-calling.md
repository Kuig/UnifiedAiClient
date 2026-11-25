# Tool Calling

The library provides minimal, transport-level tool calling support. It passes
tool definitions to the model and returns any tool calls the model requests.
The **execution loop** is the consumer's responsibility.

## Defining tools

```python
from unified_ai_client import call_ai, ToolDefinition, ToolCall, ToolResult

tools = [
    ToolDefinition(
        name="get_weather",
        description="Returns the current weather for a given city.",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. Rome",
                },
            },
            "required": ["location"],
        },
    ),
]
```

## The tool loop

Every response carries `message`, the turn the model just produced as a history entry, and
every `ToolResult` has `to_message()`. Appending both to `messages` is the whole loop, for
as many rounds as the model needs:

```python
from unified_ai_client import call_ai, ToolResult

prompt = "What is the weather in Rome right now? Use the get_weather tool."
history = [{"role": "user", "content": prompt}]

response = call_ai(provider="ollama", model="gemma4:12b", prompt=prompt, tools=tools)

while response.tool_calls:
    # Running the tools is the consumer's responsibility.
    results = [
        ToolResult(call_id=tc.id, name=tc.name, content=f"22°C and sunny in {tc.arguments['location']}.")
        for tc in response.tool_calls
    ]
    history += [response.message, *(r.to_message() for r in results)]

    # Nothing new to say: the history carries the turn, so prompt stays empty.
    response = call_ai(
        provider="ollama", model="gemma4:12b", prompt="", messages=history, tools=tools
    )

print(response.text)  # "The weather in Rome is 22°C and sunny."
```

The entries are plain dictionaries, JSON-serialisable, so a history can be stored and
resumed. Their shape is the same on every provider:

| Entry | Shape |
|---|---|
| Assistant turn (`response.message`) | `{"role": "assistant", "content": ..., "tool_calls": [{"id", "name", "arguments"}], "provider_state": {...}}` |
| Tool result (`ToolResult.to_message()`) | `{"role": "tool", "tool_call_id": ..., "name": ..., "content": ...}` |

`provider_state` holds the provider's own reply, unmodified. Some providers require parts
of it back on the next request: Claude's signed thinking blocks (on by default from the 5.5
models) and Gemini 3's thought signatures, without which a replayed function call is a
400. The same provider replays the state as it is; any other provider ignores it and
builds its request from the canonical fields, so one history can move between providers.
Edit a turn by dropping its `provider_state` along with the change: while the state is
there, it is what gets sent.

A new user message after a tool result is an ordinary entry with `prompt` set: the
library appends the current turn only when there is a prompt or a file to send, or no
history at all.

`tool_results` still works for a single follow-up, as before: pass the results through it
and keep the user turn and `response.message` in `messages`. With `tool_results` present
the prompt is never appended again.

`content`, `files` and `tool_calls` on the same history entry are all preserved together on
every provider: a message that carries an attachment alongside its `tool_calls` does not
lose the attachment (or the text) for having both. See
[Multimodal input](multimodal.md#attaching-files-to-history) for the same guarantee from
the file side.

## Forbidding tool calls: `tool_choice`

A loop with a round limit needs a way to stop. When the limit is reached and the model
still asks for tools, the last call should get a text answer out of what the history
already holds. `tool_choice="none"` asks for exactly that: the history keeps its earlier
calls and results, the definitions still travel, and the model may not call any of them.

```python
MAX_ROUNDS = 5
rounds = 0
while response.tool_calls and rounds < MAX_ROUNDS:
    ...  # run the tools and extend history, as above
    rounds += 1
    response = call_ai(provider="ollama", model="gemma4:12b", prompt="",
                       messages=history, tools=tools)

if response.tool_calls:
    # Round limit reached: one closing turn, text only.
    response = call_ai(
        provider="ollama", model="gemma4:12b",
        prompt="You cannot call more tools. Answer now with what you have.",
        messages=history, tools=tools, tool_choice="none",
    )
```

Give the closing turn a prompt that says so. Here the model still wants another tool,
and forbidding the call does not tell it what to do instead. Seen live on Gemini with an
empty prompt: `gemini-2.5-flash` answered with nothing at all, and `gemini-3.5-flash`
returned a function call anyway. With the prompt above, both answered in text.

| Value | Effect |
|---|---|
| `"auto"` (default) | The model decides whether to call a tool. Nothing is added to the request. |
| `"none"` | The model may not call a tool. |

How `"none"` reaches the provider:

- **A native control** where the API has one: `tool_choice: {"type": "none"}` on
  `anthropic` (accepted with every thinking mode), `FunctionCallingConfig(mode="NONE")`
  on `google`, `tool_choice: "none"` on `openai`, `mistral`, `groq`, `xai`, `lmstudio`
  and `llamacpp`, and a `tool_choice` field for the `script`.
- **The definitions are left out** where it has none: `ollama` (no such field; a history
  with tool calls and results is accepted without definitions), `meta` (its API answers
  `"none"` with a 400) and `cohere` (its compatibility API does not list the parameter).
  Without definitions the model has nothing to call.

The rules around it:

- With no `tools`, `"none"` changes nothing, and no error is raised.
- `"none"` together with `web_search=True` raises `ValueError` before anything is sent: on
  Anthropic the same switch also blocks the search tool, so the search would silently
  never run.
- A `tool_choice` in `extra_options` is the provider's own field and wins over this one.
- Any other value raises `ValueError`. Forcing a call (`"required"`, or one named tool)
  is not offered: Claude Opus 5.5, Sonnet 5.5 and Fable refuse it with a 400, Meta's
  API accepts `"auto"` only, and Mistral spells it `"any"`.

A model can still return `tool_calls` under `"none"`, so a loop must not assume an empty
list. Seen live: after a tool history, `gemini-3.5-flash` returns a function call under
mode NONE when the user asks for another tool, with or without the definitions.
`gemini-2.5-flash` honours the mode. On Groq, reports for the GPT-OSS models describe
the same problem. What to do with such a reply is the consumer's decision.

## Provider compatibility

All providers support tool calling, except Ollama with `use_generate=True`, whose
`/api/generate` endpoint has no place for tools and refuses them. Whether a specific model
will actually use tool calling depends on its training, not the provider.

| Provider | Tool Calling | `tool_choice="none"` | Notes |
|---|---|---|---|
| `google` | ✅ | Native | `FunctionDeclaration` / `function_call` parts. Gemini 3 may ignore it after a tool history |
| `anthropic` | ✅ | Native | `input_schema` format / `tool_use` content blocks |
| `openai` | ✅ | Native | Standard OpenAI format |
| `mistral` | ✅ | Native | Inherited from the OpenAI-compatible base |
| `cohere` | ✅ | Definitions omitted | Inherited from the OpenAI-compatible base |
| `meta` | ✅ | Definitions omitted | Inherited from the OpenAI-compatible base |
| `groq` | ✅ | Native | Inherited from the OpenAI-compatible base |
| `xai` | ✅ | Native | Inherited from the OpenAI-compatible base |
| `lmstudio` | ✅ | Native | Inherited from the OpenAI-compatible base |
| `llamacpp` | ✅ | Native | Inherited from the OpenAI-compatible base |
| `ollama` | ✅ | Definitions omitted | OpenAI-compatible format via `/api/chat` |
| `script` | ✅ | Script-defined | Extended JSON protocol, see [the script protocol](script-protocol.md) |

> [!NOTE]
> For `ollama`, tool calling requires models specifically trained for it (e.g. `gemma4`, `qwen3`, `llama3.1`). The library sends the tool definitions regardless; if the model ignores them, `AiResponse.tool_calls` will be empty and you will receive a plain text response.

The three dataclasses involved (`ToolDefinition`, `ToolCall`, `ToolResult`) are
documented field by field in the [API reference](api.md#tooldefinition).
