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

## Provider compatibility

All providers support tool calling, except Ollama with `use_generate=True`, whose
`/api/generate` endpoint has no place for tools and refuses them. Whether a specific model
will actually use tool calling depends on its training, not the provider.

| Provider | Tool Calling | Notes |
|---|---|---|
| `google` | ✅ | `FunctionDeclaration` / `function_call` parts |
| `anthropic` | ✅ | `input_schema` format / `tool_use` content blocks |
| `openai` | ✅ | Standard OpenAI format |
| `mistral` | ✅ | Inherited from the OpenAI-compatible base |
| `cohere` | ✅ | Inherited from the OpenAI-compatible base |
| `meta` | ✅ | Inherited from the OpenAI-compatible base |
| `groq` | ✅ | Inherited from the OpenAI-compatible base |
| `xai` | ✅ | Inherited from the OpenAI-compatible base |
| `lmstudio` | ✅ | Inherited from the OpenAI-compatible base |
| `llamacpp` | ✅ | Inherited from the OpenAI-compatible base |
| `ollama` | ✅ | OpenAI-compatible format via `/api/chat` |
| `script` | ✅ | Extended JSON protocol, see [the script protocol](script-protocol.md) |

> [!NOTE]
> For `ollama`, tool calling requires models specifically trained for it (e.g. `gemma4`, `qwen3`, `llama3.1`). The library sends the tool definitions regardless; if the model ignores them, `AiResponse.tool_calls` will be empty and you will receive a plain text response.

The three dataclasses involved (`ToolDefinition`, `ToolCall`, `ToolResult`) are
documented field by field in the [API reference](api.md#tooldefinition).
