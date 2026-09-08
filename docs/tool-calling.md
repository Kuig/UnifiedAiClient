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

## The two-turn exchange

```python
prompt = "What is the weather in Rome right now? Use the get_weather tool."

# Turn 1: model may respond with tool calls
response = call_ai(
    provider="ollama",
    model="gemma4:12b",
    prompt=prompt,
    tools=tools,
    temperature=0.0,
)

if response.tool_calls:
    # Execute the tool (consumer's responsibility)
    results = []
    for tc in response.tool_calls:
        if tc.name == "get_weather":
            content = f"The weather in {tc.arguments['location']} is 22°C and sunny."
        else:
            content = "Tool not found."
        results.append(ToolResult(call_id=tc.id, name=tc.name, content=content))

    # Build the conversation history for turn 2.
    # The assistant's intermediate message (with tool_calls) must be included
    # so the model can link the tool result back to its own request.
    #
    # IMPORTANT: this format is always the same regardless of the provider.
    # The library converts it internally to each provider's native format:
    #   - Anthropic  →  tool_use content blocks
    #   - Google     →  function_call Parts
    #   - Ollama / OpenAI-compat  →  OpenAI-style tool_calls (passed through)
    assistant_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": tc.name, "arguments": tc.arguments}} for tc in response.tool_calls],
    }

    # Turn 2: pass full history in messages + tool_results.
    # When tool_results are provided, the library does NOT re-append the
    # current prompt as a new user message; the consumer controls the history.
    final = call_ai(
        provider="ollama",
        model="gemma4:12b",
        prompt=prompt,           # passed for reference but not re-appended
        messages=[
            {"role": "user", "content": prompt},
            assistant_msg,
        ],
        tools=tools,
        tool_results=results,
        temperature=0.0,
    )
    print(final.text)  # "The weather in Rome is 22°C and sunny."
else:
    print(response.text)  # Model answered directly without tools
```

## A third turn: continuing after a tool result

A `tool_results` argument only ever applies to the turn it is passed on. Sending a further
message after that needs the full history rebuilt in `messages`, including the `role:
"tool"` entry the previous turn produced, with its `tool_call_id` intact so the model can
still link the answer to its own request:

```python
history = [
    {"role": "user", "content": prompt},
    assistant_msg,
    {
        "role": "tool",
        "content": results[0].content,
        "tool_call_id": results[0].call_id,
    },
]

follow_up = call_ai(
    provider="ollama",
    model="gemma4:12b",
    prompt="Should I bring an umbrella?",
    messages=history,
    temperature=0.0,
)
```

`content`, `files` and `tool_calls` on the same history entry are all preserved together on
every provider: a message that carries an attachment alongside its `tool_calls` does not
lose the attachment (or the text) for having both. See
[Multimodal input](multimodal.md#attaching-files-to-history) for the same guarantee from
the file side.

## Provider compatibility

All providers support tool calling. Whether a specific model will actually use
tool calling depends on its training, not the provider.

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
