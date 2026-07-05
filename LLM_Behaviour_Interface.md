# LLM Behaviour Interface

This document defines the interface contract that any script must satisfy to be callable
as an LLM provider via UnifiedAiClient's `ScriptProvider` (provider name: `"script"`).

Scripts that satisfy this contract can be invoked exactly like any other AI provider:

```python
response = call_ai(
    provider="script",
    model="/path/to/my_script.py",
    prompt="...",
)
```

---

## 1. Invocation Model

UnifiedAiClient spawns the script as a **subprocess** for each `call_ai()` invocation.
The script must start, process one request, write one response, and exit. There is no
persistent connection between calls (stateless model).

**Invocation command:**

| Script type | Command used by ScriptProvider |
|---|---|
| Python script (`.py` extension) | See interpreter resolution below |
| Any other file | `<script_path>` (direct execution) |

Non-Python scripts must be executable (POSIX `chmod +x`, or a registered file
association on Windows).

**Python interpreter resolution (in priority order):**

1. If a `.venv` directory exists **in the script's own directory**, its Python
   interpreter is used:
   - Windows: `.venv/Scripts/python.exe`
   - Unix / macOS: `.venv/bin/python`
2. If no local `.venv` is found, `sys.executable` is used — the same interpreter
   that UnifiedAiClient is running under (same virtual environment, same installed
   packages).

This means each script can carry its own isolated dependencies without interfering
with the calling project. If your script requires packages not installed in the
calling project's environment, place a `.venv` alongside it and install its
requirements there.

---

## 2. Communication Protocol

All communication uses **stdin** (input) and **stdout** (output). **Stderr is reserved
for diagnostic output only** and is never parsed. The format is JSON in both directions.

```
UnifiedAiClient                        Script
      │                                   │
      │── JSON request → stdin ──────────►│
      │                                   │ (processes request)
      │◄─ JSON response ← stdout ─────────│
      │                                   │
      │                              exit(0)
```

### 2.1 The `mode` field

Every request payload contains a `mode` field that tells the script which operation
is being requested:

| `mode` value | Triggered by | Description |
|---|---|---|
| `"generate"` | `call_ai()` | Standard text generation. |
| `"embed"` | `get_embedding()` | Embedding vector generation. Optional. |

Scripts that only implement `"generate"` may treat any other `mode` as an error
(exit non-zero with a descriptive stderr message). Scripts that also implement
`"embed"` must branch on this field and handle both payloads.

### 2.2 Input: `generate` payload (stdin)

The script reads **one JSON object** from stdin. The object contains the following fields:

| Field | Type | Always present | Description |
|---|---|---|---|
| `mode` | `string` | ✅ Yes | Always `"generate"` for this payload. |
| `prompt` | `string` | ✅ Yes | The user prompt text. Never null or empty. |
| `system_prompt` | `string \| null` | ✅ Yes | System instruction, or null if not provided. |
| `messages` | `array \| null` | ✅ Yes | Chat history as a list of role/content objects, or null. |
| `file_path` | `array[string]` | ✅ Yes | List of absolute paths to attached files. Empty list if no files. Never null. |
| `temperature` | `float` | ✅ Yes | Sampling temperature. Range: 0.0–2.0. |
| `thinking` | `bool \| string` | ✅ Yes | Whether extended reasoning was requested (`true`/`false`) or let the provider decide (`"default"`). |
| `format_json` | `bool` | ✅ Yes | Whether JSON-formatted output was requested. |
| `timeout` | `int` | ✅ Yes | Maximum seconds allowed for the entire call. |
| `top_k` | `int` | ✅ Yes | Sampling parameter top_k. |
| `top_p` | `float` | ✅ Yes | Sampling parameter top_p. |
| `max_tokens` | `int \| null` | ✅ Yes | Limit on the number of generated tokens, or null. |
| `extra_options` | `dict \| null` | ✅ Yes | Dictionary of provider-specific options, or null. |
| `tools` | `array \| null` | ✅ Yes | List of tool definitions the model may call, or null if no tools are provided. |
| `tool_results` | `array \| null` | ✅ Yes | List of tool execution results to inject into the conversation, or null. |

All fields are always present. Fields that are not applicable to a given call are sent
as `null` or their zero/empty value (never omitted). The script must handle `null`
values and empty lists gracefully.

**`messages` element structure** (when not null):

Each message is a dict with `role` and `content`. Additional fields depend on the role:

```json
{"role": "user" | "assistant", "content": "..."}
```
```json
{"role": "user", "content": "...", "files": ["/path/to/file.pdf"]}
```
```json
{
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "function": {
                "name": "get_weather",
                "arguments": {"location": "Rome"}
            }
        }
    ]
}
```

Rules:
- Scripts that do not support file attachments in history may ignore the `files` key.
- Scripts that do not support tool calling may ignore the `tool_calls` key on assistant messages.

**Full example input:**
```json
{
    "mode": "generate",
    "prompt": "Summarize the previous discussion.",
    "system_prompt": "You are a concise summarizer.",
    "messages": [
        {"role": "user", "content": "Tell me about photosynthesis."},
        {"role": "assistant", "content": "Photosynthesis is the process by which..."}
    ],
    "file_path": [],
    "temperature": 0.3,
    "thinking": false,
    "format_json": false,
    "timeout": 300,
    "top_k": 64,
    "top_p": 0.95,
    "max_tokens": null,
    "extra_options": null
}
```

**Example with tool calling:**
```json
{
    "mode": "generate",
    "prompt": "What is the weather in Rome?",
    "system_prompt": null,
    "messages": null,
    "file_path": [],
    "temperature": 0.0,
    "thinking": false,
    "format_json": false,
    "timeout": 300,
    "top_k": 64,
    "top_p": 0.95,
    "max_tokens": null,
    "extra_options": null,
    "tools": [
        {
            "name": "get_weather",
            "description": "Returns the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    ],
    "tool_results": null
}
```

**Example with tool results (follow-up turn):**

The follow-up call must include the full conversation history in `messages`, including
the assistant's intermediate turn with `tool_calls`. This allows the model to link the
tool result back to its own request.

When `tool_results` is non-null, the library does **not** append the current `prompt`
as a new user message — the prompt is present in `messages` already.

> **Note on `messages` format:** The `tool_calls` structure in assistant messages is
> always the same OpenAI-style `{"function": {"name": ..., "arguments": {...}}}` format,
> regardless of which provider the consumer is targeting. For native providers
> (Anthropic, Google, OpenAI-compat, Ollama), the library converts this format
> internally to the provider's native representation before the API call.
> For the `script` provider, messages are passed through as-is — the script receives
> this exact format and is responsible for handling it.

```json
{
    "mode": "generate",
    "prompt": "What is the weather in Rome?",
    "system_prompt": null,
    "messages": [
        {
            "role": "user",
            "content": "What is the weather in Rome?"
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Rome"}
                    }
                }
            ]
        }
    ],
    "file_path": [],
    "temperature": 0.0,
    "thinking": false,
    "format_json": false,
    "timeout": 300,
    "top_k": 64,
    "top_p": 0.95,
    "max_tokens": null,
    "extra_options": null,
    "tools": [
        {
            "name": "get_weather",
            "description": "Returns the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    ],
    "tool_results": [
        {
            "call_id": "get_weather_0",
            "name": "get_weather",
            "content": "22 degrees Celsius and sunny in Rome."
        }
    ]
}
```

**Example with file attachment:**
```json
{
    "mode": "generate",
    "prompt": "Describe what's in this image.",
    "system_prompt": null,
    "messages": null,
    "file_path": ["/home/user/photos/landscape.jpg"],
    "temperature": 0.7,
    "thinking": false,
    "format_json": false,
    "timeout": 300
}
```

**Example with multiple files:**
```json
{
    "mode": "generate",
    "prompt": "Compare these two documents.",
    "system_prompt": null,
    "messages": null,
    "file_path": ["/data/report_2024.pdf", "/data/report_2025.pdf"],
    "temperature": 0.5,
    "thinking": true,
    "format_json": false,
    "timeout": 180
}
```

### 2.3 Output: `generate` response (stdout)

The script must write **one JSON object** to stdout before exiting. The object must
contain the following fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | `string` | ✅ Yes | The generated response text. Must not be null or empty. |
| `input_tokens` | `int` | ❌ No | Number of input tokens consumed. Defaults to 0 if absent. |
| `output_tokens` | `int` | ❌ No | Number of output tokens generated. Defaults to 0 if absent. |
| `reasoning_tokens` | `int` | ❌ No | Number of reasoning tokens used. Defaults to 0 if absent. |
| `reasoning_text` | `string` | ❌ No | Reasoning/thinking transcript. Defaults to `""` if absent. |
| `tool_calls` | `array \| null` | ❌ No | List of tool calls requested by the script's model. Omit or set to null/empty if the model replied with text. |

The `text` field is the only mandatory field. Scripts that do not track token usage
or reasoning may omit those fields entirely. If tool calls are returned, `text` may
be an empty string.

**Minimal valid output:**
```json
{"text": "Photosynthesis converts sunlight into chemical energy stored in glucose."}
```

**Full output (with reasoning):**
```json
{
    "text": "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "input_tokens": 18,
    "output_tokens": 12,
    "reasoning_tokens": 45,
    "reasoning_text": "The user asked about photosynthesis. I should explain the core mechanism...",
    "tool_calls": null
}
```

**Output with tool calls (model requesting tool execution):**
```json
{
    "text": "",
    "input_tokens": 25,
    "output_tokens": 10,
    "tool_calls": [
        {
            "id": "get_weather_0",
            "name": "get_weather",
            "arguments": {"location": "Rome"}
        }
    ]
}
```

> **Note on `reasoning_text`**: This field contains the model's thinking/reasoning process if produced. While thinking is explicitly requested via the `thinking` flag, some models may generate thoughts even when `thinking=false`. If reasoning is not supported or not produced, return `""` or omit the field. The thinking transcript is always returned to the caller via `AiResponse.reasoning_text` if present.

### 2.4 Input: `embed` payload (stdin) — optional

Sent by `get_embedding()`. The payload is intentionally minimal:

| Field | Type | Always present | Description |
|---|---|---|---|
| `mode` | `string` | ✅ Yes | Always `"embed"` for this payload. |
| `text` | `string` | ✅ Yes | The text to embed. Never null or empty. |

**Example input:**
```json
{
    "mode": "embed",
    "text": "Photosynthesis converts sunlight into chemical energy."
}
```

### 2.5 Output: `embed` response (stdout) — optional

| Field | Type | Required | Description |
|---|---|---|---|
| `embedding` | `array[float]` | ✅ Yes | The embedding vector as a list of floats. |

**Example output:**
```json
{
    "embedding": [0.123, -0.456, 0.789]
}
```

If the script does not support embedding, it must exit with a non-zero return code
and write a descriptive message to stderr. ScriptProvider will propagate the
`RuntimeError` to the caller.

---

## 3. Exit Codes

| Exit code | Meaning | ScriptProvider behaviour |
|---|---|---|
| `0` | Success. Stdout contains valid response JSON. | Parse stdout, return `AiResponse`. |
| Any non-zero | Failure. Stderr should contain a human-readable error message. | Raise `RuntimeError` with stderr content. The retry wrapper in UnifiedAiClient will attempt the call again up to `max_retries` times. |

---

## 4. Stderr

Stderr is **never parsed** by ScriptProvider. It is captured and included verbatim in
the `RuntimeError` message when the exit code is non-zero. Scripts may write any
diagnostic, debug, or logging output to stderr freely without affecting the protocol.

Do **not** write anything to stdout except the final response JSON object. Any extra
stdout output (print statements, progress messages) will corrupt the response and cause
a `json.JSONDecodeError` in ScriptProvider.

---

## 5. Timeout

The `timeout` field in the request JSON is informational — it tells the script how long
UnifiedAiClient is willing to wait. `subprocess.run()` enforces this same timeout at
the OS level: if the script does not exit within `timeout` seconds, the process is
killed and a `subprocess.TimeoutExpired` exception is raised.

Scripts performing long operations should respect the timeout value and abort early
with a non-zero exit code rather than being forcibly killed, to allow clean error
reporting.

---

## 6. Fields the Script May Ignore

The following fields are passed for completeness but many scripts will not use them:

- `thinking`: A hint that the caller wants extended reasoning. The script may implement
  a more thorough reasoning process when this is `true` (or decide by default when `"default"`), or ignore it entirely.
  If implemented, put the reasoning transcript in `reasoning_text` in the output.
- `format_json`: A hint that the caller wants JSON-formatted output in `text`. The
  script should produce valid JSON in the `text` field when this is `true`. If the
  script cannot guarantee JSON output, it may ignore this flag.
- `file_path`: List of file paths. Scripts that do not support multimodal input
  should ignore this field. Scripts that support it are responsible for reading and
  processing the files themselves — the library passes raw paths, not encoded data.
- `temperature`: A hint for output randomness. Deterministic scripts may ignore it.

---

## 7. Reference Implementation (Python)

This template satisfies all requirements for both `generate` and `embed` modes,
including the `file_path` (list) and `reasoning_text` fields.
Copy and adapt it as a starting point.

```python
#!/usr/bin/env python
# my_llm_script.py
# Implements the LLM-Behaviour Interface for UnifiedAiClient.
from __future__ import annotations
import json
import sys


def build_context(request: dict) -> str:
    """Assemble a context string from system prompt, history, and prompt.

    Args:
        request: The parsed request dict from stdin.

    Returns:
        A formatted string combining system prompt, history, and prompt.
    """
    parts: list[str] = []

    if request.get("system_prompt"):
        parts.append(f"[System]\n{request['system_prompt']}")

    messages: list[dict] | None = request.get("messages")
    if messages:
        for message in messages:
            role = message.get("role", "user").capitalize()
            parts.append(f"[{role}]\n{message.get('content', '')}")

    parts.append(f"[User]\n{request['prompt']}")

    return "\n\n".join(parts)


def handle_generate(request: dict) -> dict:
    """Handle a generate request and return the response dict.

    Replace the body of this function with your actual generation logic.

    Args:
        request: The full parsed request dict.

    Returns:
        A response dict with at least a 'text' key.
    """
    context: str = build_context(request)

    # file_path is now always a list (may be empty)
    file_paths: list[str] = request.get("file_path") or []

    # thinking flag: implement extended reasoning when True
    thinking: bool = request.get("thinking", False)

    # Replace with real logic.
    result_text = f"Echo: {request['prompt']}"
    reasoning_text = ""

    if thinking:
        # Example: add a reasoning step
        reasoning_text = f"Thinking about: {request['prompt']}"

    return {
        "text": result_text,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "reasoning_text": reasoning_text,
    }


def handle_embed(request: dict) -> dict:
    """Handle an embed request and return the response dict.

    Remove this function entirely if the script does not support embeddings.
    In that case, the 'embed' branch in main() should exit non-zero.

    Args:
        request: The parsed request dict (contains 'text').

    Returns:
        A response dict with an 'embedding' key (list of floats).
    """
    text: str = request["text"]

    # Replace with real embedding logic.
    embedding: list[float] = [0.0]

    return {"embedding": embedding}


def main() -> None:
    """Entry point. Reads request from stdin, dispatches on mode, writes response."""
    try:
        request: dict = json.loads(sys.stdin.read())
        mode: str = request.get("mode", "generate")

        if mode == "generate":
            response = handle_generate(request)
        elif mode == "embed":
            response = handle_embed(request)
        else:
            print(f"Unsupported mode: '{mode}'", file=sys.stderr)
            sys.exit(1)

        print(json.dumps(response))

    except Exception as exc:
        print(f"Script error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## 8. Integration Checklist for Existing Scripts

For each pre-existing script to be integrated as a `"script"` provider:

- [ ] **Read from stdin, not from CLI args.** Replace `argparse` input with
  `json.loads(sys.stdin.read())`.
- [ ] **Dispatch on `mode`.** Read `request.get("mode", "generate")` and branch
  accordingly. If only `generate` is needed, treat any other mode as an error.
- [ ] **Write one JSON object to stdout, then exit.** Replace any `print()` output
  with a single `print(json.dumps(response))` at the end.
- [ ] **Move all diagnostic output to stderr.** Replace `print(...)` debug/log lines
  with `print(..., file=sys.stderr)`.
- [ ] **Return exit code 0 on success.** Ensure no `sys.exit(1)` is called on the
  happy path.
- [ ] **Return exit code 1 on failure.** Wrap `main()` in a `try/except` that calls
  `sys.exit(1)` after printing the error to stderr.
- [ ] **Handle null fields.** `system_prompt` and `messages` may be `null`.
  Guard with `if request.get("field_name"):` before using them.
- [ ] **Add `reasoning_text` to output** if the script implements thinking.
  Return `""` if not implemented.
- [ ] **Do not write anything else to stdout.** Remove or redirect all intermediate
  print statements.
- [ ] **Respect the timeout** if the script runs long operations (optional but
  recommended).
- [ ] **If the script has its own dependencies:** create a `.venv` in the script's
  directory and install requirements there. ScriptProvider will detect and use it
  automatically. No changes to the calling project's environment are needed.

---

## 9. Testing a Script Manually

Before integrating with UnifiedAiClient, verify the script directly from the terminal.

**Test `generate` mode (no files):**
```bash
echo '{"mode": "generate", "prompt": "Hello", "system_prompt": null, "messages": null, "file_path": [], "temperature": 0.7, "thinking": false, "format_json": false, "timeout": 30}' | python my_llm_script.py
```

**Test `generate` mode (with file):**
```bash
echo '{"mode": "generate", "prompt": "Summarize this.", "system_prompt": null, "messages": null, "file_path": ["/tmp/doc.txt"], "temperature": 0.7, "thinking": false, "format_json": false, "timeout": 30}' | python my_llm_script.py
```

**Test with thinking:**
```bash
echo '{"mode": "generate", "prompt": "What is consciousness?", "system_prompt": null, "messages": null, "file_path": [], "temperature": 0.7, "thinking": true, "format_json": false, "timeout": 60}' | python my_llm_script.py
```

Expected: a single line of valid JSON on stdout with at least a `"text"` key.

**Test `embed` mode (if supported):**
```bash
echo '{"mode": "embed", "text": "Hello world"}' | python my_llm_script.py
```

Expected: a single line of valid JSON on stdout with an `"embedding"` key containing
a list of floats.

**Test unsupported mode (if embed is not implemented):**
```bash
echo '{"mode": "embed", "text": "Hello world"}' | python my_llm_script.py
echo $?   # Must be non-zero
```

Expected: a non-zero exit code and a descriptive message on stderr.

**Check exit codes:**
```bash
echo $?           # Unix / macOS
echo %ERRORLEVEL% # Windows
```
