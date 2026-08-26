# UnifiedAiClient

`UnifiedAiClient` is a shared Python library providing a unified AI provider abstraction.
It routes all AI calls through a single `call_ai()` interface, hiding provider-specific 
encoding, upload, retry, and error-handling details from consuming projects.

See [COMPARISON.md](COMPARISON.md) for a feature-by-feature comparison with other libraries.

Supported providers:

| Provider | Type | Implementation Notes / SDK |
|---|---|---|
| `"google"` | Cloud | Official `google-genai` SDK |
| `"anthropic"` | Cloud | Native HTTP API via `urllib` (no SDK dependencies) |
| `"openai"` | Cloud | Native HTTP API via `urllib` (no SDK dependencies) |
| `"mistral"` | Cloud | OpenAI-compatible endpoint via `urllib` |
| `"cohere"` | Cloud | OpenAI-compatible endpoint via `urllib` |
| `"meta"` | Cloud | OpenAI-compatible endpoint via `urllib` |
| `"groq"` | Cloud | OpenAI-compatible endpoint via `urllib` |
| `"xai"` | Cloud | OpenAI-compatible endpoint via `urllib` |
| `"ollama"` | Local | Native Ollama HTTP API via `urllib` (no SDK dependencies) |
| `"lmstudio"` | Local | OpenAI-compatible API via `urllib` |
| `"llamacpp"` | Local | OpenAI-compatible API via `urllib` |
| `"script"` | External | Subprocess runner with JSON stdin/stdout protocol |

---

## Installation

**Local development** (when you're also editing the library):

```bash
# Activate your project's .venv first, then run:
pip install -e /path/to/UnifiedAiClient
```

**Production / other machines** — declare in your project's `requirements.txt`:

```text
unified-ai-client @ git+https://github.com/Kuig/UnifiedAiClient.git@v0.3.2
```

then run `pip install -r requirements.txt`.

---

## Configuration

All configuration is **optional**. The library falls back to built-in defaults when
nothing is provided.

### API credentials — environment variables and/or `secrets.json`

Only API keys for cloud providers belong here. Server URLs for local providers (Ollama,
LM Studio, llama.cpp) are **not** credentials — register them via `configure_provider()`.

**Precedence** (highest → lowest):
1. Environment variables (`os.environ`) — shell, Docker, CI/CD
2. `secrets.json` in the consuming project's root — for local development

Supported keys (see [`secrets.json.example`](secrets.json.example) and [`.env.example`](.env.example)):

| `secrets.json` key    | Environment variable  |
|-----------------------|-----------------------|
| `google_api_key`      | `GOOGLE_API_KEY`      |
| `anthropic_api_key`   | `ANTHROPIC_API_KEY`   |
| `openai_api_key`      | `OPENAI_API_KEY`      |
| `mistral_api_key`     | `MISTRAL_API_KEY`     |
| `cohere_api_key`      | `COHERE_API_KEY`      |
| `meta_api_key`        | `META_API_KEY` or `LLAMA_API_KEY` |
| `groq_api_key`        | `GROQ_API_KEY`        |
| `xai_api_key`         | `XAI_API_KEY`         |

**Via environment variables** (shell / CI/CD / Docker):
```bash
export GOOGLE_API_KEY=your-google-api-key-here
```

**Via `secrets.json`** (copy from [`secrets.json.example`](secrets.json.example)):
```json
{
    "google_api_key": "your-google-api-key-here",
    "anthropic_api_key": "your-anthropic-api-key-here",
    "openai_api_key": "your-openai-api-key-here",
    "mistral_api_key": "your-mistral-api-key-here",
    "cohere_api_key": "your-cohere-api-key-here",
    "meta_api_key": "your-llama-api-key-here",
    "groq_api_key": "your-groq-api-key-here",
    "xai_api_key": "your-xai-api-key-here"
}
```

> **Add `secrets.json` to each consuming project's `.gitignore`** to avoid leaking credentials.

### Provider settings — `configure_provider()`

Controls server URLs, timeouts, rate-limit delays, and provider-specific options such
as `context_size` or `visual_token_budget`. Call this once at application startup for
any setting that should not change per-call:

```python
from unified_ai_client import configure_provider

# Ollama on a remote host with a custom context window
configure_provider(
    "ollama",
    url="http://192.168.1.5:11434",
    timeout=240,
    context_size=8000,
    visual_token_budget=1120,
)

# Google with rate-limiting delay
configure_provider("google", sleep_time=3)
```

If `configure_provider()` is never called, built-in defaults are used automatically
(Ollama: `http://localhost:11434`, timeout 300 s; cloud providers: their standard
endpoints). Per-call overrides via `extra_options` in `call_ai()` always take
precedence over values registered here.

Known top-level fields: `url`, `timeout`, `sleep_time`. Everything else is treated as
a provider-specific setting and forwarded to the API payload (e.g. `context_size` →
`num_ctx` in Ollama, `max_tokens` → `max_output_tokens` in cloud providers).

---

## Usage

All features are exported from the top-level package namespace.

### Basic Generation

```python
from unified_ai_client import call_ai, silence_sdks

silence_sdks()   # Suppress verbose SDK loggers at startup

response = call_ai(
    provider="ollama",
    model="gemma4:e2b",
    prompt="List 3 primary colors.",
)
print(response.text)
```

### Multi-turn Chat, Temperature, and JSON Output

```python
from unified_ai_client import call_ai

chat_history = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there! How can I assist you?"},
]

response = call_ai(
    provider="google",
    model="gemini-2.5-pro",
    prompt="Explain quantum computing in one sentence as JSON.",
    messages=chat_history,
    temperature=0.2,
    format_json=True,
)
print(response.text)
print(f"Tokens used: In={response.input_tokens}, Out={response.output_tokens}")
```

### Multimodal Input (Files)

Pass a local file path (or list of paths) via `file_path`. The library handles all
encoding, upload, and fallback logic depending on provider and file type:

```python
# Single image
response = call_ai(
    provider="ollama",
    model="gemma4:e2b",
    prompt="Describe this diagram.",
    file_path="C:/docs/figures/diagram.png",
)

# Multiple files (image + text document)
response = call_ai(
    provider="google",
    model="gemini-2.5-pro",
    prompt="Summarize the attached report.",
    file_path=["report.pdf", "notes.md"],
)
```

Supported file types per provider:

| Provider | Images | Audio | Text files | PDFs |
|---|---|---|---|---|
| `google` | ✅ Upload | ✅ Upload | ✅ Upload | ✅ Upload |
| `ollama` | ✅ base64 | ✅ base64 (multimodal models) | ✅ Inlined | ❌ Not supported |
| `openai` | ✅ base64 | ✅ base64 | ✅ Inlined | ✅ base64 |
| `anthropic` | ✅ base64 | ❌ Not supported | ✅ Inlined | ✅ base64 |
| `mistral` | ✅ base64 | ❌ Not supported | ✅ Inlined | ❌ Not supported |
| `cohere` | ✅ base64 | ❌ Not supported | ✅ Inlined | ❌ Not supported |
| `meta` | ✅ base64 | ❌ Not supported | ✅ Inlined | ❌ Not supported |
| `groq` | ✅ base64 | ❌ Not supported | ✅ Inlined | ❌ Not supported |
| `xai` | ✅ base64 | ❌ Not supported | ✅ Inlined | ❌ Not supported |
| `lmstudio` / `llamacpp` | ✅ base64 | ❌ Not supported | ✅ Inlined | ❌ Not supported |
| `script` | Path passed | Path passed | Path passed | Path passed |

### Thinking / Reasoning

```python
response = call_ai(
    provider="google",
    model="gemini-2.5-pro",
    prompt="What is the best sorting algorithm and why?",
    thinking=True,
)
print(response.text)          # Final answer
print(response.reasoning_text)  # Thinking process
```

> [!NOTE]
> * **Supported Providers**: `google` (Gemini 2.5/3.x), `ollama` (via `think` option), and `anthropic` (adaptive thinking) support explicit control over the `thinking` parameter (`True` or `False`).
> * **OpenAI-Compatible Providers**: `openai`, `mistral`, `cohere`, `meta`, `groq`, `xai`, `lmstudio`, and `llamacpp` do not support explicit API control over thinking. For these providers, the parameter always behaves as `"default"` (leaving control to the model/server).
> * **Reasoning Extraction**: Regardless of the `thinking` parameter value, `reasoning_text` will always be populated if the model returns a reasoning trace (e.g. `reasoning_content` for OpenAI or `thinking` blocks parsed by Ollama/Anthropic/Google).

### Model Pre-loading (Ollama only)

Pre-loads a model into GPU/CPU memory and registers its settings so they
propagate automatically into all subsequent `call_ai()` calls:

```python
from unified_ai_client import preload_model

preload_model(
    provider="ollama",
    model="gemma4:e2b",
    keep_alive="15m",
    context_size=8000,                       # allocates VRAM with correct num_ctx
    extra_options={"visual_token_budget": 1120},  # any other provider-specific setting
)
```

Passing `context_size` here (rather than in each `call_ai()` call) is important:
Ollama allocates the context window once at preload time. If a different `num_ctx`
value arrives at the first `call_ai()`, Ollama reloads the model and reallocates VRAM.

For providers that do not support preloading (Google, Anthropic, OpenAI, etc.) the
warm-up part is a no-op, but any provided `context_size` / `extra_options` are still
registered and will apply to `call_ai()` calls.

### Text Embeddings

Generate text embedding vectors using supported providers (`openai`, `ollama`, `mistral`, `cohere`, `xai`, `lmstudio`, `llamacpp`).

Example using local Ollama model:

```python
from unified_ai_client import get_embedding

vector = get_embedding(
    provider="ollama",
    model="bge-m3",
    text="I like rusty spoons.",
)
print(f"Embedding vector length: {len(vector)}")
print(f"First 5 coordinates: {vector[:5]}")
```

Example using Mistral cloud model:

```python
from unified_ai_client import get_embedding

vector = get_embedding(
    provider="mistral",
    model="mistral-embed",
    text="I like rusty spoons.",
)
```

### Script Provider

Run any script implementing the [`LLM_Behaviour_Interface.md`](LLM_Behaviour_Interface.md) interface:

```python
response = call_ai(
    provider="script",
    model="/path/to/my_scripts/my_llm.py",
    prompt="Summarize this.",
)
```

### Tool Calling (Function Calling)

The library provides minimal, transport-level tool calling support. It passes
tool definitions to the model and returns any tool calls the model requests.
The **execution loop** is the consumer's responsibility.

#### Defining Tools

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

#### Two-Turn Example

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
    # current prompt as a new user message — the consumer controls the history.
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

#### Provider Compatibility

All providers support tool calling. Whether a specific model will actually use
tool calling depends on its training, not the provider.

| Provider | Tool Calling | Notes |
|---|---|---|
| `google` | ✅ | `FunctionDeclaration` / `function_call` parts |
| `anthropic` | ✅ | `input_schema` format / `tool_use` content blocks |
| `openai` | ✅ | Standard OpenAI format |
| `mistral` | ✅ | Inherited from `openai_compat` |
| `cohere` | ✅ | Inherited from `openai_compat` |
| `meta` | ✅ | Inherited from `openai_compat` |
| `groq` | ✅ | Inherited from `openai_compat` |
| `xai` | ✅ | Inherited from `openai_compat` |
| `lmstudio` | ✅ | Inherited from `openai_compat` |
| `llamacpp` | ✅ | Inherited from `openai_compat` |
| `ollama` | ✅ | OpenAI-compatible format via `/api/chat` |
| `script` | ✅ | Extended JSON protocol (see `LLM_Behaviour_Interface.md`) |

> [!NOTE]
> For `ollama`, tool calling requires models specifically trained for it (e.g. `gemma4`, `qwen3`, `llama3.1`). The library sends the tool definitions regardless; if the model ignores them, `AiResponse.tool_calls` will be empty and you will receive a plain text response.

### Resource Cleanup

`atexit` cleanup is registered automatically on first call. For eager cleanup:

```python
from unified_ai_client import cleanup

try:
    response = call_ai(...)
finally:
    cleanup()   # Deletes uploaded Google AI files
```

---

## API Reference

### Functions

#### `call_ai`
The primary function to execute text generation requests across any provider.

```python
def call_ai(
    provider: str,
    model: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    messages: list[dict] | None = None,
    file_path: str | list[str] | None = None,
    temperature: float = 0.7,
    thinking: bool | str = "default",
    format_json: bool = False,
    timeout: int = 300,
    max_retries: int = 3,
    retry_base_delay: float = 5.0,
    top_k: int = 64,
    top_p: float = 0.95,
    max_tokens: int | None = None,
    sleep_time: int | None = None,
    extra_options: dict | None = None,
    tools: list[ToolDefinition] | None = None,
    tool_results: list[ToolResult] | None = None,
) -> AiResponse:
```

* **Parameters:**
  * `provider` (`str`): The name of the AI provider (`"ollama"`, `"google"`, `"openai"`, `"anthropic"`, `"lmstudio"`, `"llamacpp"`, `"script"`).
  * `model` (`str`): The target model identifier (or script path for `"script"` provider).
  * `prompt` (`str`): The user prompt.
  * `system_prompt` (`str | None`, optional): System instructions. Defaults to `None`.
  * `messages` (`list[dict] | None`, optional): A list of previous chat messages in the format `[{"role": "user" | "assistant", "content": "..."}]`. Messages can also include an optional `"files"` key containing a list of local file paths. Defaults to `None`.
  * `file_path` (`str | list[str] | None`, optional): Local file path(s) to attach for multimodal prompts. Supports images, audio, PDFs, and text files. Defaults to `None`.
  * `temperature` (`float`, optional): Sampling temperature. Defaults to `0.7`.
  * `thinking` (`bool | str`, optional): Enables extended reasoning/thinking mode (`True`/`False`), or delegates to the provider's default behavior (`"default"`). Defaults to `"default"`.
  * `format_json` (`bool`, optional): Forces the model to respond in valid JSON format. Defaults to `False`.
  * `timeout` (`int`, optional): Network timeout in seconds. Defaults to `300`.
  * `max_retries` (`int`, optional): Number of retry attempts on network/rate-limit failures. Defaults to `3`.
  * `retry_base_delay` (`float`, optional): Initial backoff delay for exponential retries in seconds. Defaults to `5.0`.
  * `top_k` (`int`, optional): Sampling parameter top_k. Defaults to `64`.
  * `top_p` (`float`, optional): Sampling parameter top_p. Defaults to `0.95`.
  * `max_tokens` (`int | None`, optional): Maximum number of tokens in the response. Maps to `num_predict`, `max_output_tokens`, etc. Defaults to `None`.
  * `sleep_time` (`int | None`, optional): Pre-call rate limit sleep delay in seconds. Defaults to `None` (falls back to config `sleep_time`).
  * `extra_options` (`dict | None`, optional): Optional dict of arbitrary provider-specific options merged into the API payload at call time. Override any config-level defaults for the same key. Defaults to `None`.
  * `tools` (`list[ToolDefinition] | None`, optional): Tool definitions the model may call. When provided, the model may respond with `AiResponse.tool_calls` instead of (or in addition to) text. Defaults to `None`.
  * `tool_results` (`list[ToolResult] | None`, optional): Results of previously requested tool calls. Pass these on the follow-up call to continue the conversation after executing the tools. Defaults to `None`.
* **Returns:** `AiResponse` dataclass containing response text, token metrics, optional reasoning text, and any tool calls requested by the model.

---

#### `get_embedding`
Generates a numerical embedding vector for the provided text.

```python
def get_embedding(
    provider: str,
    model: str,
    text: str,
) -> list[float]:
```

* **Parameters:**
  * `provider` (`str`): The AI provider supporting embeddings (e.g., `"ollama"`).
  * `model` (`str`): The embedding model name (e.g., `"bge-m3"`).
  * `text` (`str`): The input text to embed.
* **Returns:** A list of float numbers (`list[float]`) representing the high-dimensional vector.

---

#### `preload_model`
Pre-loads a model into system memory and registers its settings for all
subsequent `call_ai()` calls. Currently only Ollama performs an actual warm-up;
all other providers treat the warm-up as a no-op but still register the settings.

```python
def preload_model(
    provider: str,
    model: str,
    keep_alive: str = "15m",
    context_size: int | None = None,
    extra_options: dict | None = None,
) -> None:
```

* **Parameters:**
  * `provider` (`str`): Provider name (e.g. `"ollama"`).
  * `model` (`str`): The model identifier to preload.
  * `keep_alive` (`str`, optional): How long to keep the model resident in memory
    (e.g. `"15m"`, `"1h"`, `"0"`). Defaults to `"15m"`. Ollama-specific.
  * `context_size` (`int`, optional): Context window size in tokens. Ollama maps
    this to `num_ctx` in the API payload. Registered via `configure_provider()`
    so it persists across all `call_ai()` calls without needing to be repeated.
    Pass it **here** rather than in per-call `extra_options` to prevent Ollama
    from reloading the model with a different context window mid-session.
  * `extra_options` (`dict`, optional): Additional provider-specific settings
    (e.g. `{"visual_token_budget": 1120}`). Merged with any previously registered
    settings and persisted via `configure_provider()`.

---

#### `configure_provider`
Registers or updates provider-specific configuration programmatically. Calls are
**merge-based**: only the fields explicitly passed are updated; previously
registered values for other fields are preserved.

Known top-level fields (`url`, `timeout`, `sleep_time`) are stored as typed
attributes on `ProviderConfig`. All other keyword arguments are collected into
`extra_options` and forwarded as provider-specific settings (e.g. `context_size`
→ `num_ctx` in Ollama payloads, `visual_token_budget`, `disable_safety`, etc.).

Thread-safe. Concurrent calls for different provider names are fully safe.
Calling it for the same name from multiple threads is safe but last-writer-wins.

```python
def configure_provider(name: str, **kwargs) -> None:
```

* **Parameters:**
  * `name` (`str`): The provider name (e.g. `"ollama"`, `"google"`).
  * `**kwargs`: Configuration values. Known fields: `url`, `timeout`, `sleep_time`.
    All other keys go into `extra_options` as provider-specific settings.

```python
# Examples
from unified_ai_client import configure_provider

configure_provider("ollama",
                   url="http://192.168.1.5:11434",
                   timeout=240,
                   context_size=8000,
                   visual_token_budget=1120)
configure_provider("google", sleep_time=3)
```

---

#### `silence_sdks`
Suppresses verbose debug and info log messages from downstream provider SDKs (such as `google_genai`, `httpx`, and `httpcore`).

```python
def silence_sdks() -> None:
```

---

#### `cleanup`
Triggers remote and local resource purging across all active cached providers. For example, it deletes uploaded Gemini files from the Google remote cloud cache to release quota. It is automatically registered to run at process exit via `atexit`, but can be called manually.

```python
def cleanup() -> None:
```

---

#### `load_secrets`
Loads API credentials from environment variables or a local JSON file. Environment variables take priority over JSON.

```python
def load_secrets(
    project_root: str,
    filename: str = "secrets.json",
) -> dict[str, str]:
```

* **Parameters:**
  * `project_root` (`str`): The root directory of the project.
  * `filename` (`str`, optional): The name of the secrets JSON file. Defaults to `"secrets.json"`.
* **Returns:** A dictionary of loaded credentials.

---

#### `load_config`
Loads configurations from a JSON file into a dataclass type, handling defaults and dynamic option collecting.

```python
def load_config(
    config_path: str,
    dataclass_type: type[Any],
    section: str | None = None,
) -> Any:
```

* **Parameters:**
  * `config_path` (`str`): The path to the configuration JSON file.
  * `dataclass_type` (`type`): The dataclass type to load configuration into.
  * `section` (`str | None`, optional): A subsection key to load from the JSON. Defaults to `None`.
* **Returns:** An instance of `dataclass_type`.

---

### Dataclasses

#### `AiResponse`
The standard response object returned by `call_ai()`.

* **Fields:**
  * `text` (`str`): The final generated response text.
  * `input_tokens` (`int`): The number of prompt/input tokens consumed. Defaults to `0`.
  * `output_tokens` (`int`): The number of completion/output tokens generated. Defaults to `0`.
  * `reasoning_tokens` (`int`): The number of tokens spent on internal reasoning/thinking. Defaults to `0`.
  * `reasoning_text` (`str`): The full reasoning/thinking transcript if produced by the model. Defaults to `""`.
  * `reasoning_is_summary` (`bool`): `True` when `reasoning_text` is a summary the provider generated from the model's actual chain of thought, rather than the raw trace. Defaults to `False`.
  * `tool_calls` (`list[ToolCall]`): Tool calls requested by the model. Empty list if the model responded with text directly. Defaults to `[]`.

---

#### `ToolDefinition`
Describes a function the model can call.

* **Fields:**
  * `name` (`str`): Unique function name (must be a valid identifier).
  * `description` (`str`): Human-readable description of what the tool does and when to use it.
  * `parameters` (`dict`): JSON Schema object describing the function's parameters (type `"object"` with `"properties"`).

---

#### `ToolCall`
A tool call requested by the model, found in `AiResponse.tool_calls`.

* **Fields:**
  * `id` (`str`): Provider-assigned identifier. Pass back in `ToolResult.call_id`.
  * `name` (`str`): Name of the tool function to execute.
  * `arguments` (`dict`): Parsed argument dictionary for the tool function.

---

#### `ToolResult`
The result of a tool execution, passed back to `call_ai()` via `tool_results`.

* **Fields:**
  * `call_id` (`str`): The `id` of the `ToolCall` this result corresponds to.
  * `name` (`str`): Name of the tool function that was executed (required by some providers for result routing).
  * `content` (`str`): String result of the tool execution.

---

## Design Concepts: Caching and Lifecycle

`UnifiedAiClient` utilises a hybrid architectural design that separates stateful
infrastructure configuration from stateless call parameters.

### 1. General Settings (Stateful Configuration)
The Python client is **stateful** regarding its connection infrastructure. Provider
instances are created and cached inside the global module scope upon their first
invocation:
* Settings like `url`, network `timeout`, rate-limiting `sleep_time`, and
  provider-specific options such as `context_size` are registered once via
  `configure_provider()` at application startup.
* Local settings like Ollama's `context_size` (`num_ctx`) are configuration-level
  settings because changing them dynamically at call-time would force local model
  reloads or re-allocation of GPU VRAM.
* API credentials (API keys) loaded from `secrets.json` or system variables are cached.
* Thread-safe singletons avoid unnecessary config loading and connection
  re-initialisation on high-frequency requests.

### 2. Call-Time Parameters (Stateless Requests)
The conversation logic is **stateless**. `UnifiedAiClient` does not track chat history,
context, or previous requests:
* Parameters like `temperature`, `top_k`, `top_p`, `max_tokens` (mapped automatically
  to `num_predict`, `max_output_tokens`, etc.), and JSON formatting flags can be
  modified per-call in `call_ai()`.
* Consuming projects maintain their own state (e.g. accumulating message history in
  `messages`) and must pass the full history to each call.
* Call-time overrides can be passed via the `extra_options` dictionary to customise
  behaviour temporarily. They take precedence over values registered via
  `configure_provider()` for the same key.

---

## Architecture

```
unified_ai_client/
├── __init__.py           # Public API exports
├── client.py             # call_ai(), configure_provider() router + provider cache
├── models.py             # AiRequest, AiResponse, ProviderConfig dataclasses
├── file_utils.py         # classify_file, encode_file_base64, inline_text_attachments, ...
├── config.py             # load_secrets(), load_config() (utility)
├── retry.py              # Exponential backoff
├── silence.py            # silence_sdks()
└── providers/
    ├── base.py           # BaseProvider ABC
    ├── google.py         # Google AI (Gemini) — upload caching, thinking, cleanup
    ├── ollama.py         # Ollama — urllib, images[], audio, context_size
    ├── openai_compat.py  # Base for OpenAI-compatible APIs
    ├── openai.py         # OpenAI — native audio + PDF blocks
    ├── anthropic.py      # Anthropic — urllib, image/document blocks, thinking
    ├── mistral.py        # Mistral — OpenAI-compatible
    ├── cohere.py         # Cohere — OpenAI-compatible
    ├── meta.py           # Meta (Llama API) — OpenAI-compatible
    ├── groq.py           # Groq — OpenAI-compatible
    ├── xai.py            # xAI — OpenAI-compatible
    ├── lmstudio.py       # LM Studio — OpenAI-compatible
    ├── llamacpp.py       # llama.cpp — OpenAI-compatible
    └── script.py         # External script via stdin/stdout JSON
```

All code follows PEP 8, Google-style docstrings, `from __future__ import annotations`,
and modern type hints throughout.
