# UnifiedAiClient

`UnifiedAiClient` is a shared Python library providing a unified AI provider abstraction.
It routes all AI calls through a single `call_ai()` interface, hiding provider-specific 
encoding, upload, retry, and error-handling details from consuming projects.

Supported providers:

| Provider | Type | Notes |
|---|---|---|
| `"google"` | Cloud | google-genai SDK |
| `"anthropic"` | Cloud | urllib only, no SDK |
| `"openai"` | Cloud | urllib only, no SDK |
| `"mistral"` | Cloud | urllib only, no SDK |
| `"cohere"` | Cloud | urllib only, no SDK |
| `"meta"` | Cloud | Llama API (llama-api.com), urllib only |
| `"groq"` | Cloud | urllib only, no SDK |
| `"xai"` | Cloud | urllib only, no SDK |
| `"ollama"` | Local | urllib only, no SDK |
| `"lmstudio"` | Local | OpenAI-compatible API |
| `"llamacpp"` | Local | OpenAI-compatible API |
| `"script"` | External | stdin/stdout JSON protocol |

---

## Installation

**Local development** (when you're also editing the library):

```bash
# Activate your project's .venv first, then run:
pip install -e /path/to/UnifiedAiClient
```

**Production / other machines** — declare in your project's `requirements.txt`:

```text
unified-ai-client @ git+https://github.com/Kuig/UnifiedAiClient.git@v0.1.1
```

then run `pip install -r requirements.txt`.

---

## Configuration

All configuration is **optional**. The library falls back to built-in defaults when
nothing is provided.

### Where do the files live?

| File | Lives in | Created by | Purpose |
|---|---|---|---|
| `secrets.json` | **Each consuming project's root** | You, per-project | Cloud API keys (local dev) |
| `config.json` | **The library directory** (`UnifiedAiClient/`) | You, once | Provider URLs, timeouts, etc. |

**`secrets.json`** is looked up at runtime via `os.getcwd()`, so it must be in whichever
directory you launch your script from — which is conventionally the consuming project's
root. Each project has its own `secrets.json` with its own API keys.

**`config.json`** is the library's shared configuration. It lives once inside the
`UnifiedAiClient/` directory and applies to all consuming projects. If a project needs
different provider settings (e.g. a different Ollama port), it can call
`set_default_config("path/to/its/own/config.json")` at startup to override.

```
/your-workspace/
├── UnifiedAiClient/
│   ├── config.json          ← shared library config (URLs, timeouts, etc.)
│   ├── config.json.example  ← template to copy and edit
│   └── secrets.json.example ← template to copy and edit
│
├── AppAlpha/
│   ├── secrets.json         ← AppAlpha's API keys (gitignored)
│   ├── my_config.json       ← Custom AppAlpha's library config (set_default_config)
│   └── ...
│
└── AppBeta/
    ├── secrets.json         ← AppBeta's API keys (gitignored)
    └── ...
```

---

### API credentials — environment variables and/or `secrets.json`

Only API keys for cloud providers belong here. Server URLs for local providers (Ollama,
LM Studio, llama.cpp) are **not** credentials — put them in `config.json`.

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

### Provider settings — `config.json`

Controls server URLs, timeouts, rate-limit delays, and other provider-specific settings.
Edit `UnifiedAiClient/config.json` directly (or copy from [`config.json.example`](config.json.example)):

```json
{
    "ollama":    { "url": "http://localhost:11434", "timeout": 120, "keep_alive": "15m", "context_size": 0 },
    "google":    { "sleep_time": 3, "disable_safety": false, "timeout": 30 },
    "openai":    { "url": "https://api.openai.com", "timeout": 120, "max_tokens": 8192 },
    "anthropic": { "url": "https://api.anthropic.com", "timeout": 120, "max_tokens": 8192 },
    "mistral":   { "url": "https://api.mistral.ai", "timeout": 120 },
    "cohere":    { "url": "https://api.cohere.ai/compatibility/v1", "timeout": 120 },
    "meta":      { "url": "https://api.llama-api.com", "timeout": 120 },
    "groq":      { "url": "https://api.groq.com/openai", "timeout": 120 },
    "xai":       { "url": "https://api.x.ai", "timeout": 120 },
    "lmstudio":  { "url": "http://localhost:1234", "timeout": 120, "context_size": 0 },
    "llamacpp":  { "url": "http://localhost:8080", "timeout": 120, "context_size": 0 }
}
```

> **Dynamic Option Loading**: Except for base config fields (`url`, `timeout`, `sleep_time`), any unrecognized keys present in `config.json` for a provider are automatically collected into the provider's `extra_options` dictionary. This prevents payload payload definitions from breaking when updating client settings in `config.json`.

If a consuming project needs its own provider settings, call this once at startup:
```python
from unified_ai_client import set_default_config
set_default_config("path/to/its/own/config.json")
```

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
| `ollama` | ✅ base64 | ✅ base64 (multimodal models) | ✅ Inlined | ⚠️ Inline attempt |
| `openai` | ✅ base64 | ✅ base64 | ✅ Inlined | ✅ base64 |
| `anthropic` | ✅ base64 | ⚠️ Skipped | ✅ Inlined | ✅ base64 |
| `mistral` | ✅ base64 | ⚠️ Inline attempt | ✅ Inlined | ⚠️ Inline attempt |
| `cohere` | ✅ base64 | ⚠️ Inline attempt | ✅ Inlined | ⚠️ Inline attempt |
| `meta` | ✅ base64 | ⚠️ Inline attempt | ✅ Inlined | ⚠️ Inline attempt |
| `groq` | ✅ base64 | ⚠️ Inline attempt | ✅ Inlined | ⚠️ Inline attempt |
| `xai` | ✅ base64 | ⚠️ Inline attempt | ✅ Inlined | ⚠️ Inline attempt |
| `lmstudio` / `llamacpp` | ✅ base64 | ⚠️ Skipped | ✅ Inlined | ⚠️ Inline attempt |
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

```python
from unified_ai_client import preload_model

preload_model(
    provider="ollama",
    model="gemma4:e2b",
    keep_alive="15m",
)
```

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
    timeout: int = 120,
    max_retries: int = 3,
    retry_base_delay: float = 5.0,
    top_k: int = 64,
    top_p: float = 0.95,
    max_tokens: int | None = None,
    sleep_time: int | None = None,
    extra_options: dict | None = None,
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
  * `timeout` (`int`, optional): Network timeout in seconds. Defaults to `120`.
  * `max_retries` (`int`, optional): Number of retry attempts on network/rate-limit failures. Defaults to `3`.
  * `retry_base_delay` (`float`, optional): Initial backoff delay for exponential retries in seconds. Defaults to `5.0`.
  * `top_k` (`int`, optional): Sampling parameter top_k. Defaults to `64`.
  * `top_p` (`float`, optional): Sampling parameter top_p. Defaults to `0.95`.
  * `max_tokens` (`int | None`, optional): Maximum number of tokens in the response. Maps to `num_predict`, `max_output_tokens`, etc. Defaults to `None`.
  * `sleep_time` (`int | None`, optional): Pre-call rate limit sleep delay in seconds. Defaults to `None` (falls back to config `sleep_time`).
  * `extra_options` (`dict | None`, optional): Optional dict of arbitrary provider-specific options merged into the API payload at call time. Override any config-level defaults for the same key. Defaults to `None`.
* **Returns:** `AiResponse` dataclass containing response text, token metrics, and optional reasoning text.

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
Pre-loads a model into system memory (currently only supported by the `"ollama"` provider).

```python
def preload_model(
    provider: str,
    model: str,
    keep_alive: str = "15m",
) -> None:
```

* **Parameters:**
  * `provider` (`str`): Must be `"ollama"`.
  * `model` (`str`): The Ollama model name to preload.
  * `keep_alive` (`str`, optional): How long to keep the model resident in memory (e.g., `"15m"`, `"1h"`, `"0"`). Defaults to `"15m"`.

---

#### `set_default_config`
Overrides the default `config.json` path used by the library. Typically called at application startup.

```python
def set_default_config(path: str) -> None:
```

* **Parameters:**
  * `path` (`str`): The absolute path to the project-specific configuration JSON file.

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
  * `reasoning_text` (`str`): The full reasoning/thinking transcript if produced by the model (supported models may generate reasoning even when thinking is not explicitly requested). Defaults to `""`.

---

## Design Concepts: Caching and Lifecycle

`UnifiedAiClient` utilizes a hybrid architectural design that separates stateful infrastructure configuration from stateless call parameters.

### 1. General Settings (Stateful Configuration)
The Python client is **stateful** regarding its connection infrastructure. Provider instances are created and cached inside the global module scope upon their first invocation:
* Config parameters like `url`, basic network `timeout`, and rate-limiting `sleep_time` are structural settings loaded from `config.json` once.
* Local settings like Ollama's `context_size` (`num_ctx`) are configuration-level settings because changing them dynamically at call-time would force local model reloads or re-allocation of GPU VRAM.
* API credentials (API keys) loaded from `secrets.json` or system variables are cached.
* Thread-safe singletons avoid unnecessary config loading and connection re-initialization on high-frequency requests.

> [!WARNING]
> **Concurrent Multi-Configuration Limitation**:
> Since configuration settings and provider instance caches are stored in global module-level variables, calling `set_default_config()` concurrently from different threads within the same process is **not** isolated. Doing so will clear the global cache and mutate the active config path for all threads. If you need to run concurrent calls with different configurations, run them in separate operating system processes (which is the standard case for individual CLI tools or microservices), or pass request-level overrides via `extra_options` in `call_ai()`.

### 2. Call-Time Parameters (Stateless Requests)
The conversation logic is **stateless**. `UnifiedAiClient` does not track chat history, context, or previous requests:
* Parameters like `temperature`, `top_k`, `top_p`, `max_tokens` (mapped automatically to `num_predict`, `max_output_tokens`, etc.), and JSON formatting flags can be modified per-call in `call_ai()`.
* Consuming projects maintain their own state (e.g. accumulating message history in `messages`) and must pass the full history to each call.
* Call-time overrides can be passed via the `extra_options` dictionary to customize behaviour temporarily without mutating the cached provider instance config.

---

## Architecture

```
unified_ai_client/
├── __init__.py           # Public API exports
├── client.py             # call_ai() router + provider cache
├── models.py             # AiRequest, AiResponse, ProviderConfig dataclasses
├── file_utils.py         # classify_file, encode_file_base64, inline_text_attachments, ...
├── config.py             # load_config(), load_secrets()
├── retry.py              # Exponential backoff
├── silence.py            # silence_sdks()
└── providers/
    ├── base.py           # BaseProvider ABC
    ├── google.py         # Google AI (Gemini) — upload caching, thinking, cleanup
    ├── ollama.py         # Ollama — urllib, images[], audio, context_size
    ├── openai_compat.py  # Base for OpenAI-compatible APIs
    ├── openai.py         # OpenAI — native audio + PDF blocks
    ├── anthropic.py      # Anthropic — urllib, image/document blocks, thinking
    ├── lmstudio.py       # LM Studio — OpenAI-compatible
    ├── llamacpp.py       # llama.cpp — OpenAI-compatible
    └── script.py         # External script via stdin/stdout JSON
```

All code follows PEP 8, Google-style docstrings, `from __future__ import annotations`,
and modern type hints throughout.
