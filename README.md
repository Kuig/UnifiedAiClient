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

**Via environment variables** (shell / CI/CD / Docker):
```bash
export GOOGLE_API_KEY=your-google-api-key-here
```

**Via `secrets.json`** (copy from [`secrets.json.example`](secrets.json.example)):
```json
{
    "google_api_key": "your-google-api-key-here",
    "anthropic_api_key": "your-anthropic-api-key-here",
    "openai_api_key": "your-openai-api-key-here"
}
```

> **Add `secrets.json` to each consuming project's `.gitignore`** to avoid leaking credentials.

### Provider settings — `config.json`

Controls server URLs, timeouts, rate-limit delays, context sizes, etc.
Edit `UnifiedAiClient/config.json` directly (or copy from [`config.json.example`](config.json.example)):

```json
{
    "ollama":    { "url": "http://localhost:11434", "timeout": 120, "keep_alive": "15m", "context_size": 0 },
    "google":    { "sleep_time": 3, "disable_safety": false, "timeout": 30 },
    "openai":    { "url": "https://api.openai.com", "timeout": 120, "max_tokens": 8192 },
    "anthropic": { "url": "https://api.anthropic.com", "timeout": 120, "max_tokens": 8192 },
    "lmstudio":  { "url": "http://localhost:1234", "timeout": 120, "context_size": 0 },
    "llamacpp":  { "url": "http://localhost:8080", "timeout": 120, "context_size": 0 }
}
```

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

### Model Pre-loading (Ollama only)

```python
from unified_ai_client import preload_model

preload_model(
    provider="ollama",
    model="gemma4:e2b",
    keep_alive="15m",
)
```

### Text Embeddings (Ollama only)

Generate text embedding vectors using local models such as `bge-m3`:

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
    thinking: bool = False,
    format_json: bool = False,
    timeout: int = 120,
    max_retries: int = 3,
    retry_base_delay: float = 5.0,
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
  * `thinking` (`bool`, optional): Enables extended reasoning/thinking mode (e.g. on supported models). Defaults to `False`.
  * `format_json` (`bool`, optional): Forces the model to respond in valid JSON format. Defaults to `False`.
  * `timeout` (`int`, optional): Network timeout in seconds. Defaults to `120`.
  * `max_retries` (`int`, optional): Number of retry attempts on network/rate-limit failures. Defaults to `3`.
  * `retry_base_delay` (`float`, optional): Initial backoff delay for exponential retries in seconds. Defaults to `5.0`.
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
