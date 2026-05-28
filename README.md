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
unified-ai-client @ git+https://github.com/Kuig/UnifiedAiClient.git@v0.1.0
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
    include_reasoning=True,   # Return the thinking transcript
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

### Script Provider

Run any script implementing the `LLM_Behaviour_Interface.md` interface:

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
