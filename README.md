# UnifiedAiClient

`UnifiedAiClient` is a shared Python library providing a unified AI provider abstraction.
It routes all AI calls through a single `call_ai()` interface, hiding provider-specific
encoding, upload, retry, and error-handling details from consuming projects.

The only runtime dependency is `google-genai`, and only the Google adapter imports it.
Every other provider speaks HTTP through the standard library. See
[COMPARISON.md](COMPARISON.md) for a feature-by-feature comparison with other libraries.

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

## Documentation

| Document | What it covers |
|---|---|
| [docs/api.md](docs/api.md) | Full API reference: every public function, dataclass and exception. |
| [docs/configuration.md](docs/configuration.md) | Key-by-key reference for credentials and provider settings. |
| [docs/multimodal.md](docs/multimodal.md) | Attaching files, the per-provider support matrix, and how unsupported files are refused. |
| [docs/reasoning.md](docs/reasoning.md) | The `thinking` parameter, what each provider does with it, and reading the trace back. |
| [docs/tool-calling.md](docs/tool-calling.md) | Defining tools, the two-turn exchange, provider compatibility. |
| [docs/warm-up.md](docs/warm-up.md) | `warm_up()`, `preload_model()` and `cleanup()`: preparing a provider and releasing what it holds. |
| [docs/script-protocol.md](docs/script-protocol.md) | The JSON stdin/stdout contract any script must satisfy to act as a provider. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Internal design: the codemap, the layer boundaries, and the invariants. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development install, the test suite, and how to add a provider. |
| [COMPARISON.md](COMPARISON.md) | How this library compares with LiteLLM, aisuite and the OpenAI SDK. |

---

## Installation

**Local development** (when you're also editing the library):

```bash
# Activate your project's .venv first, then run:
pip install -e /path/to/UnifiedAiClient
```

**Production / other machines**: declare in your project's `requirements.txt`:

```text
unified-ai-client @ git+https://github.com/Kuig/UnifiedAiClient.git@v0.4.0
```

then run `pip install -r requirements.txt`.

---

## Configuration

All configuration is **optional**. The library falls back to built-in defaults when
nothing is provided.

**Credentials** for cloud providers come from environment variables, or from a
`secrets.json` in your project's working directory. Environment variables win:

```bash
export GOOGLE_API_KEY=your-google-api-key-here
```

```json
{
    "google_api_key": "your-google-api-key-here",
    "anthropic_api_key": "your-anthropic-api-key-here"
}
```

> **Add `secrets.json` to each consuming project's `.gitignore`** to avoid leaking credentials.

**Everything else** is infrastructure, registered once at startup. Server URLs for local
providers are not credentials and belong here:

```python
from unified_ai_client import configure_provider

configure_provider("ollama", url="http://192.168.1.5:11434", timeout=240, context_size=8000)
configure_provider("google", sleep_time=3)
```

Every credential key, every registrable setting and every built-in default is listed in
[docs/configuration.md](docs/configuration.md).

---

## Usage

All features are exported from the top-level package namespace.

### Basic Generation

```python
from unified_ai_client import call_ai, silence_sdks, set_verbosity

silence_sdks()          # Suppress verbose SDK loggers at startup
set_verbosity("debug")  # See everything the library does, useful during development

response = call_ai(
    provider="ollama",
    model="gemma4:e2b",
    prompt="List 3 primary colors.",
)
print(response.text)
```

See [`set_verbosity`](docs/api.md#set_verbosity) for the full level list.

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

The library holds no conversation state: your project owns the history and passes it in
`messages` on every call.

### Multimodal Input (Files)

Pass a local file path, or a list of paths, via `file_path`. The library handles the
encoding and upload, choosing the native mechanism each provider offers:

```python
response = call_ai(
    provider="google",
    model="gemini-2.5-pro",
    prompt="Summarize the attached report.",
    file_path=["report.pdf", "notes.md"],
)
```

Text files are always accepted. Everything else must have a native mechanism on the
target provider, and where there is none the call raises `UnsupportedFileError` rather
than sending something the model cannot read. Which provider accepts what, and why a
refusal beats a guess, is in [docs/multimodal.md](docs/multimodal.md).

### Thinking / Reasoning

```python
response = call_ai(
    provider="google",
    model="gemini-2.5-pro",
    prompt="What is the best sorting algorithm and why?",
    thinking=True,
)
print(response.text)            # Final answer
print(response.reasoning_text)  # Thinking process
```

`thinking` takes `True`, `False`, or `"default"`, which leaves the decision to the
provider. What each provider does with those three values, and why the parameter is
deliberately this coarse, is in [docs/reasoning.md](docs/reasoning.md).

### Tool Calling

The library provides transport-level tool calling: it passes your `ToolDefinition` list to
the model and returns any `ToolCall` the model requests. The execution loop stays yours.
All providers support it. The two-turn exchange is worked through in
[docs/tool-calling.md](docs/tool-calling.md).

### Warm-up and Preloading

Every provider charges some costs once per process: an SDK import, a TLS handshake, a
model load, a file upload. Without a warm-up, all of it lands on whichever `call_ai()`
runs first.

```python
from unified_ai_client import warm_up

warm_up("google", "gemini-2.5-flash", file_paths=["paper.pdf"])
```

`warm_up()` works on every provider and never raises. On Ollama, `preload_model()` also
pins the model in VRAM with the right context window. Both, plus the `cleanup()` that
releases what they upload, are covered in [docs/warm-up.md](docs/warm-up.md).

### Text Embeddings

Generate text embedding vectors. Every provider implements this except `anthropic`,
which offers no embeddings API of its own and points at a third-party service:

| Provider | Endpoint used |
|---|---|
| `google` | `embed_content` via the SDK (`task_type` and `output_dimensionality` through `configure_provider()`) |
| `ollama` | `/api/embed` |
| `openai`, `mistral`, `cohere`, `meta`, `groq`, `xai`, `lmstudio`, `llamacpp` | `/v1/embeddings` |
| `script` | `mode: "embed"` in the [script protocol](docs/script-protocol.md) |
| `anthropic` | not available, raises `NotImplementedError` |

Implementing the call is not the same as the endpoint being served: whether a given
cloud provider exposes embeddings, and for which models, is up to that provider. Use a
model documented as an embedding model for the provider you picked.

```python
from unified_ai_client import get_embedding

vector = get_embedding(provider="ollama", model="bge-m3", text="I like rusty spoons.")
print(f"Embedding vector length: {len(vector)}")
print(f"First 5 coordinates: {vector[:5]}")

vector = get_embedding(provider="mistral", model="mistral-embed", text="I like rusty spoons.")
```

### Script Provider

Run any script implementing the [script protocol](docs/script-protocol.md):

```python
response = call_ai(
    provider="script",
    model="/path/to/my_scripts/my_llm.py",
    prompt="Summarize this.",
)
```

### Resource Cleanup

`cleanup()` deletes files uploaded to a remote provider. It is registered with `atexit` on
the first `call_ai()` or `warm_up()`, so you normally never call it; see
[docs/warm-up.md](docs/warm-up.md) for eager cleanup.
