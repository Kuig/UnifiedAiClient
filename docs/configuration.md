# Configuration Reference

Everything that can be configured, key by key. All of it is **optional**: the library
falls back to built-in defaults when nothing is provided.

Configuration is split across two axes that never mix. Credentials are secrets and live in
the environment or in `secrets.json`. Everything else is infrastructure, registered once
per provider through `configure_provider()`.

## API credentials

Only API keys for cloud providers belong here. Server URLs for local providers (Ollama,
LM Studio, llama.cpp) are **not** credentials; register them through
[`configure_provider()`](#provider-settings-configure_provider).

### Precedence

Highest to lowest:

1. Environment variables (`os.environ`): shell, Docker, CI/CD.
2. `secrets.json` in the consuming project's working directory: for local development.

Both sources may coexist. When the same key is present in both, the environment variable
wins, so a deployment can override a checked-out development file without editing it.

Credentials are resolved from `os.getcwd()`, meaning the **consuming project's** working
directory, not the directory this library is installed in.

### Key to environment variable

| `secrets.json` key | Environment variable |
|---|---|
| `google_api_key` | `GOOGLE_API_KEY` |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` |
| `openai_api_key` | `OPENAI_API_KEY` |
| `mistral_api_key` | `MISTRAL_API_KEY` |
| `cohere_api_key` | `COHERE_API_KEY` |
| `meta_api_key` | `META_API_KEY` or `LLAMA_API_KEY` |
| `groq_api_key` | `GROQ_API_KEY` |
| `xai_api_key` | `XAI_API_KEY` |

The local providers (`ollama`, `lmstudio`, `llamacpp`) and `script` take no credentials
and have no row here.

**Via environment variables** (shell / CI/CD / Docker):

```bash
export GOOGLE_API_KEY=your-google-api-key-here
```

**Via `secrets.json`** (copy from [`secrets.json.example`](../secrets.json.example)):

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

See also [`.env.example`](../.env.example) for the environment-variable form.

### Missing credentials, and the proxy escape hatch

A cloud provider called without its key raises `ValueError` naming the key it wants, rather
than sending an unauthenticated request. The one exception is when you set an explicit `url`
via `configure_provider()`: that means you have pointed the adapter somewhere other than the
vendor's own endpoint, so no credential is required. This is what makes it possible to drive
a local OpenAI-compatible server through the `openai` adapter:

```python
configure_provider("openai", url="http://localhost:11434")    # Ollama's /v1 endpoint
call_ai(provider="openai", model="gemma4:12b", prompt="...")  # no OpenAI key needed
```

## Provider settings: `configure_provider()`

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

Calls are **merge-based**: only the fields you pass are updated, and previously registered
values for other fields survive. Each call invalidates the cached provider instance for
that name, so the next call builds a fresh one with the new settings.

Per-call `extra_options` in `call_ai()` always take precedence over values registered here.

### Known top-level fields

These three are typed attributes. Everything else is a provider-specific setting and is
forwarded to the API payload.

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | `str` or `None` | `None` | Base URL of the endpoint. `None` means "use this provider's own default", listed below. An explicit value is always honoured as given. |
| `timeout` | `int` | `300` | Seconds to wait for a response. |
| `sleep_time` | `int` | `0` | Seconds to sleep before a call, for rate limiting. Applied once per `call_ai()`, before the retry loop, so retries do not multiply it. |

### Default endpoint per provider

Used whenever `url` is left unset.

| Provider | Default URL |
|---|---|
| `ollama` | `http://localhost:11434` |
| `lmstudio` | `http://localhost:1234` |
| `llamacpp` | `http://localhost:8080` |
| `openai` | `https://api.openai.com` |
| `anthropic` | `https://api.anthropic.com` |
| `mistral` | `https://api.mistral.ai` |
| `cohere` | `https://api.cohere.ai/compatibility` |
| `meta` | `https://api.llama-api.com` |
| `groq` | `https://api.groq.com/openai` |
| `xai` | `https://api.x.ai` |
| `google` | not applicable, the SDK resolves the endpoint |
| `script` | not applicable, `model` is the script path |

### Provider-specific options

Any keyword that is not `url`, `timeout` or `sleep_time` is collected into
`extra_options` and forwarded to the provider. Unknown keys are not rejected, which is
what lets a provider gain an option without a code change here. The ones with defined
behaviour:

| Option | Providers | Default | Description |
|---|---|---|---|
| `context_size` | `ollama` | unset | Context window in tokens, sent as `num_ctx`. Prefer setting it through `preload_model()`, so Ollama allocates VRAM once instead of reloading on the first call. |
| `keep_alive` | `ollama` | `"15m"` | How long the model stays resident after a call. `"0"` unloads it immediately. |
| `max_tokens` | all | unset | Response length cap, mapped per provider (`num_predict` on Ollama, `max_output_tokens` on Google). Also a per-call argument, which wins. |
| `use_generate` | `ollama` | `False` | Routes to `/api/generate` instead of `/api/chat`. That endpoint carries no tool calling and does not separate the thinking trace, so leave it off unless a model requires it. |
| `disable_safety` | `google` | `False` | Turns off Gemini's safety filters. |
| `upload_poll_timeout` | `google` | `15` | Seconds to wait for an uploaded file to reach the `ACTIVE` state before giving up. |
| `task_type` | `google` | unset | Embedding task type, passed to `embed_content`. |
| `output_dimensionality` | `google` | unset | Embedding vector size, passed to `embed_content`. |
| `top_k`, `top_p` | all | `64` / `0.95` | Sampling parameters. Also per-call arguments, which win. |

Anything else is passed through verbatim to the provider payload. On Ollama that means it
lands inside the `options` object, which is how a model-specific knob such as
`visual_token_budget` reaches Gemma 4 without the library needing to know it exists.

## `config.json`, the file-based fallback

Settings can also be written to a `config.json` shipped inside the library directory,
which is read only for providers that `configure_provider()` was never called for. It
exists for backwards compatibility; new code should configure providers in code.

Resolution order, highest first:

1. `configure_provider()`, called programmatically.
2. `config.json` inside the library directory (git-ignored).
3. The built-in `ProviderConfig` defaults listed above.

[`config.json.example`](../config.json.example) documents the shape and mirrors the
built-in defaults.
