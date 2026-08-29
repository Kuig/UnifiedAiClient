# Warm-up, Preloading and Cleanup

Three calls that bracket the work `call_ai()` does: one prepares the channel, one pins a
local model in memory, one releases what was uploaded.

## Warm-up (all providers)

Every provider charges some costs exactly once per process: importing an SDK,
building a client, the DNS and TLS handshake, loading a model, uploading a file.
Without a warm-up, all of it lands on whichever `call_ai()` happens to run first.
That first call then looks slow purely because it went first, which matters as
soon as you are measuring or comparing timings.

```python
from unified_ai_client import warm_up

warm_up("google", "gemini-2.5-flash", file_paths=["paper.pdf"])

# The upload, the TLS handshake and the client construction are already paid for.
response = call_ai(
    provider="google",
    model="gemini-2.5-flash",
    prompt="Summarize the attached paper.",
    file_path=["paper.pdf"],
)
```

What each provider does:

| Provider | Strategy | Consumes tokens |
|---|---|---|
| `google` | Builds the client, issues a free `models.get` metadata request, uploads `file_paths` | No |
| `ollama` | Loads the model through Ollama's own warm-up request | No |
| `lmstudio`, `llamacpp` | Sends a one-token completion | No (see the note below) |
| `openai`, `mistral`, `cohere`, `meta`, `groq`, `xai` | Free `GET /v1/models` | No |
| `anthropic` | Free `GET /v1/models` | No |
| `script` | Sends `mode: "warm_up"` to the script | Depends on the script |

> [!IMPORTANT]
> LM Studio and llama.cpp are the one exception to "warm-up is free". Both load
> a model lazily on first inference, so a metadata request would return instantly
> and leave the load cost exactly where `warm_up()` is meant to remove it. They
> therefore send a real one-token completion. Against the local servers these
> providers are built for, this is free. If you have pointed either of them at a
> paid remote endpoint, that request is billable.

`warm_up()` never raises. A failed warm-up is a missed optimisation, not an error, so it
returns `False` and lets the `call_ai()` that follows report the real problem through its
own retries. It is safe to call on every provider without checking first: where there is
nothing to warm up, it is a free no-op. Call `get_provider(...).warm_up(...)` directly if
you want the exception instead.

## Model preloading (Ollama)

`preload_model()` pre-loads a model into GPU/CPU memory and registers its settings so they
propagate automatically into all subsequent `call_ai()` calls:

```python
from unified_ai_client import preload_model

preload_model(
    provider="ollama",
    model="gemma4:e2b",
    keep_alive="15m",
    context_size=8000,                            # allocates VRAM with correct num_ctx
    extra_options={"visual_token_budget": 1120},  # any other provider-specific setting
)
```

Passing `context_size` here (rather than in each `call_ai()` call) is important:
Ollama allocates the context window once at preload time. If a different `num_ctx`
value arrives at the first `call_ai()`, Ollama reloads the model and reallocates VRAM.

For providers that do not support preloading (Google, Anthropic, OpenAI, and the rest)
the warm-up part is a no-op, but any provided `context_size` / `extra_options` are still
registered and will apply to `call_ai()` calls.

## Which of the two to call

| | `preload_model()` | `warm_up()` |
|---|---|---|
| What it prepares | A model in resident memory | The whole channel: client, connection, authentication, and on Google the uploaded files |
| Where it works | Ollama, and scripts that implement it | Every provider |
| Also does | Registers `context_size` / `extra_options` for later calls | Nothing persistent beyond the warmed resources |

## Cleanup

Files uploaded by `warm_up()` go into the same cache `call_ai()` reads from, and are
deleted by `cleanup()` along with every other uploaded file. Warming up does not change
the resource lifecycle.

`atexit` cleanup is registered automatically on the first `call_ai()` or `warm_up()`,
whichever comes first, so files uploaded by a warm-up are cleaned up even in a process
that never reaches a real call. For eager cleanup:

```python
from unified_ai_client import cleanup

try:
    response = call_ai(...)
finally:
    cleanup()   # Deletes uploaded Google AI files
```
