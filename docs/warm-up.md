# Warm-up, Residency and Cleanup

Four calls that bracket the work `call_ai()` does: one prepares the channel, one pins a
local model in memory, one releases a model, one releases everything still held.

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

`warm_up()` also takes `keep_alive` and `timeout`, which is what lets a caller pin a model
for exactly as long as it is needed without reconfiguring the provider for the whole
process:

```python
from unified_ai_client import warm_up, unload_model

warm_up("ollama", "gemma4:12b", keep_alive=-1, timeout=600)
...
unload_model("ollama", "gemma4:12b")
```

`keep_alive` accepts a number of seconds, `0` to unload once idle, `-1` to stay resident
indefinitely, or a Go duration string such as `"15m"`. When it is omitted, whatever was
registered via `configure_provider()` applies, so a warm-up and the calls that follow it
always agree on the residency. Only `ollama` and `script` have a residency concept; the
rest accept the argument and ignore it.

Note that the residency timer is an idle countdown that starts when a request *finishes*,
not a budget for the request. A model serving a call is never evicted mid-wait.

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

## Releasing a model

`unload_model()` is the counterpart to `preload_model()`. On Ollama it sends the request
`ollama stop` sends, expiring the model's idle timer immediately:

```python
from unified_ai_client import unload_model

unload_model("ollama", "gemma4:12b")
```

Doing this through the library rather than through the `ollama` CLI has two advantages:
nothing needs the `ollama` binary on the PATH, and it reaches whichever server the
provider is configured for, including a remote one or a non-default port, where the CLI
would talk to the wrong server.

Like `warm_up()`, it never raises. Failing to free VRAM is not a reason to bring down the
caller, and the next request simply reloads the model.

Its timeout defaults to a deliberately short value rather than to the provider's
configured one. The request goes through the same queue as a generation, so it waits
behind whatever is already running, and an unload that cannot get through must not stall
the process for the minutes a generation timeout allows.

## Which one to call

| | `preload_model()` | `warm_up()` | `unload_model()` |
|---|---|---|---|
| What it does | Pins a model in resident memory | Prepares the whole channel: client, connection, authentication, and on Google the uploaded files | Releases a resident model |
| Where it works | Ollama, and scripts that implement it | Every provider | Ollama, and scripts that implement it |
| Also does | Registers `context_size` / `extra_options` for later calls | Nothing persistent beyond the warmed resources | Drops the model from the set `cleanup()` drains |

## Cleanup

`cleanup()` releases both kinds of resource: remote files uploaded to a provider are
deleted to free cloud quota, and local models still resident are unloaded to free VRAM.
Files uploaded by `warm_up()` go into the same cache `call_ai()` reads from, so warming up
does not change the resource lifecycle.

`atexit` cleanup is registered automatically on the first `call_ai()`, `warm_up()` or
`get_embedding()`, whichever comes first, so a process that only ever warms up still
cleans up after itself. Because it runs at exit, it also covers an unhandled exception and
a Ctrl+C, and that is what makes an indefinite `keep_alive=-1` a safe policy rather than a
leak: whatever happens to the process, the VRAM comes back.

For eager cleanup:

```python
from unified_ai_client import cleanup

try:
    response = call_ai(...)
finally:
    cleanup()   # Deletes uploaded files and unloads resident models
```

Pass `unload_models=False` to release the remote resources but leave the models warm,
which is what a long-lived process wants when it calls `cleanup()` between batches rather
than at the end:

```python
cleanup(unload_models=False)
```
