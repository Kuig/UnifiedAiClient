# API Reference

Every public symbol of `unified_ai_client`. All of them are importable from the top-level
package namespace.

## Functions

### `call_ai`

Executes a text generation request against any provider.

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
    timeout: int | None = None,
    max_retries: int = 3,
    retry_base_delay: float = 5.0,
    top_k: int | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    sleep_time: int | None = None,
    extra_options: dict | None = None,
    tools: list[ToolDefinition] | None = None,
    tool_results: list[ToolResult] | None = None,
) -> AiResponse:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | required | One of `google`, `anthropic`, `openai`, `mistral`, `cohere`, `meta`, `groq`, `xai`, `ollama`, `lmstudio`, `llamacpp`, `script`. |
| `model` | `str` | required | Model identifier, or the script path for the `script` provider. |
| `prompt` | `str` | required | The user prompt. |
| `system_prompt` | `str \| None` | `None` | System instructions. |
| `messages` | `list[dict] \| None` | `None` | Chat history as `[{"role": "user" \| "assistant", "content": "..."}]`. An entry may also carry a `"files"` key with local file paths to attach to that turn. |
| `file_path` | `str \| list[str] \| None` | `None` | Local file path(s) to attach to the current turn. See [Multimodal Input](multimodal.md). |
| `temperature` | `float` | `0.7` | Sampling temperature. |
| `thinking` | `bool \| str` | `"default"` | `True` / `False` to force reasoning on or off, `"default"` to leave it to the provider. See [Thinking and Reasoning](reasoning.md). |
| `format_json` | `bool` | `False` | Forces the model to respond in valid JSON. |
| `timeout` | `int \| None` | `None` | Network timeout in seconds. `None` uses the timeout registered for the provider via `configure_provider()`, falling back to 300. Since a non-streaming request sends nothing until the answer is complete, this behaves as a deadline for the whole call. |
| `max_retries` | `int` | `3` | Retry attempts on network and rate-limit failures, a timeout included. See the note below. |
| `retry_base_delay` | `float` | `5.0` | Initial backoff delay in seconds, doubled on each attempt. |
| `top_k` | `int \| None` | `None` | Sampling parameter. `None` sends nothing, leaving the provider's own default in place. There is no portable value: OpenAI's Chat Completions API rejects `top_k` as an unknown argument, while Ollama and Anthropic accept it. Ollama applies a house default of 64. |
| `top_p` | `float \| None` | `None` | Sampling parameter. `None` leaves the provider's own default in place. Ollama applies a house default of 0.95. |
| `max_tokens` | `int \| None` | `None` | Response length cap, mapped per provider (`num_predict`, `max_output_tokens`). |
| `sleep_time` | `int \| None` | `None` | Seconds to sleep before the call, for rate limiting. Applied once, not per retry. Falls back to the configured `sleep_time`. |
| `extra_options` | `dict \| None` | `None` | Arbitrary provider-specific options merged into the payload last, overriding config-level values for the same key. |
| `tools` | `list[ToolDefinition] \| None` | `None` | Tools the model may call. See [Tool Calling](tool-calling.md). |
| `tool_results` | `list[ToolResult] \| None` | `None` | Results of previously requested tool calls, for the follow-up turn. When present, `prompt` is **not** re-appended to `messages`. |

A timeout is a retryable failure like any other, so the same request is sent again up to
`max_retries` times with exponential backoff. A call that keeps timing out therefore occupies
a local model for roughly `(max_retries + 1) x timeout` plus the backoff, not for `timeout`.
Pass `max_retries=0` where a deadline has to be a real deadline.

**Returns:** an [`AiResponse`](#airesponse) with the response text, token counts, any
reasoning trace, and any tool calls the model requested.

**Raises:** [`UnsupportedFileError`](#unsupportedfileerror) if the provider cannot carry an
attached file, [`MissingFileError`](#missingfileerror) if a path does not exist,
[`FileDecodeError`](#filedecodeerror) if a text attachment is not valid UTF-8. All three
are raised while the request is being assembled, before anything is sent, and none is
retried.

```python
from unified_ai_client import call_ai

response = call_ai(provider="ollama", model="gemma4:e2b", prompt="List 3 primary colors.")
print(response.text)
```

### `get_embedding`

Generates a numerical embedding vector for the provided text.

```python
def get_embedding(
    provider: str,
    model: str,
    text: str,
) -> list[float]:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | required | Any provider except `anthropic`. |
| `model` | `str` | required | An embedding model of that provider, for example `bge-m3`. |
| `text` | `str` | required | The text to embed. |

**Returns:** the vector as a `list[float]`.

**Raises:** `NotImplementedError` on `anthropic`, which serves no embeddings API of its
own. `RuntimeError` if the provider returns no vector.

### `warm_up`

Pays a provider's one-off costs before the first real call, so they are not charged to
whichever `call_ai()` happens to run first. Never raises: see
[Warm-up, Residency and Cleanup](warm-up.md).

```python
def warm_up(
    provider: str,
    model: str,
    file_paths: str | list[str] | None = None,
    *,
    keep_alive: str | int | None = None,
    timeout: int | None = None,
) -> bool:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | required | Provider name. |
| `model` | `str` | required | Model identifier to warm up. |
| `file_paths` | `str \| list[str] \| None` | `None` | Files to upload ahead of time. Used only by providers with a remote file store (currently `google`) and by scripts that act on it. Ignored elsewhere. |
| `keep_alive` | `str \| int \| None` | `None` | How long the model stays resident once loaded. `None` uses whatever `configure_provider()` registered, so a warm-up and the calls after it agree. Values as in `preload_model`. Honoured by `ollama` and `script`. |
| `timeout` | `int \| None` | `None` | Seconds to wait for the warm-up. `None` uses the provider's configured timeout. Useful when a cold model takes longer to load than a normal request takes to answer. |

**Returns:** `True` if something was actually warmed up, `False` if the provider had
nothing to do or the warm-up failed.

### `preload_model`

Pre-loads a model into system memory and registers its settings for all subsequent
`call_ai()` calls. Ollama, and a script that implements the `preload` mode, perform an
actual load; every other provider treats it as a no-op but still registers the settings.

```python
def preload_model(
    provider: str,
    model: str,
    keep_alive: str | int = "15m",
    context_size: int | None = None,
    extra_options: dict | None = None,
) -> None:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | required | Provider name, for example `"ollama"`. |
| `model` | `str` | required | Model identifier to preload. |
| `keep_alive` | `str \| int` | `"15m"` | How long the model stays resident. A number is seconds, `0` unloads it once idle, `-1` keeps it resident indefinitely; a string is a Go duration such as `"15m"` or `"1h"`. Honoured by `ollama` and `script`, ignored elsewhere. Unlike the two below it is not persisted via `configure_provider()`, being a property of the request rather than of the provider. |
| `context_size` | `int \| None` | `None` | Context window in tokens, mapped to `num_ctx`. Registered so it persists across calls. Pass it here rather than per-call to stop Ollama reloading the model with a different window mid-session. |
| `extra_options` | `dict \| None` | `None` | Further provider-specific settings, for example `{"visual_token_budget": 1120}`. Merged with anything already registered. |

The residency timer is an idle countdown that starts when a request *finishes*, not a
budget for the request: a model serving a call is never evicted mid-wait.

### `unload_model`

Releases a model a local provider is holding resident, freeing its VRAM. The counterpart
to `preload_model`. Only `ollama` and `script` have a residency concept; for every other
provider this is a no-op, because a cloud endpoint holds nothing on the caller's behalf.

The model is dropped from the set `cleanup()` drains, so an explicit unload is not
attempted a second time at exit. Never raises: failing to free VRAM is not a reason to
bring down the caller, and the next request simply reloads the model. Call
`get_provider(...).unload_model(...)` if you want the exception instead.

```python
def unload_model(
    provider: str,
    model: str,
    *,
    timeout: int | None = None,
) -> None:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | required | Provider name, for example `"ollama"`. |
| `model` | `str` | required | Model identifier to release. |
| `timeout` | `int \| None` | `None` | Seconds to wait. `None` uses a deliberately short default rather than the provider's configured timeout: the request queues behind any generation already in flight. |

```python
warm_up("ollama", "gemma4:12b", keep_alive=-1)
...
unload_model("ollama", "gemma4:12b")
```

### `configure_provider`

Registers or updates provider configuration programmatically. Calls are **merge-based**:
only the fields passed are updated, and previously registered values survive. Each call
invalidates the cached provider instance for that name.

Thread-safe. Concurrent calls for different provider names are fully safe; calling it for
the same name from several threads is safe but last-writer-wins.

```python
def configure_provider(name: str, **kwargs: Any) -> None:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Provider name, for example `"ollama"`, `"google"`. |
| `**kwargs` | any | none | Known fields `url`, `timeout` and `sleep_time` become typed attributes on `ProviderConfig`. Every other key goes into `extra_options` and is forwarded as a provider-specific setting. |

The full list of settable options is in the
[Configuration Reference](configuration.md#provider-settings-configure_provider).

```python
from unified_ai_client import configure_provider

configure_provider("ollama",
                   url="http://192.168.1.5:11434",
                   timeout=240,
                   context_size=8000,
                   visual_token_budget=1120)
configure_provider("google", sleep_time=3)
```

### `get_provider`

Resolves, configures and caches the provider instance for a name, and returns it. This is
the same instance `call_ai()` uses.

```python
def get_provider(provider_name: str) -> BaseProvider:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `provider_name` | `str` | required | Provider name. |

**Returns:** the cached [`BaseProvider`](#baseprovider) subclass instance.

**Raises:** `ValueError` for an unknown provider name.

Reach for this when you want a provider's behaviour without the wrapper `call_ai()` puts
around it, for instance `get_provider("google").warm_up(...)` to get the exception that
`warm_up()` deliberately swallows.

### `silence_sdks`

Suppresses verbose debug and info messages from downstream provider SDKs such as
`google_genai`, `httpx` and `httpcore`, without touching the root logger.

```python
def silence_sdks() -> None:
```

### `set_verbosity`

Controls how much of the library's own internal activity is printed to the console. Attaches
(or removes) a console handler on the library's own logger only, never the root logger, so it
cannot duplicate output onto a host application's logging setup or interfere with it. Calling
this again replaces the previous handler instead of stacking a second one. Independent of
`silence_sdks()`, which targets third-party SDK loggers instead: call both for first-party and
third-party debug output together.

```python
def set_verbosity(level: str) -> None:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `level` | `str` | required | One of `"silent"`, `"error"`, `"warning"`, `"debug"`. Case-insensitive. |

| Level | Shows |
|---|---|
| `"silent"` | Nothing, ever. |
| `"error"` | `ERROR` and above. |
| `"warning"` | `WARNING` and above. |
| `"debug"` | Everything, including fine-grained per-file and per-block detail. |

**Raises:** `ValueError` for an unknown level.

Console lines are prefixed with `"UAC :: "` so they stay attributable when interleaved with the
consuming project's own output.

```python
from unified_ai_client import call_ai, set_verbosity

set_verbosity("debug")   # see everything during development
response = call_ai(provider="ollama", model="gemma4:12b", prompt="Hello")
```

### `cleanup`

Releases what every cached provider is holding. Two kinds of resource: remote ones, such
as files uploaded to Gemini, which are deleted to free cloud quota; and local models still
resident, which are unloaded to free VRAM.

Registered to run at process exit via `atexit` on the first `call_ai()`, `warm_up()` or
`get_embedding()`, so it also covers an unhandled exception and a Ctrl+C. That is what
makes an indefinite `keep_alive=-1` a safe policy rather than a leak. Callable manually
for eager cleanup.

```python
def cleanup(*, unload_models: bool = True) -> None:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `unload_models` | `bool` | `True` | Whether to also unload local models. Pass `False` to release the remote resources but leave the models warm, for a caller that runs this mid-session. |

### `load_secrets`

Loads API credentials from environment variables and an optional JSON file. Environment
variables take priority. Used internally when a provider is first built; exposed for
consumers that need the same resolution.

```python
def load_secrets(
    project_root: str,
    filename: str = "secrets.json",
) -> dict[str, str]:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `project_root` | `str` | required | Directory to look for the secrets file in. |
| `filename` | `str` | `"secrets.json"` | Name of the secrets file. |

**Returns:** a `dict[str, str]` of snake_case key to value, empty if neither source
provides anything. The key names are listed in the
[Configuration Reference](configuration.md#key-to-environment-variable).

### `load_config`

Loads a JSON config file into a dataclass, falling back to the dataclass defaults when the
file is missing or malformed. Keys that do not match a field are collected into
`extra_options` if the dataclass has one.

```python
def load_config(
    config_path: str,
    dataclass_type: type[Any],
    section: str | None = None,
) -> Any:
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config_path` | `str` | required | Path to the JSON file. |
| `dataclass_type` | `type` | required | Target dataclass. Every field must have a default. |
| `section` | `str \| None` | `None` | Key to extract from a nested JSON before parsing. |

**Returns:** an instance of `dataclass_type`.

## Dataclasses

### `AiResponse`

The standard response object returned by `call_ai()`.

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | required | The generated response text. |
| `input_tokens` | `int` | `0` | Prompt tokens consumed. |
| `output_tokens` | `int` | `0` | Completion tokens generated. |
| `reasoning_tokens` | `int` | `0` | Tokens spent on internal reasoning. `0` where the provider does not report it. On Ollama the figure is estimated, not reported. |
| `reasoning_text` | `str` | `""` | The reasoning transcript, when the model returned one. Populated whatever `thinking` was set to. |
| `reasoning_is_summary` | `bool` | `False` | `True` when `reasoning_text` is a provider-generated summary of the chain of thought rather than the raw trace. Google summarises; the others do not. |
| `tool_calls` | `list[ToolCall]` | `[]` | Tool calls the model requested. Empty when it answered with text. |

### `ToolDefinition`

Describes a function the model can call.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Unique function name. Must be a valid identifier. |
| `description` | `str` | required | What the tool does and when to use it. Clear descriptions improve model accuracy. |
| `parameters` | `dict` | required | JSON Schema object describing the parameters: type `"object"` with a `"properties"` field. |

### `ToolCall`

A tool call requested by the model, found in `AiResponse.tool_calls`.

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required | Provider-assigned identifier. Pass it back in `ToolResult.call_id`. Providers that return no id get a generated fallback. |
| `name` | `str` | required | Name of the tool function to execute. |
| `arguments` | `dict` | required | Parsed argument dictionary for that function. |

### `ToolResult`

The result of a tool execution, passed back through `call_ai(tool_results=...)`.

| Field | Type | Default | Description |
|---|---|---|---|
| `call_id` | `str` | required | The `id` of the `ToolCall` this answers. |
| `name` | `str` | required | Name of the tool function that ran. Some providers need it to route the result. |
| `content` | `str` | required | String result of the execution. |

### `ProviderConfig`

The stateful settings of one provider, as registered by `configure_provider()`. Consumers
rarely construct this directly; the field reference lives in the
[Configuration Reference](configuration.md#known-top-level-fields).

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | `str \| None` | `None` | Endpoint base URL. `None` means "use this provider's own default". |
| `timeout` | `int` | `300` | Seconds to wait for a response. |
| `sleep_time` | `int` | `0` | Seconds to sleep before a call, for rate limiting. |
| `extra_options` | `dict` | `{}` | Provider-specific settings, including every unrecognised key. |

### `AiRequest`

The normalised request handed to a provider adapter. Built by `call_ai()` from its own
arguments; consumers never create one. Exported so that anyone writing an adapter against
`BaseProvider` can type against it. Its fields mirror the `call_ai()` parameters above,
minus `max_retries` and `retry_base_delay`, which belong to the retry wrapper rather than
to the request.

### `BaseProvider`

The abstract base every adapter subclasses. Exported for consumers that call a provider
directly through `get_provider()`, and for anyone implementing an adapter out of tree. Its
class-level `SUPPORTED_FILE_TYPES` declares which file classes the adapter accepts, and
`SUPPORTS_UNLOAD` whether it holds a model resident at all. Both `warm_up()` and
`unload_model()` are concrete rather than abstract, because "nothing to warm up" and
"nothing to release" are valid answers, and an adapter written before either hook existed
keeps working unchanged.

## Exceptions

All of them are importable from `unified_ai_client`.

### `NonRetryableError`

Base class for failures that retrying cannot fix. The retry wrapper re-raises these
immediately instead of spending its budget on a request that will fail identically every
time. Catch this to handle every deterministic failure in one place.

### `UnsupportedFileError`

Raised when a file is passed to a provider that has no way to transmit it, for example
audio to `anthropic` or a PDF to `groq`. The message names the path, the detected type,
and what the provider does accept. Subclasses `NonRetryableError` and `ValueError`, so
handlers written before the exception existed keep working.

The caller is expected to route the file elsewhere, convert it, or drop it. Retrying is
pointless.

### `MissingFileError`

Raised when an attachment path does not exist. Subclasses `NonRetryableError` and
`FileNotFoundError`, so existing handlers keep working.

### `FileDecodeError`

Raised when a file classified as text cannot be decoded as UTF-8, meaning its content does
not match its extension. Subclasses `NonRetryableError` and `ValueError`. The caller is
expected to fix the file or attach it as something other than text; the library will not
substitute placeholder content, which a model cannot distinguish from the real thing.
