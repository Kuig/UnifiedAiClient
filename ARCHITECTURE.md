# Architecture

## The problem

A project that talks to language models ends up talking to several of them: a local one
during development, a cheap cloud one in bulk, an expensive one where quality matters, and
sooner or later a second vendor because the first had an outage. Each of those speaks a
different dialect. They disagree on where the system prompt goes, on how an image is
attached, on what a reasoning trace is called, on whether token counts come back at all.
So the switching cost is not the API key, it is the pile of per-provider branches that
accumulates in the calling code.

`UnifiedAiClient` removes that pile. One function, `call_ai()`, takes a provider name and
returns the same `AiResponse` whatever answered. The dialects stay inside the library,
one adapter per provider, and the consuming project never learns them.

The second constraint is dependency weight. A library that unifies twelve providers by
importing twelve SDKs has moved the problem rather than solved it: the install becomes
minutes long, the transitive tree becomes a maintenance surface, and a version conflict in
any one SDK breaks a project that never used that provider. Here, exactly one runtime
dependency exists, `google-genai`, and only the Google adapter imports it. Every other
provider speaks HTTP through `urllib` from the standard library.

## Codemap

### Project layout

```
unified_ai_client/
├── __init__.py           # Public API exports
├── client.py             # call_ai(), warm_up(), configure_provider() router + cache
├── models.py             # AiRequest, AiResponse, ProviderConfig dataclasses
├── file_utils.py         # classify_file, validate_files, encode_file_base64, ...
├── exceptions.py         # UnsupportedFileError, MissingFileError, FileDecodeError
├── config.py             # load_secrets(), load_config() (utility)
├── retry.py              # Exponential backoff
├── silence.py            # silence_sdks()
└── providers/
    ├── base.py           # BaseProvider ABC
    ├── google.py         # Google AI (Gemini): upload caching, thinking, cleanup
    ├── ollama.py         # Ollama: urllib, images[], context_size
    ├── openai_compat.py  # Base for OpenAI-compatible APIs
    ├── openai.py         # OpenAI: native audio + PDF blocks
    ├── anthropic.py      # Anthropic: urllib, image/document blocks, thinking
    ├── mistral.py        # Mistral (OpenAI-compatible)
    ├── cohere.py         # Cohere (OpenAI-compatible)
    ├── meta.py           # Meta / Llama API (OpenAI-compatible)
    ├── groq.py           # Groq (OpenAI-compatible)
    ├── xai.py            # xAI (OpenAI-compatible)
    ├── lmstudio.py       # LM Studio (OpenAI-compatible)
    ├── llamacpp.py       # llama.cpp: OpenAI-compatible, native audio
    └── script.py         # External script via stdin/stdout JSON
```

### The entry layer

`client.py` is the only module a consumer needs. It holds `call_ai()`, `get_embedding()`,
`warm_up()`, `preload_model()`, `configure_provider()`, `get_provider()` and `cleanup()`,
plus the two registries behind them: one mapping a provider name to its registered
configuration, the other caching the built provider instance.

`call_ai()` does four things and delegates the rest. It packs its arguments into an
`AiRequest`, resolves a cached provider through `get_provider()`, sleeps for the
rate-limiting interval if one is configured, and invokes the provider through the retry
wrapper. It does not know what an image is, which provider supports reasoning, or how a
tool call is spelled.

`models.py` holds the dataclasses that cross those boundaries: `AiRequest` going in,
`AiResponse` coming out, `ProviderConfig` describing a provider's registered settings, and
the tool-calling triple `ToolDefinition`, `ToolCall`, `ToolResult`.

### The provider layer

Every adapter subclasses `BaseProvider` and implements the same contract: take an
`AiRequest`, produce an `AiResponse`. The twelve of them fall into three families.

**The OpenAI-compatible family** is the workhorse. `OpenAiCompatProvider` implements the
whole `/v1/chat/completions` conversation: message assembly, file blocks, tool schemas,
reasoning parameters, response parsing. Seven providers (`mistral`, `cohere`, `meta`,
`groq`, `xai`, `lmstudio`, `llamacpp`) subclass it in about a dozen lines each, setting
little more than a default URL and whether the endpoint wants a key. `openai` itself adds
one override, for its native audio and PDF content blocks. A bug fixed in the base class
is a bug fixed for eight providers.

**The bespoke adapters** are the three whose APIs are not OpenAI-shaped. `google` uses the
`google-genai` SDK, uploads attachments through the Files API and polls them to the
`ACTIVE` state, keeps an upload cache so the same file is not sent twice, and is the only
provider with remote resources to release. `ollama` speaks the native `/api/chat`
endpoint, which carries images in a single `images[]` field and maps `context_size` to
`num_ctx` and `max_tokens` to `num_predict`. `anthropic` speaks `/v1/messages` over
`urllib`.

**The script provider** spawns an arbitrary executable and exchanges JSON over stdin and
stdout, dispatching on a `mode` field. It turns any program into a provider, which is how
a mock, a local binary or a bespoke inference pipeline gets used without changing the
calling code. The wire format is a public contract, documented in the script protocol
reference under `docs/`.

### The support layer

`file_utils.py` classifies an attachment by extension into `image`, `audio`, `text`,
`document` or `unknown`, validates a list of paths against what a provider declares it
accepts, encodes to base64, and wraps text attachments in delimited blocks. Every adapter
branches on the classification it returns, which is why adding a file type starts here.

`config.py` resolves credentials and loads JSON config into dataclasses. `retry.py`
implements the exponential backoff. `exceptions.py` holds the deterministic failures.
`silence.py` quiets third-party SDK loggers.

## Boundaries

### Configuration layering

Two axes are kept deliberately apart, because they have different lifetimes.

**Stateful infrastructure** is registered once, at application startup, through
`configure_provider()`: the endpoint URL, the network timeout, the rate-limiting delay,
and provider-specific settings such as `context_size` or `keep_alive`. These describe the
connection, not the question. They are cached along with the provider instance, and
resolved credentials are cached with them, so a high-frequency caller does not re-read a
config file or rebuild a client per request. The registries behind that cache are
thread-safe: concurrent configuration of different providers is fully safe, and of the
same provider is safe but last-writer-wins. A local model's context size belongs here
specifically because changing it at call time would force the model to reload and
reallocate GPU memory.

**Stateless per-call parameters** go to `call_ai()` on every invocation: `temperature`,
`top_k`, `top_p`, `max_tokens`, `format_json`, `thinking`, and an `extra_options`
dictionary that overrides any config-level value of the same key. These describe the
question, not the connection.

The library keeps **no conversation state**. It does not track history, context or
previous requests. The consuming project owns its chat history and passes the whole of it
in `messages` on every call. That is what makes a provider instance safely shared and
makes two concurrent calls to the same provider independent.

### The consumer boundary

What crosses out of the library is `AiResponse` and the typed exceptions, and nothing
else. No provider object, no SDK type, no raw payload. A consumer that pattern-matches on
a provider name has reintroduced the problem the library exists to remove.

## Invariants

These hold across every adapter. They are invisible in any single file, and they are the
first thing an edit breaks.

**Rate limiting lives in `call_ai()` and nowhere else.** The sleep happens once, before
the retry wrapper, and therefore not once per attempt. No provider sleeps on its own. Move
it inside the retry loop and a rate-limited provider silently triples its own latency.

**A provider reads its configuration once.** Each HTTP adapter resolves its base URL in
its constructor, from the configured URL or its own default, and never touches the config
object again. Writing a resolved default back into that object corrupts the registered
configuration for everyone holding it.

**A URL that was configured explicitly is honoured exactly.** There is no sentinel value
that means "unset" and also happens to be a real address. An explicitly configured
endpoint is never redirected to a vendor default, even when it matches another provider's.

**An explicit URL suppresses the missing-credential error.** Pointing a cloud adapter at
an address of your own is read as "this is a proxy or a gateway", which may legitimately
need no key. This is deliberate, and it is what lets a local OpenAI-compatible server be
driven through the `openai` adapter.

**Text attachments are always accepted; everything else must be native or the call is
refused.** No provider has a content block for a `.md` or a `.csv`, so text is inlined
into the prompt in delimited blocks. Every other kind of file needs a native mechanism,
and a provider that lacks one raises rather than improvising. The library once did
improvise, reading unsupported files as text and substituting a placeholder when the
decode failed; an audio file then reached the model as the literal string
`[File could not be read as text]`, and the model answered as though it had heard the
recording. A refused call is recoverable. A confidently wrong one is not.

**Deterministic failures are not retried.** Everything under `NonRetryableError` is
re-raised immediately by the retry wrapper. An unsupported attachment or a missing path
fails identically on every attempt, so spending the backoff budget on it only delays the
error the caller needs to see.

**When `tool_results` are present, the prompt is not re-appended.** The consumer is
expected to have placed the original user turn in `messages` already, so appending it
again would duplicate it. Every adapter honours this, and message assembly must keep
honouring it.

**Only the Google adapter imports an SDK.** Every other provider uses `urllib`. The
absence of a dependency tree is the library's main selling point, so a new provider that
needs a vendor SDK needs a conversation first.

## Cross-cutting concerns

**Retry and backoff.** Every provider call goes through the same wrapper: exponential
backoff, all exceptions retryable except the deterministic ones. Providers contain no
retry logic of their own.

**Errors.** `UnsupportedFileError` and `FileDecodeError` subclass both `NonRetryableError`
and `ValueError`; `MissingFileError` pairs `NonRetryableError` with `FileNotFoundError`.
The second base in each pair keeps handlers written against the old plain-exception
behaviour working. Any future deterministic failure belongs under the same base.

**Credentials.** Resolved when a provider instance is first built, from environment
variables first and a `secrets.json` in the consuming project's working directory second.
The check for a missing key fires at request time rather than at construction, because
providers are built eagerly, long before anyone knows whether a request will follow.

**Resource lifecycle.** Only Google holds remote resources. Uploaded files go into a cache
that both `warm_up()` and `call_ai()` read, and `cleanup()` deletes them. Cleanup is
registered with `atexit` on the first call of either, so a process that warms up and then
crashes still releases its uploads.

**Warm-up.** Every provider pays some cost once per process: an SDK import, a client
construction, a TLS handshake, a model load, a file upload. Without `warm_up()` all of it
lands on whichever call runs first, which makes that call look slow purely for going
first. Warming up must not consume generation tokens, so cloud providers use a free
metadata request. LM Studio and llama.cpp are the deliberate exception: both load models
lazily on first inference, so a metadata request would leave the cost exactly where the
warm-up was meant to remove it, and they send a one-token completion instead.

**Logging.** Each provider module owns a logger named after itself. The library never
configures the root logger, and `silence_sdks()` exists so that a consumer can quiet noisy
third-party SDKs without doing so either.
