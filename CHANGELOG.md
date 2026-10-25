# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- An option key an adapter drops is now named in a log record instead of vanishing: `DEBUG`
  for a library key another provider owns or an Ollama field of another endpoint, `WARNING`
  for a key `google` cannot forward to `GenerateContentConfig`.
- `load_secrets()` logs a `WARNING` naming any `secrets.json` key no provider reads (values
  are never logged; keys starting with `_` are comments). `load_config()` does the same when
  the file or the requested section is not a JSON object.

## [0.6.0] - 2026-10-25

### Added

- `list_providers()` returns the providers usable right now as `ProviderInfo` objects: cloud
  providers with a credential (or an explicit `url`), local servers that answer a one second
  probe, and `script`. `probe_local=False` skips the network entirely.
- `list_models(provider)` returns a provider's models, queried live, as `ModelInfo` objects with
  `reasoning`, `tool_use`, `file_types`, `embeddings` and `context_length`. A field the provider's
  listing does not report is `None` (unknown), never `False`. Ollama reports all of them through
  `/api/show`, Google through its SDK, Anthropic from the model name; the OpenAI-compatible
  providers expose only the id. `script` raises `NotImplementedError`.
- `BaseProvider.list_models()` (concrete, raises `NotImplementedError` by default), plus the
  class-level `SUPPORTS_EMBEDDINGS` and `PROBE_PATH`. An adapter written before them keeps
  working unchanged.
- `get_embedding()` takes keyword-only `file_path`, `dimensions`, `timeout`, `max_retries`,
  `retry_base_delay`, `sleep_time` and `extra_options`, and `text` may be empty when files
  are given. Media are embedded with the text into one vector on `ollama` (image, audio),
  `llamacpp` (image, audio, video) and `google` (image, audio, video, PDF with
  `gemini-embedding-2`); `dimensions` requests a shorter Matryoshka vector on `ollama`,
  `google` and `openai`, and is refused elsewhere rather than ignored. Verified live on
  Ollama with `embeddinggemma-2` and on Google with `gemini-embedding-2`.
- `get_embeddings(provider, model, texts)`, the batch counterpart for text, one request where
  the endpoint takes a list.
- `UnsupportedOperationError` (a `NonRetryableError` and a `NotImplementedError`) and
  `EmptyResponseError` (a `NonRetryableError` and a `RuntimeError`), replacing the plain
  `NotImplementedError` and `RuntimeError` the embedding paths raised, so neither is retried.
- `BaseProvider.embed(EmbeddingRequest)`, the method adapters now implement, and
  `SUPPORTED_EMBEDDING_FILE_TYPES`. `get_embedding(model, text)` is no longer abstract; an
  adapter that overrides it keeps working, served one text at a time.
- `script` protocol: the `embed` payload gains the optional `file_path` and `dimensions`
  fields, sent only when set, so existing scripts see no change for text.
- Video attachments: `classify_file()` has a `video` class (`.mp4`, `.webm`, `.mov`, `.avi`,
  `.mpeg`, `.mpg`). `llamacpp` sends them as llama-server's `input_video` block and `google`
  uploads them through the Files API like any other file; every other provider raises
  `UnsupportedFileError`. These extensions used to classify as `unknown` and be refused
  everywhere.
- `ModelRefusalError` (a `NonRetryableError`), raised when a provider declines on policy
  grounds instead of answering, with `provider`, `category`, `explanation` and
  `partial_text`. Anthropic raises it for `stop_reason: "refusal"`, which used to come back
  as an empty or truncated `AiResponse`.

### Changed

- **Breaking (behaviour):** the library no longer invents sampling or residency values. A
  `temperature`, `top_k`, `top_p` or `keep_alive` that neither the call nor
  `configure_provider()` sets is now left off the request, so the provider's own default
  applies, and on a local server the model's own preset. Previously every request carried
  `temperature=0.7`, and Ollama requests also carried `top_k=64`, `top_p=0.95` and
  `keep_alive="15m"`, overriding the Modelfile. The newer reasoning models reject any
  explicit temperature, so the invented value also turned a plain call into a 400. To keep
  the old output, configure the old values:
  `configure_provider("ollama", temperature=0.7, top_k=64, top_p=0.95, keep_alive="15m")`.
  `BaseProvider.DEFAULT_TEMPERATURE`, `OllamaProvider.DEFAULT_TOP_K` and
  `OllamaProvider.DEFAULT_TOP_P` are removed.
- Anthropic still sends `max_tokens=8192` when none is set, because `/v1/messages` requires
  the field. The value is now `AnthropicProvider.DEFAULT_MAX_TOKENS` and is documented.
- **Breaking (script protocol):** `temperature` in the `generate` payload is now
  `float | null`, like `top_k` and `top_p`, null meaning "use your own default". The
  `preload` payload's `keep_alive` is nullable too. A script doing arithmetic on either must
  handle null.
- `preload_model()`'s `keep_alive` now defaults to `None`, like `warm_up()`'s.

- Ollama: an `extra_options` key that names a request-level field of the endpoint in use
  (`think`, `format`, `truncate`, `shift`, `logprobs`, `top_logprobs`; on `/api/generate`
  also `raw`, `suffix`, `template`, `context`; on `/api/embed` `truncate` and `dimensions`)
  now goes into the request body instead of the model `options`, where Ollama discarded it
  with only a server-side warning. Being merged last, `{"think": "high"}` and a JSON-schema
  `format` override what `thinking` and `format_json` derive. A completion-only field on a
  chat request raises `NonRetryableError`, pointing at `use_generate`.

- Anthropic: `thinking=True` asks for `"display": "summarized"` from Claude 4.7 on, where the
  API otherwise returns thinking blocks with empty text, so `reasoning_text` was always
  empty there. `reasoning_is_summary` is now `True` for a Claude 4 or later trace, which is a
  summary, not the raw chain of thought.
- Anthropic: with thinking on, an explicit `temperature`, `top_k` or `top_p` is dropped with
  a warning (the API refuses them). `temperature` used to be forced to `1`; it is now left
  out, the same default without sending a value.

- OpenAI-compatible family: `API_PREFIX` (default `/v1`) builds every request path, and
  `_reasoning_payload(thinking)` maps `thinking` onto request fields. A subclass sets the
  first when its API lives elsewhere and overrides the second when its reasoning control is
  not a single string field; a dict it returns is merged with the caller's own under the
  same key. Purely additive for existing subclasses.
- `meta` now targets Meta's Model API at `https://api.meta.ai` (models such as
  `muse-spark-1.3`). It used to point at `https://api.llama-api.com`, a third-party domain,
  and Meta's Llama API preview it stood in for has closed. Verified against Meta's
  documentation, not live.

### Fixed

- `get_embedding()` now goes through the same retry, `sleep_time` and timeout handling as
  `call_ai()`. It used to call the adapter directly: a transient 429 failed at once, a
  configured `sleep_time` was ignored, and `script` waited a hard-coded 300 seconds
  regardless of the configured timeout.
- An HTTP 501 Not Implemented is no longer retried: it says the server does not do what was
  asked (Ollama answers it for media sent to an embedding model without the encoder).
- llama.cpp ignored `thinking` altogether. It now sends `reasoning_effort` (`"none"` is
  llama-server's own off switch) together with `chat_template_kwargs.enable_thinking`, which
  Qwen-style templates such as MiniCPM-V 4.5's read instead; a `chat_template_kwargs` in
  `extra_options` is merged, its keys winning.
- Cohere ignored `thinking`; its compatibility API takes `reasoning_effort` `"none"` and
  `"high"`, now sent for `False` and `True`.
- `meta`, `groq` and `xai` declared embeddings support they do not have: none of them
  documents an embeddings endpoint. `SUPPORTS_EMBEDDINGS` is now False for all three, and
  `get_embedding()` raises `NotImplementedError` before sending anything.
- Anthropic ignored `format_json` entirely, although it is documented as forcing JSON. The
  Messages API has no schema-less JSON mode, so `format_json=True` now raises
  `NonRetryableError` unless `extra_options` carries a schema in
  `output_config.format`, which is sent as given.
- Anthropic `thinking=False` was a 400 on Claude Opus 5.5, Fable and Sonnet 5.5, which
  refuse `{"type": "disabled"}`. Sonnet 5.5 now gets `{"type": "between_tools"}`; Opus 5.5 and
  Fable, which cannot turn thinking off, get `output_config.effort: "low"` instead (an effort
  set in `extra_options` wins). Verified against Anthropic's documentation, not live.
- Ollama's `use_generate` sends the system prompt in `/api/generate`'s own `system` field
  instead of refusing it, sends `think` for `thinking=True`/`False` instead of dropping it,
  and reads the trace back from the response's `thinking` field into `reasoning_text`,
  which was always empty. Only `tools`, `tool_results` and `messages` are still refused.
- `preload_model()` honours the `keep_alive` registered via `configure_provider()`. It used
  to send a hard-coded `"15m"`, so the model was loaded with one residency and then served
  by every `call_ai()` with another. Applies to Ollama and `script` alike.
- `script` receives what was registered through `configure_provider("script", ...)`. The
  `generate` and `preload` payloads used to carry only the call-time `extra_options`, so
  configured script settings, and a configured `temperature`, `top_k`, `top_p` or
  `max_tokens`, never reached the script. `warm_up` and `preload` also fall back to a
  configured `keep_alive` and `context_size`.
- `docs/configuration.md` listed `top_k` / `top_p` as defaulting to `64` / `0.95` on every
  provider; only Ollama ever applied them.

## [0.5.8] - 2026-09-12

### Changed

- The provider registry (`get_provider()`, `configure_provider()`, `cleanup()`,
  `unload_model()`, and the four module-level registries behind them) moved out of
  `client.py` into a new `unified_ai_client/registry.py`. `client.py` keeps `call_ai()`,
  `preload_model()`, `warm_up()` and `get_embedding()`. Purely internal: `__init__.py`
  re-exports both halves under the same public names, so `from unified_ai_client import
  get_provider, cleanup` is unaffected. Only code importing `unified_ai_client.client.get_provider`
  directly (or the module itself) needs to update its import.
- `get_provider()`'s dispatch is now table-driven from a single `_PROVIDER_SPECS` registry
  instead of duplicating the provider list by hand across an `if`/`elif` chain, eight
  eager `api_key_*` lookups, a hand-written `ValueError` message, and two docstrings in
  `client.py` alone, plus three separate hand-typed tables in the test suite. Adding a
  provider is now one entry in `registry.py` instead of edits spread across several files.
- The test suite's own provider/class tables (`test_provider_contracts.py`'s
  `_PROVIDER_CLASSES`, `test_warmup.py`'s `_ALL_PROVIDER_CLASSES`) are now derived from
  the same registry instead of being independently hand-typed and unsynchronized with
  each other. `TestDispatch` collapsed from 12 near-identical methods to one `subTest`
  loop over that table. The three duplicated live "reply with PONG" smoke tests
  (Google, Anthropic, OpenAI) consolidated into one `TestLivePongSmoke`, and the seven
  duplicated `load_secrets(...).get(key) or self.skipTest(...)` guards became a shared
  `RequiresCredential` mixin.

### Fixed

- `get_provider()` now looks up only the one credential the requested provider actually
  needs. Every cache miss used to compute all eight cloud providers' secrets
  unconditionally, including for `"ollama"` and `"script"`, which use none of them.

## [0.5.7] - 2026-09-12

### Fixed

- Ollama's `use_generate` no longer silently drops `tools`, `tool_results`, `messages` or
  `system_prompt`. That option routes to `/api/generate`, which has no place for any of the
  four, and `call()` posted only the prompt (and any images) without ever reading them,
  so a caller who turned it on lost its system prompt, its history and its tool calling with
  no error. It now raises `NonRetryableError` naming whichever of the four are set, before any
  of the now-pointless file or history processing runs.

## [0.5.6] - 2026-09-08

### Fixed

- `cleanup()` no longer loses track of a model whose `unload_model()` call fails.
  `_LOADED_MODELS` used to be cleared before the unload attempts, so a single provider
  failing to release lost the tracking for every model in that call, itself included, and
  no later `cleanup()` retried it. It is now cleared entry by entry, only on success.
- Anthropic and Google no longer send an empty text block or Part when the prompt is empty
  but a file is attached. `call_ai(m, prompt="", file_path="photo.png")`, a legitimate
  "describe this" call with the instruction in the system prompt, used to produce a block
  both APIs reject.
- A history message's `content`, `files` and `tool_calls` are preserved together on every
  provider. Anthropic dropped the file when `tool_calls` was present, Google dropped the
  text, and Ollama dropped the tool calls, three different bugs from the same shape: an
  exclusive branch on `tool_calls` where the fields should have been handled independently.
- `openai_compat`'s message builder now preserves `tool_call_id` on a `role: "tool"` history
  entry, which a consumer reconstructing a third turn after `tool_results` has to supply by
  hand and which used to be read from nowhere and silently dropped, producing a 400.
- `set_verbosity()` is now protected by a lock. Two concurrent calls could previously leave
  a handler attached to the logger that nothing referenced any more, duplicating every line
  this library logs with no later call able to detach it.

## [0.5.5] - 2026-09-08

### Fixed

- Options that belong to this library no longer travel to an endpoint that never
  defined them. `context_size` and `use_generate` reached the body of
  `/v1/chat/completions` on every OpenAI-compatible provider, and Google's
  `disable_safety` and friends reached Ollama's model `options`. Only Ollama filtered
  anything, and only in one direction. Copying `config.json.example`, which shipped
  `"context_size": 0` for `lmstudio` and `llamacpp`, was enough to trigger it: harmless
  noise on a permissive local server, a 400 on a strict cloud endpoint. Options the
  library does not define are still forwarded verbatim, which is unchanged and is the
  point of `extra_options`.
- A `temperature` registered through `configure_provider()` now reaches the payload. It
  was the last common lever still defaulting to a value rather than `None`, so "not given"
  and "given 0.7" were indistinguishable and the configured value was overwritten on every
  call, while `top_k` and `top_p` beside it were honoured.

### Changed

- Everything the adapters had in common moved onto `BaseProvider`: option merging, the
  three-source resolution of a common lever, the option namespace and its filter, timeout
  resolution, the reasoning-token split, and the attachment partition with a single
  wording for its refusal. Two of those had already drifted into behaving differently in
  different copies.
- `preload_model()` is a concrete no-op instead of `@abstractmethod`, matching `warm_up()`,
  `unload_model()` and `cleanup()`. Three stub overrides are gone, and a contract test now
  holds it and `SUPPORTS_UNLOAD` together.
- `LM Studio` and `llama.cpp` declare `LAZY_MODEL_LOAD = True` instead of each carrying a
  verbatim copy of the same 27-line `warm_up()`.
- `temperature` is `float | None = None` in `call_ai()` and `AiRequest`. Every existing
  call keeps compiling and keeps sending the same value: the `0.7` that was the signature
  default is now `BaseProvider.DEFAULT_TEMPERATURE`, applied when neither the caller nor
  the configuration supplies one. The `script` protocol still receives a concrete float,
  never `null`, since `docs/script-protocol.md` declares that field non-nullable.
- `config.json.example` no longer lists `context_size` under `lmstudio` and `llamacpp`. It
  never had an effect there and is now explicitly dropped.

## [0.5.4] - 2026-09-08

### Added

- `ProviderHttpError` and `NonRetryableHttpError`, both importable from
  `unified_ai_client`. An HTTP error from a provider endpoint now carries the message the
  provider actually sent, read out of the response body, in `detail` and in `str(exc)`.
  Both subclass `urllib.error.HTTPError`, so existing handlers keep working.

### Fixed

- An HTTP error status no longer arrives as a bare `HTTP Error 400: Bad Request`. The
  three urllib-backed adapters never read the response body, which is where every one of
  these APIs says what it rejected, so a wrong model name, a malformed payload and an
  unsupported parameter were indistinguishable from each other at the call site.
- A 4xx no longer consumes the retry budget. No HTTP provider raised `NonRetryableError`
  at all, so a rejected credential (401), an unknown model (404) or a malformed request
  (400) cost four attempts and 5+10+20 = 35s of backoff before reporting a failure that
  the first response had already settled. The four transient client codes, 408, 409, 425
  and 429, plus every 5xx and any connection failure, stay retryable as before.

### Changed

- The urllib transport moved into a single module, `unified_ai_client/http.py`. The three
  HTTP adapters keep their own `_post`/`_get`, now thin wrappers that resolve the URL and
  add their authentication headers through a new `_auth_headers()` hook. Status
  classification and error-body extraction live in one place instead of being absent from
  three.
- `REQUIRES_API_KEY`, `SECRETS_KEY` and `_require_api_key()` moved onto `BaseProvider`.
  `anthropic` and `openai_compat` carried the same check with a character-for-character
  identical error message; there is now one copy. Adapters that need no credential inherit
  a check that returns before touching anything they do not define.

### Upgrade note

Not breaking for exception handlers: a handler catching `urllib.error.HTTPError`,
`OSError` or `Exception` still catches these. What changes is the timing. A 4xx now
surfaces immediately rather than after roughly 35 seconds of backoff, and the message it
carries is the provider's own rather than the HTTP reason phrase. Code that relied on a
4xx eventually being retried into success will now see it fail on the first attempt, which
is the intended correction: that retry could never have succeeded.

## [0.5.3] - 2026-09-08

### Fixed

- A script's stdout and stderr are now decoded as UTF-8 instead of the platform's locale
  encoding, which is `cp1252` on Windows, the platform this project develops on. The
  documented JSON payload happened to survive the old behaviour, since `json.dumps()`
  escapes every non-ASCII character by default, but a script's stderr on a crash is raw
  text and could come back as mojibake or trigger a decode error.
- Ollama: an explicit `top_k`/`top_p`/`temperature` passed to `call_ai()` now wins over
  the same key registered via `configure_provider()`, matching the precedence
  `docs/configuration.md` documents. `_fold_options()` used to run after those three were
  set from the request and silently overwrote them.
- Ollama: an attachment is no longer read and base64-encoded on a tool-result
  continuation, where the result was always discarded. `call_ai()` now logs a warning
  when `file_path` and `tool_results` are both set, on every provider, since the file is
  never attached to that turn either way and previously vanished without a trace.
- The script provider's `preload_model()`, `warm_up()` and `unload_model()` distinguish a
  genuine failure (a timeout, a malformed response, a missing interpreter) from a mode the
  script does not implement. All three used to catch every exception and report "not
  implemented", discarding the real error, including the script's own stderr.
- `preload_model()` no longer emits a `UserWarning` via the stdlib `warnings` module, the
  one place in the library that did not log through `logging`. `set_verbosity("silent")`
  did not silence it, since it never reached the `unified_ai_client` logger tree.
- Google: a prompt blocked by safety or content filters now logs the `block_reason`
  instead of silently returning `AiResponse(text="")`, indistinguishable from the model
  saying nothing. A failed delete during `cleanup()` is now logged at `debug` instead of
  being swallowed without a trace, the same failure `cleanup()` exists to prevent.

### Documentation

- Corrected several references to a class method that does not exist
  (`_build_file_content_blocks`, `_build_user_content` in `openai_compat.py`'s own
  docstring and in `CLAUDE.md`): the real hook is `_build_native_block()`.
- `google.py`'s `_build_thinking_config()` type hint corrected to `bool | str`, matching
  what it actually receives.
- `load_secrets()`'s credential table now lists all nine supported environment variables,
  not three.
- `docs/api.md` corrected to say that a script implementing `preload` performs a real
  load, matching `docs/warm-up.md` and the code.
- `ARCHITECTURE.md`'s codemap now lists `verbosity.py`.

## [0.5.2] - 2026-09-08

### Changed

- `call_ai()` no longer sends `top_k` and `top_p` unless the caller asks for them. Both
  defaulted to Gemini-shaped values (64 and 0.95) and were sent on every request, which made
  `openai` reject every call: `top_k` is not part of the Chat Completions schema. Ollama keeps
  64 and 0.95 as its own documented defaults, so generation there is unchanged.
- **Script protocol**: `top_k` and `top_p` in the `generate` payload may now be `null`, meaning
  "use your own default". Scripts that read either value without a null check need updating.
- `call_ai(timeout=...)` defaults to `None` instead of 300, and resolves to the timeout
  registered for the provider. See the fix below.
- `configure_provider()` merges on top of the provider's `config.json` section instead of
  replacing it.

### Fixed

- A timeout registered with `configure_provider()` now applies to generation. `call_ai()`
  defaulted to a hard 300 seconds that silently outranked it, and the configured value was
  read only by warm-up, preload and embeddings.
- `configure_provider()` no longer discards the rest of a provider's `config.json` section.
  Registering a single option, such as `context_size`, dropped the `url` and `timeout` the file
  had set, so requests went to the default endpoint with nothing logged to explain it.
- `preload_model()` registers the `atexit` cleanup hook. It tracked the model it loaded but
  never armed the hook that frees it, so a process that only preloaded left the model resident
  for good, which is exactly what `keep_alive=-1` relies on `cleanup()` to prevent.
- Anthropic model ids carrying a release date, such as `claude-opus-4-20250514`, are no longer
  read as version 4.20250514. The date was parsed as the minor version, so the whole Claude 4.0
  line received the adaptive thinking form that only 4.6 and later accept, and `thinking=True`
  failed with a 400.
- `thinking=True` on Anthropic no longer sends `temperature`, `top_k` and `top_p` alongside it.
  The Messages API refuses all three with extended thinking enabled.
- The Google call timeout now ends the wait. The thread pool ran inside a `with` block, whose
  `__exit__` waits for the call to finish, so a hanging request blocked the caller for its full
  duration regardless of the timeout, and the retry loop repeated that cost.
- The capability tables in the test suite now assert that they cover every registered provider.
  `CONTRIBUTING.md` promised this for file-type support, but only the unload matrix enforced it,
  so a provider added without a `SUPPORTED_FILE_TYPES` row passed the suite. The script
  provider's file passthrough, a documented part of the protocol, now has a test as well.

## [0.5.1] - 2026-09-08

### Fixed

- Live tests against a local Ollama server release the model they load, and skip instead of
  failing when the server is unavailable. `ProviderRegistryIsolation` restores the registry
  after every test, so the `atexit` cleanup no longer saw the loaded model and a test run left
  it resident; one struggling server also turned into a cascade of errors rather than one.

## [0.5.0] - 2026-09-08

### Added

- `unload_model(provider, model)` releases a model that a local provider holds resident.
  Implemented by `ollama` and `script`; every other provider inherits a no-op.
- `BaseProvider.SUPPORTS_UNLOAD` declares whether a provider has a residency concept at all.
- `warm_up()` accepts `keep_alive` and `timeout`, so a caller can pin a model for a chosen
  duration without going through `configure_provider()`.
- `OllamaProvider.preload_model()` accepts `timeout`, which previously came only from the
  provider configuration.
- Ollama embedding requests honour `keep_alive`.
- New `unload` mode in the script protocol.
- DEBUG logging for every Ollama request: endpoint, model, effective timeout, effective
  keep_alive and the option keys sent.

### Changed

- `cleanup()` now also unloads the local models this library loaded, not only the remote files
  it uploaded. Pass `unload_models=False` to keep the previous behaviour.
- `keep_alive` accepts an integer as well as a string. A number is seconds, `0` unloads the
  model once idle, and `-1` keeps it resident indefinitely.

### Fixed

- `use_generate` and `url` no longer leak into the Ollama model `options` payload, where they
  are not valid parameters.
- Files uploaded to Google are no longer lost when `configure_provider("google", ...)` rebuilds
  the provider. The upload cache used to live on the provider instance, which the rebuild
  discarded, so the files stayed on the remote quota with nothing left to delete them and the
  next call re-uploaded them from scratch.
- `cleanup()` reaches every provider built in the process, not only those whose instance is
  still cached. A provider reconfigured after it uploaded a file was previously skipped, so its
  remote resources were never released.

## [0.4.1] - 2026-09-06

### Added

- `set_verbosity()` and multi-level logging through the standard library `logging` module.

## [0.4.0] - 2026-08-30

### Added

- `warm_up()`, which pays a provider's one-off setup costs before the first real call.

### Changed

- Multimedia file validation simplified. A file a provider cannot send natively is now refused
  instead of being passed through as text.

### Fixed

- Provider URL resolution.
- Thinking support across providers.

## [0.3.3] - 2026-08-26

### Added

- `AiResponse.reasoning_is_summary`, which tells the caller whether the reasoning text is a
  synthesised summary or a raw trace.

## [0.3.2] - 2026-07-05

### Added

- Tool calling support.
- `use_generate` option for the `ollama` provider.

### Changed

- Default request timeout is now 300 seconds.

### Fixed

- Tool calling on providers other than `ollama`.

## [0.2.0] - 2026-06-11

### Added

- Support for OpenAI-compatible providers.
- `"default"` value for `thinking`, which leaves the provider's own behaviour in place.

### Changed

- Model configuration redefined.
- Provider-specific options rationalised.

### Removed

- The `include_reasoning` flag.

## [0.1.1] - 2026-06-03

### Added

- MIT license.

### Fixed

- Token counting.

## [0.1.0] - 2026-05-29

### Added

- First release. `call_ai()` as the single entry point, returning a normalised `AiResponse`.

[0.5.0]: https://github.com/Kuig/UnifiedAiClient/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/Kuig/UnifiedAiClient/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Kuig/UnifiedAiClient/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/Kuig/UnifiedAiClient/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/Kuig/UnifiedAiClient/compare/v0.2.0...v0.3.2
[0.2.0]: https://github.com/Kuig/UnifiedAiClient/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Kuig/UnifiedAiClient/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Kuig/UnifiedAiClient/releases/tag/v0.1.0
