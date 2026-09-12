# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
