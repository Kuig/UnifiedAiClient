# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
