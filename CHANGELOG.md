# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
