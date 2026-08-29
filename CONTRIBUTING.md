# Contributing

## Development install

Clone the repository, create a virtual environment, and install the package in editable
mode so that source changes are picked up without reinstalling:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e .
```

If you are developing a consuming project against a local checkout, install the library
into **that** project's environment the same way:

```bash
pip install -e /path/to/UnifiedAiClient
```

The only runtime dependency is `google-genai`, pulled in automatically. There is nothing
else to install, including for the tests.

## Running the tests

The suite is stdlib `unittest`. **Not pytest**, and not by preference: test discovery,
the skip behaviour and the helper imports below all assume `unittest`.

```bash
python -m unittest discover -s tests            # everything
python -m unittest tests.test_warmup            # one module
python -m unittest tests.test_warmup.TestWarmUpOllama.test_warm_up_delegates_to_preload_model
```

For a fast feedback loop while working:

```bash
python -m unittest tests.test_provider_contracts tests.test_warmup
```

Those two modules run fully offline with the transport patched, take about 15 seconds, and
cover URL resolution, credential handling, the thinking payloads and the whole warm-up
matrix. Run the full `discover` before committing.

**A full run takes around 10 minutes** when Ollama is up, almost all of it in the live
generation tests.

### What must pass, and what may skip

Offline tests must always pass: imports, dataclasses, file utilities, dispatch, payload
building, tool-call parsing, warm-up transport.

Live tests against real providers are best-effort. A test that needs a resource it cannot
reach, no API key for a cloud provider, or Ollama not running, calls `self.skipTest()` and
reports SKIP. A skip is not a failure, and a suite full of skips on a machine with no
credentials is the expected result.

### Two traps in the suite

**Helper imports are top-level, not package-qualified.** `tests/test_warmup.py` imports
`ProviderRegistryIsolation`, `_make_script` and `_ollama_available` from `test_providers`,
not from `tests.test_providers`. Discovery puts `tests/` on `sys.path`, so the
package-qualified form breaks.

**Anything touching `configure_provider()` must subclass `ProviderRegistryIsolation`**,
defined in `tests/test_providers.py`. It snapshots and restores the global provider
config and instance registries. Under alphabetical test ordering, one leaked registry
entry silently changes the behaviour of a later test.

## Code style

- `from __future__ import annotations` at the top of every module.
- Modern type hints throughout: `X | None`, `list[str]`, `dict[str, int]`. Mandatory on
  every function signature, parameters and return type alike.
- Google-style docstrings on every public function, with `Args:` and `Returns:`.
- PEP 8.

There is no linter, formatter or type-checker configured, and `pyproject.toml` carries
packaging metadata only. Match the surrounding style by hand rather than adding tooling.

## Adding a provider

For an OpenAI-compatible endpoint, most of the work is already done. Subclass
`OpenAiCompatProvider` and set `DEFAULT_URL`, plus `REQUIRES_API_KEY` and `SECRETS_KEY`
for a cloud endpoint, and `REASONING_PARAM` if the API exposes a reasoning control.

Set `SUPPORTED_FILE_TYPES` to what you have **verified** the endpoint accepts. The
inherited default is `{"image"}`, and declaring a class whose content block the builder
cannot produce raises at request-build time rather than at import.

Then wire it in and document it:

- a branch in the dispatch chain in `client.py`, including the name in its `ValueError`
  message, and the API-key lookup beside it;
- the entry in the environment-variable map in `config.py`;
- `config.json.example`, `secrets.json.example` and `.env.example`;
- the provider tables in `README.md`, `COMPARISON.md`, `docs/multimodal.md`,
  `docs/reasoning.md`, `docs/tool-calling.md` and `docs/warm-up.md`;
- a `test_dispatch_<name>` in `TestDispatch`, and a row in every table in
  `tests/test_provider_contracts.py`. Those tables are keyed by provider name, and
  `TestFileSupportMatrix._SUPPORT` asserts the declared capabilities against an explicit
  list, so a provider added without a row there fails the suite.

A provider that needs a vendor SDK is a different conversation: the absence of a
dependency tree is the point of the library.

## Documentation

Reference material and internal design live outside the README, which stays the entry
document. `docs/api.md` is hand-written, not generated: its `###` headings are the anchors
other repositories deep-link to, so they are stable by hand. When content moves between
documents, update whatever pointed at it in the same change.
