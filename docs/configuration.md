# Configuration

> [!NOTE]
> **Placeholder.** This file is not written yet. The content below describes what belongs
> here; the [README](../README.md) still holds it for now.

The complete configuration reference. The README keeps only a minimal working example and
links here. Expected contents, moved from the README's `## Configuration` section:

## API credentials

- The precedence order between environment variables, `secrets.json` and built-in
  defaults, with the rationale for each layer.
- **The full key-to-variable mapping**: one row per provider, giving the `secrets.json`
  key and the equivalent environment variable. This table is deep-linked from other
  repositories.
- The shape of `secrets.json`, and which values are *not* credentials and therefore do not
  belong in it.

## Provider settings: `configure_provider()`

Every registrable setting (server URL, timeout, context size, visual token budget, rate
limiting delay, and any provider-specific option), its default, which providers honour it,
and worked examples for the non-obvious combinations.
