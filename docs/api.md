# API Reference

> [!NOTE]
> **Placeholder.** This file is not written yet. The content below describes what belongs
> here; the [README](../README.md) still holds it for now.

Full reference for every public symbol of `unified_ai_client`, one heading per symbol, so
that each can be deep-linked from outside the repository. Expected contents, moved here
from the README's `## API Reference` section:

## Functions

One `###` heading per function, named exactly as the symbol, each with its full
signature, a table of parameters (name, type, default, one-line description), the return
value, and the exceptions it raises. Ordered by usefulness, entry point first:

- `call_ai` (anchor `#call_ai`, linked from other repositories)
- `get_embedding`
- `preload_model`
- `warm_up`
- `configure_provider`
- `silence_sdks`
- `cleanup`
- `load_secrets`
- `load_config`

## Dataclasses

One `###` heading per class, with a field table (field, type, default, description):
`AiResponse`, `ToolDefinition`, `ToolCall`, `ToolResult`.

## Exceptions

One `###` heading per exception: `UnsupportedFileError`, `MissingFileError`,
`FileDecodeError`. For each, when it is raised and what the caller is expected to do
about it.

## Design Concepts: Caching and Lifecycle

The distinction between stateful configuration registered once and stateless per-call
parameters, and how the two interact.
