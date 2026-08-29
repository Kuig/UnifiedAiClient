# Tool Calling

> [!NOTE]
> **Placeholder.** This file is not written yet. The content below describes what belongs
> here; the [README](../README.md) still holds it for now.

Function calling end to end. Expected contents, moved from the README's
`### Tool Calling (Function Calling)` section:

- **Defining tools**: the `ToolDefinition` shape and how a schema is declared.
- **The two-turn exchange**: the model answering with `ToolCall` objects, the caller
  executing them, and the `ToolResult` values going back in a second `call_ai()`.
- **Provider compatibility**: which providers support tool calling, and how the ones that
  do differ.
