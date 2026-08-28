# UnifiedAiClient — Library Comparison

This page compares `UnifiedAiClient` with the most commonly used Python libraries
for the same use case: calling LLM providers through a unified interface.

## Feature Matrix

| Feature | **UnifiedAiClient** | **LiteLLM** | **aisuite** | **openai SDK** |
|---|---|---|---|---|
| **Mandatory dependencies** | `urllib` (stdlib) + `google-genai` (Google only) | `openai`, `httpx`, `pydantic`, `tiktoken`, `aiohttp` + ~10 more | Provider SDKs (optional, per-provider) | `httpx`, `pydantic`, `anyio` |
| **Cloud providers** | 8 | 100+ | 10+ | OpenAI / Azure only |
| **Local providers** | Ollama, LM Studio, llama.cpp | Ollama (via OpenAI-compat) | Ollama | — |
| **Script / subprocess provider** | ✅ stdin/stdout JSON protocol | ❌ | ❌ | ❌ |
| **Unified call interface** | `call_ai(provider=, model=)` | `completion("provider/model")` | `client.chat.completions.create("provider:model")` | `client.chat.completions.create()` |
| **Multimodal — images** | ✅ all supporting providers | ✅ | ✅ partial | ✅ |
| **Multimodal — audio** | ✅ Google, OpenAI, Ollama | ✅ partial | ❌ | ✅ |
| **Multimodal — PDF** | ✅ Google, OpenAI, Anthropic | ✅ | ❌ | ✅ |
| **Text embeddings** | ✅ `get_embedding()` | ✅ | ❌ | ✅ |
| **Tool / function calling** | ✅ all providers | ✅ | ✅ | ✅ |
| **Thinking / reasoning** | ✅ Google, Anthropic, Ollama | ✅ (transparent) | ❌ explicit | ❌ explicit |
| **Ollama VRAM-aware preloading** | ✅ `preload_model()` | ❌ | ❌ | — |
| **Provider warm-up (zero-token)** | ✅ `warm_up()`, all providers | ❌ | ❌ | ❌ |
| **Normalised response object** | ✅ `text`, `input_tokens`, `output_tokens`, `reasoning_tokens`, `reasoning_text`, `tool_calls` | ✅ `ModelResponse` | ❌ (provider-native) | ❌ (provider-native) |
| **Built-in retry / backoff** | ✅ | ✅ | ❌ | ✅ |
| **JSON output mode** | ✅ `format_json=True` | ✅ | ❌ | ✅ |
| **Streaming** | ❌ | ✅ | ✅ | ✅ |
| **Async** | ❌ | ✅ | ❌ | ✅ |
| **Cost tracking** | ❌ | ✅ advanced | ❌ | ❌ |
| **Proxy / gateway mode** | ❌ | ✅ | ❌ | ❌ |
| **Stateless by design** | ✅ | ✅ | ✅ | ✅ |
| **Remote file cleanup** | ✅ (Google uploads) | ✅ auto | — | — |

## Where UnifiedAiClient Stands Out

**Zero heavy dependencies.** Every provider except Google is implemented via `urllib`
from the standard library — no SDK juggling, no transitive dependency bloat. The entire
library installs in seconds and works in minimal environments out of the box.

**Local-first, VRAM-aware.** `preload_model()` pins a model into GPU memory with the
correct context window size (`num_ctx`) before the first call, preventing Ollama from
silently reloading the model mid-session when a different `num_ctx` arrives. No other
library in this comparison exposes this control.

**Script provider.** Any script that implements the
[`LLM_Behaviour_Interface`](LLM_Behaviour_Interface.md) JSON stdin/stdout protocol can
be used as a drop-in backend. This makes it straightforward to mock models in tests,
wrap local binaries, or route calls to custom inference pipelines — all without changing
the calling code.

**Clean tool calling separation.** Tool definitions, tool call responses, and tool
results are three distinct typed dataclasses (`ToolDefinition`, `ToolCall`,
`ToolResult`). The library handles only the transport layer — serialising definitions
into each provider's wire format and deserialising responses back into a normalised
structure. The execution loop stays in the consumer, which is where it belongs.

**Rich, normalised `AiResponse`.** Every call returns the same dataclass regardless of
provider: response text, input/output token counts, reasoning token count, reasoning
transcript, and tool calls. Accessing usage metrics or reasoning traces does not require
knowing which provider was used.
