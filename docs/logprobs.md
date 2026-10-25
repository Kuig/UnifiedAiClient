# Token Log Probabilities

`call_ai()` can return the log probability of every generated token, and the most likely
alternatives at each position, on the providers that expose them.

```python
from unified_ai_client import call_ai

response = call_ai(
    provider="ollama",
    model="gemma4:e2b",
    prompt="Is the sky blue? Answer yes or no.",
    thinking=False,
    logprobs=3,
)

for entry in response.logprobs:
    alternatives = ", ".join(f"{alt.token!r} {alt.logprob:.2f}" for alt in entry.top_logprobs)
    print(f"{entry.token!r} {entry.logprob:.2f}  [{alternatives}]")
```

## The `logprobs` parameter

| Value | Effect |
|---|---|
| `None` (default) | Nothing is requested; `AiResponse.logprobs` is `None`. |
| `0` | The log probability of each generated token, with no alternatives. |
| `1` to `20` | The same, plus that many most likely tokens at each position in `top_logprobs`. |

Each entry of `AiResponse.logprobs` is a [`TokenLogprob`](api.md#tokenlogprob): the
token's text, its natural-log probability (at most `0`) and the alternatives, most likely
first. An empty list means the log probabilities were asked for and the provider
returned none, which is also logged as a warning.

A value outside `0` to `20`, or a `bool`, raises `ValueError`. Asking a provider that
offers no log probabilities raises `UnsupportedOperationError`. Both happen before
anything is sent, so neither is retried. A provider may cap the alternatives lower than
`20` and answers a larger value with an error of its own.

## Provider support

| Provider | Supported | Notes |
|---|---|---|
| `ollama` | ✅ | On `/api/chat` and `/api/generate`. The entries cover the whole generation, the thinking trace and its template markers included: pass `thinking=False` for the answer alone. Verified live. |
| `openai` | ✅ | Not on the reasoning models, which answer with a 400. |
| `mistral` | ✅ | Per the API reference. |
| `llamacpp` | ✅ | Mapped by llama-server onto its own `n_probs`. |
| `script` | ✅ | Passed in the payload; the script decides. See [the script protocol](script-protocol.md). |
| `google` | ❌ | The Gemini Developer API answers "Logprobs is not enabled for this model" on every current model. |
| `anthropic` | ❌ | The Messages API has no log probabilities. |
| `groq` | ❌ | Documented as not yet supported by any model. |
| `cohere` | ❌ | `top_logprobs` is not supported by its compatibility API. |
| `xai` | ❌ | Ignored silently by `grok-4.20` and newer, which would read as an empty answer. |
| `lmstudio` | ❌ | Not among the parameters its OpenAI-compatible endpoint documents. |
| `meta` | ❌ | Not documented. |

Whether a provider supports log probabilities is a class attribute,
`BaseProvider.SUPPORTS_LOGPROBS`, so `get_provider(name).SUPPORTS_LOGPROBS` answers it
without sending anything.
