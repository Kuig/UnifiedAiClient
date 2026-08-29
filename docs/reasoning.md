# Thinking and Reasoning

## Asking a model to think

```python
response = call_ai(
    provider="google",
    model="gemini-2.5-pro",
    prompt="What is the best sorting algorithm and why?",
    thinking=True,
)
print(response.text)            # Final answer
print(response.reasoning_text)  # Thinking process
```

`thinking` takes three values: `True`, `False`, and `"default"` (the default), which
leaves the decision to the provider. Here is what each provider actually does with them:

| Provider | `True` | `False` | `"default"` |
|---|---|---|---|
| `google` | `thinking_level` on Gemini 3.x, `thinking_budget` on 2.5 | minimised explicitly | provider default, thoughts still captured |
| `ollama` | sends `think: true` | sends `think: false` | field omitted |
| `anthropic` | `{"type": "adaptive"}` on Claude 4.6 and later, `{"type": "enabled", "budget_tokens": N}` below | `{"type": "disabled"}` on 4.6 and later; below that the field is omitted, which is already the off state | field omitted |
| `openai`, `mistral`, `groq`, `xai` | `reasoning_effort: "high"` | `reasoning_effort: "none"` | field omitted |
| `cohere`, `meta`, `lmstudio`, `llamacpp` | no reasoning control in the API; the value is ignored | ignored | ignored |

> [!IMPORTANT]
> On `openai`, `mistral`, `groq` and `xai`, `reasoning_effort` is only sent when you pass
> `True` or `False` explicitly. This matters because a **non-reasoning model rejects the
> parameter outright**: asking `gpt-4o` to think returns an error rather than silently
> doing nothing. Leaving `thinking` at `"default"` never sends the field, so ordinary
> models keep working untouched.

## Why the parameter is this coarse

`thinking` is deliberately a quick, universal lever: three values that mean the same
thing everywhere, so you can switch provider without rewriting the call. That
universality is exactly why it cannot express what each provider offers on its own, such
as Anthropic's `output_config.effort`, the intermediate `reasoning_effort` levels, Groq's
`reasoning_format`, or an exact `thinking_budget` on Google. Those go through
`extra_options`, which is merged into the payload last and therefore always overrides the
mapping above:

```python
call_ai(provider="openai", model="o3", prompt="...", thinking=True,
        extra_options={"reasoning_effort": "medium"})   # wins over "high"
```

It is the same split described in
[Configuration layering](../ARCHITECTURE.md#configuration-layering): the unified surface
covers what every provider shares, `extra_options` covers the rest.

## Reading the trace back

Reasoning extraction is independent of all this. `reasoning_text` is populated whenever
the model returns a trace, whatever `thinking` was set to (`reasoning_content` for the
OpenAI-compatible providers, `thinking` blocks for Ollama, Anthropic and Google).

Check `reasoning_is_summary` before comparing traces across providers: Google returns a
summary of its reasoning, the others return the raw text. Measuring trace length or
composition across the two without checking that flag compares a summary against a
transcript.

`reasoning_tokens` reports the tokens spent thinking, and is `0` where the provider does
not report it separately. Ollama is the one case where the figure is **estimated** rather
than reported: it returns a single `eval_count`, which the adapter splits between the
thinking and the answer in proportion to their character counts.
