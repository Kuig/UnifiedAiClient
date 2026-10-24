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
| `ollama` | sends `think: true`, on `/api/chat` and on `/api/generate` alike | sends `think: false` | field omitted |
| `anthropic` | `{"type": "adaptive"}` on Claude 4.6 and later (with `"display": "summarized"` from 4.7, where the trace is otherwise returned empty), `{"type": "enabled", "budget_tokens": N}` below | `{"type": "disabled"}` on 4.6 and later; `{"type": "between_tools"}` on Sonnet 5.5; on Opus 5.5 and Fable, which cannot turn thinking off, the field is omitted and `output_config.effort` is set to `"low"`, the closest the API allows; below 4.6 the field is omitted, which is already the off state | field omitted |
| `openai`, `mistral`, `groq`, `xai`, `cohere` | `reasoning_effort: "high"` | `reasoning_effort: "none"` | field omitted |
| `llamacpp` | `reasoning_effort: "high"` plus `chat_template_kwargs: {"enable_thinking": true}` | `reasoning_effort: "none"` plus `chat_template_kwargs: {"enable_thinking": false}` | fields omitted |
| `meta`, `lmstudio` | no documented reasoning control in the API; the value is ignored | ignored | ignored |

On `llamacpp` the two fields reach two kinds of chat template: `reasoning_effort: "none"`
is llama-server's own off switch, while Qwen-style templates (MiniCPM-V 4.5 among them)
read `enable_thinking`. A `chat_template_kwargs` you pass in `extra_options` is merged with
the one above, your keys winning.

> [!IMPORTANT]
> On `openai`, `mistral`, `groq`, `xai` and `cohere`, `reasoning_effort` is only sent when you pass
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

call_ai(provider="ollama", model="gpt-oss:20b", prompt="...",
        extra_options={"think": "low"})                 # Ollama's per-model levels
```

It is the same split described in
[Configuration layering](../ARCHITECTURE.md#configuration-layering): the unified surface
covers what every provider shares, `extra_options` covers the rest.

## Reading the trace back

Reasoning extraction is independent of all this. `reasoning_text` is populated whenever
the model returns a trace, whatever `thinking` was set to (`reasoning_content` for the
OpenAI-compatible providers, which llama-server fills by default and LM Studio when set
to separate the trace, `thinking` blocks for Anthropic and Google, the `thinking`
field for Ollama on both of its endpoints).

Check `reasoning_is_summary` before comparing traces across providers: Google returns a
summary of its reasoning, and so does Anthropic from Claude 4 on; the others return the raw
text. Measuring trace length or composition across the two without checking that flag
compares a summary against a transcript.

On Anthropic, sampling parameters do not survive thinking: with thinking on, the Messages
API refuses `top_k` and `top_p` and accepts only the default `temperature`, so an explicit
value is dropped with a warning in the log. From Claude 4.7 on the API refuses all three
whether thinking is on or not, and the resulting error is the API's own.

`reasoning_tokens` reports the tokens spent thinking, and is `0` where the provider does
not report it separately. Ollama is the one case where the figure is **estimated** rather
than reported: it returns a single `eval_count`, which the adapter splits between the
thinking and the answer in proportion to their character counts.
