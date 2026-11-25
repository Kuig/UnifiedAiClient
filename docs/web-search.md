# Web Search

`call_ai(web_search=True)` lets the model search the web through its provider's own search
tool and ground its answer on what it finds. The searches it ran and the pages it drew on
come back in `AiResponse.grounding`.

```python
from unified_ai_client import call_ai

response = call_ai(
    provider="google",
    model="gemini-3.5-flash",
    prompt="What is the most recent stable release of Python?",
    web_search=True,
)

print(response.text)
print(response.grounding.queries)
for source in response.grounding.sources:
    print(source.title, source.url)
```

## The `web_search` parameter

| Value | Effect |
|---|---|
| `False` (default) | No search tool is offered; `AiResponse.grounding` is `None`. |
| `True` | The provider's search tool is switched on, and the model decides whether to use it. |

`AiResponse.grounding` is a [`Grounding`](api.md#grounding):

| Field | Holds |
|---|---|
| `queries` | The searches the model ran, in order. |
| `sources` | One [`WebSource`](api.md#websource) (`url`, `title`) per page the answer drew on, deduplicated by URL. |
| `search_suggestions` | On `google`, the HTML snippet Google requires you to display (see below). `None` elsewhere. |

An empty `Grounding` means the search was available and the model answered without it,
which is a legitimate answer for a question it could settle on its own.

A value that is not a `bool` raises `ValueError`, and `web_search=True` on a provider
without a search tool raises `UnsupportedOperationError`. Both happen before anything is
sent: without the tool the model would still answer, from its training data alone, and
nothing in the response would say the search never happened.

Searches are billed by the provider on top of the tokens, at a rate of its own.

## Provider support

| Provider | Supported | How |
|---|---|---|
| `google` | ✅ | The `google_search` tool, Gemini 2.0 and newer. Verified live. |
| `anthropic` | ✅ | The `web_search` server tool. Verified offline only. |
| `openai` | ✅ | `web_search_options`, which Chat Completions honours only on the search models (`gpt-5-search-api`, `gpt-4o-search-preview`, `gpt-4o-mini-search-preview`). Any other model answers with a 400. Verified offline only. |
| `script` | ✅ | Passed in the payload; the script decides. See [the script protocol](script-protocol.md). |
| `xai` | ❌ | Search is on the Responses API, which this library does not speak. |
| `mistral` | ❌ | Search is on the Agents API, not Chat Completions. |
| `groq` | ❌ | The compound models search on their own, with no request-level switch. |
| `cohere`, `meta` | ❌ | No search tool on their compatibility endpoints. |
| `ollama`, `lmstudio`, `llamacpp` | ❌ | Local servers, no search. |

Whether a provider supports it is a class attribute, `BaseProvider.SUPPORTS_WEB_SEARCH`,
so `get_provider(name).SUPPORTS_WEB_SEARCH` answers it without sending anything. Whether
a given *model* supports it is the provider's answer to the request.

## Provider notes

### Google

- **Display the search suggestions.** Google's terms for Grounding with Google Search
  require an application showing a grounded answer to display the search suggestions next
  to it. `grounding.search_suggestions` is that HTML, ready to render.
- **The source URLs are redirects.** Each `url` points to
  `vertexaisearch.cloud.google.com/grounding-api-redirect/...`, which leads to the page,
  and the `title` is the site's domain. They are reported as they come: resolving each
  link would cost a request of its own.
- **Search with function calling.** With `tools` as well, Gemini 3 needs the server-side
  tool invocations returned, and the library asks for them. The search then arrives as
  parts of the turn: its queries on the turn that ran it, its sources on the turn that
  writes the grounded answer, which in a tool loop may be a later one. Gemini 2.5 cannot
  combine the two at all (verified live on `gemini-2.5-flash`, `-flash-lite` and `-pro`),
  so on a model named `gemini-2.x` the library raises `UnsupportedOperationError` before
  sending. Either one alone works there. A name without a version, such as
  `gemini-flash-latest`, is treated as a recent model.

### Anthropic

- The tool version follows the model: `web_search_20260209` (with dynamic filtering) on
  Opus and Sonnet 4.6 and newer and on Fable and Mythos, `web_search_20250305` everywhere
  else.
- Web search must be enabled for your organization in the Claude Console.
- A long search can stop with `stop_reason: "pause_turn"`, when Anthropic's own loop
  reaches its iteration limit. The library resumes the turn by itself, up to 5 times, and
  returns it as one response with the usage summed; a turn still paused after that raises
  `NonRetryableError` rather than coming back unfinished.
- A failed search (`max_uses_exceeded`, `too_many_requests`, ...) arrives inside a normal
  response; it is logged as a warning and the model answers without it.
- For settings such as `max_uses`, `allowed_domains` or `user_location`, pass your own tool
  definition in `extra_options={"tools": [{"type": "web_search_20260209", "name":
  "web_search", "max_uses": 3}]}`. It is used instead of the default one, and travels
  alongside any `tools` you declared.

### OpenAI

- The endpoint does not report the queries it ran, so `queries` is always empty. The
  sources are the `url_citation` annotations of the answer.
- Settings such as `user_location` or `search_context_size` go in
  `extra_options={"web_search_options": {...}}`, merged into what the library sends.

## In a tool loop

`web_search=True` can be combined with `tools`. Pass it on every call of the loop: the
search results each provider returns are kept in `response.message`, and replayed with the
history like the rest of the turn (see [tool-calling.md](tool-calling.md)).
