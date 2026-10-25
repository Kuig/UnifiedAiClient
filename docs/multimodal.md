# Multimodal Input

How to attach files to a call, what each provider accepts, and what happens when it
accepts nothing suitable.

## Attaching files

Pass a local file path, or a list of paths, via `file_path`. The library handles the
encoding and upload for you, choosing the native mechanism each provider offers:

```python
# Single image
response = call_ai(
    provider="ollama",
    model="gemma4:e2b",
    prompt="Describe this diagram.",
    file_path="C:/docs/figures/diagram.png",
)

# Multiple files (image + text document)
response = call_ai(
    provider="google",
    model="gemini-2.5-pro",
    prompt="Summarize the attached report.",
    file_path=["report.pdf", "notes.md"],
)
```

Files attached this way apply to the current turn. To attach files to an earlier turn of
a conversation, add a `"files"` key to that entry of `messages`:

```python
response = call_ai(
    provider="google",
    model="gemini-2.5-pro",
    prompt="And what changed in the second one?",
    messages=[
        {"role": "user", "content": "What is in this chart?", "files": ["q1.png"]},
        {"role": "assistant", "content": "It shows quarterly revenue..."},
    ],
    file_path=["q2.png"],
)
```

Both paths are validated the same way.

### Attaching files to history {#attaching-files-to-history}

A history entry's `files`, its `content`, and a `tool_calls` list, when present, are
independent: none of them is dropped for the others being there. A message that recorded
both the file the model looked at and the tool call it made from that still carries both on
every provider. See [Tool Calling](tool-calling.md#a-third-turn-continuing-after-a-tool-result)
for how a `role: "tool"` history entry and its `tool_call_id` are handled, which differs by
provider.

## Supported file types per provider

| Provider | Images | Audio | Video | Text files | PDFs |
|---|---|---|---|---|---|
| `google` | ✅ Upload | ✅ Upload | ✅ Upload | ✅ Upload | ✅ Upload |
| `openai` | ✅ base64 | ✅ base64 | ❌ Raises | ✅ Inlined | ✅ base64 |
| `anthropic` | ✅ base64 | ❌ Raises | ❌ Raises | ✅ Inlined | ✅ base64 |
| `llamacpp` | ✅ base64 | ✅ base64 | ✅ base64 | ✅ Inlined | ❌ Raises |
| `ollama` | ✅ base64 | ❌ Raises | ❌ Raises | ✅ Inlined | ❌ Raises |
| `mistral` | ✅ base64 | ❌ Raises | ❌ Raises | ✅ Inlined | ❌ Raises |
| `cohere` | ✅ base64 | ❌ Raises | ❌ Raises | ✅ Inlined | ❌ Raises |
| `meta` | ✅ base64 | ❌ Raises | ❌ Raises | ✅ Inlined | ❌ Raises |
| `groq` | ✅ base64 | ❌ Raises | ❌ Raises | ✅ Inlined | ❌ Raises |
| `xai` | ✅ base64 | ❌ Raises | ❌ Raises | ✅ Inlined | ❌ Raises |
| `lmstudio` | ✅ base64 | ❌ Raises | ❌ Raises | ✅ Inlined | ❌ Raises |
| `script` | Path passed | Path passed | Path passed | Path passed | Path passed |

Video files are `.mp4`, `.webm`, `.mov`, `.avi`, `.mpeg` and `.mpg`, the containers both
Gemini and llama-server's ffmpeg decoder take. On `llamacpp` a video travels as an
`input_video` block, llama-server's own extension to the OpenAI format; the model and its
`mmproj` must handle video, or the server refuses the request. On `google` a long video
takes a while to reach the `ACTIVE` state after upload: raise `upload_poll_timeout` (15
seconds by default) if the upload gives up before the file is ready. `ollama` has no
field for video on `/api/chat`; extract frames and attach them as images instead.

## The two rules

**Text files are always accepted, everything else must be native.** No provider
exposes a dedicated block type for a `.md` or a `.csv`, so text attachments are
wrapped in a delimited block and prepended to the prompt. Google is the exception:
it uploads text through the Files API alongside every other attachment, because
Gemini reads those files natively. Every other kind of file needs a native
mechanism, and where the provider has none, the call raises `UnsupportedFileError`
rather than sending something the model cannot read:

```python
call_ai(provider="groq", model="...", prompt="Transcribe this.", file_path="talk.mp3")
# UnsupportedFileError: Provider 'groq' cannot accept audio files: 'talk.mp3'.
# Accepted: image, text.
```

The alternative, which earlier versions did, is to read the file as text anyway and
substitute a placeholder when that fails. The model then answers as though it had
received the recording, and nothing in the response says otherwise. A refused call
is recoverable; a confident answer about a file the model never saw is not.

Three consequences worth knowing:

- Files are validated **before** any of them is encoded or uploaded, so one bad path
  in a list fails immediately instead of after the others have been sent.
- A missing file raises `MissingFileError`, a `FileNotFoundError` subclass. Neither it
  nor `UnsupportedFileError` is retried: both would fail identically on every attempt,
  so the error surfaces at once instead of after the backoff budget.
- Unrecognised file types raise rather than being guessed at as text. Read such a
  file yourself and pass the content in `prompt`.

The `script` provider is exempt: it receives raw paths and decides for itself, since
only the script knows what it can open. See
[the script protocol](script-protocol.md).

## What counts as a text file

Classification is by extension, covering the usual source, markup, data and config
formats: `.txt`, `.md`, `.csv`, `.json`, `.jsonl`, `.yaml`, `.toml`, `.xml`, `.svg`,
`.html`, `.py`, `.js`, `.jsx`, `.ts`, `.tsx`, `.vue`, `.svelte`, `.css`, `.scss`,
`.c`, `.h`, `.cpp`, `.hpp`, `.cs`, `.java`, `.kt`, `.go`, `.rs`, `.rb`, `.php`,
`.swift`, `.dart`, `.sh`, `.ps1`, `.sql`, `.proto`, `.graphql`, `.tf`, `.patch`, and
others in the same vein.

Files whose name is their type are recognised too, matched case-insensitively:
`Dockerfile`, `Makefile`, `LICENSE`, `README`, `CHANGELOG`, `.gitignore`, `.env`,
`.editorconfig`, `.dockerignore` and similar. A suffixed variant such as
`Dockerfile.dev` is not recognised and raises.

`.svg` is treated as text rather than as an image on purpose: no provider accepts SVG
in an image block, and the markup is more useful to the model than a refusal.

> [!WARNING]
> `.env` files and similar config are inlined verbatim, secrets included.
> Nothing is scanned or redacted, so check what a file holds before attaching it.

## OCR models

OCR models such as GLM-OCR are ordinary vision models driven by a fixed task prompt
(`"Text Recognition:"`, `"Table Recognition:"`, `"Formula Recognition:"`), and they want
near-greedy sampling. Since the library sends no sampling value you did not set, pass the
ones the model's authors recommend:

```python
response = call_ai(
    provider="ollama",                 # or "llamacpp" with the model's mmproj loaded
    model="glm-ocr",
    prompt="Text Recognition:",
    file_path="page-001.png",
    temperature=0.0, top_p=0.00001, top_k=1, max_tokens=8192,
    extra_options={"repeat_penalty": 1.1, "use_generate": True},
)
```

On Ollama and llama.cpp the repetition setting is spelled `repeat_penalty`; the
`repetition_penalty` spelling used by vLLM and Hugging Face is not recognised and is
dropped by the server. `use_generate` is optional on Ollama: GLM-OCR's own guide
recommends `/api/generate`, and the native `/api/chat` works too. These models read
images, not PDFs: rasterise each page first.

## Embedding files

`get_embedding()` takes a `file_path` too, embedding the files together with the text into
one vector. The rules are the same two, with a set of classes of its own per provider:
`ollama` embeds images and audio, `llamacpp` images, audio and video, `google` images,
audio, video and PDFs, `script` receives the paths and decides, and the rest embed text
only. Text files are inlined into the input.
Whether a given model can read the medium is up to the model: a text-only tag answers with
an error of its own, which is not retried. See
[`get_embedding`](api.md#get_embedding) for the table.

## Ollama and audio

Ollama serves audio-capable models such as Gemma 4 `e2b`, but its native `/api/chat`
endpoint, which this adapter uses, has no field to carry audio; putting it in
`images[]` is silently ignored. Reaching those models needs Ollama's
OpenAI-compatible endpoint, which costs the `/api/chat` options this adapter depends
on (`context_size`, `keep_alive`, native thinking). Until that is wired up, audio to
`ollama` in `call_ai()` raises. This is a limitation of the adapter, not of Ollama.
Embeddings are different: `/api/embed` does carry audio, so `get_embedding()` accepts it.
