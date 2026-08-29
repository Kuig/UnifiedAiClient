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

## Supported file types per provider

| Provider | Images | Audio | Text files | PDFs |
|---|---|---|---|---|
| `google` | ✅ Upload | ✅ Upload | ✅ Upload | ✅ Upload |
| `openai` | ✅ base64 | ✅ base64 | ✅ Inlined | ✅ base64 |
| `anthropic` | ✅ base64 | ❌ Raises | ✅ Inlined | ✅ base64 |
| `llamacpp` | ✅ base64 | ✅ base64 | ✅ Inlined | ❌ Raises |
| `ollama` | ✅ base64 | ❌ Raises | ✅ Inlined | ❌ Raises |
| `mistral` | ✅ base64 | ❌ Raises | ✅ Inlined | ❌ Raises |
| `cohere` | ✅ base64 | ❌ Raises | ✅ Inlined | ❌ Raises |
| `meta` | ✅ base64 | ❌ Raises | ✅ Inlined | ❌ Raises |
| `groq` | ✅ base64 | ❌ Raises | ✅ Inlined | ❌ Raises |
| `xai` | ✅ base64 | ❌ Raises | ✅ Inlined | ❌ Raises |
| `lmstudio` | ✅ base64 | ❌ Raises | ✅ Inlined | ❌ Raises |
| `script` | Path passed | Path passed | Path passed | Path passed |

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

## Ollama and audio

Ollama serves audio-capable models such as Gemma 4 `e2b`, but its native `/api/chat`
endpoint, which this adapter uses, has no field to carry audio; putting it in
`images[]` is silently ignored. Reaching those models needs Ollama's
OpenAI-compatible endpoint, which costs the `/api/chat` options this adapter depends
on (`context_size`, `keep_alive`, native thinking). Until that is wired up, audio to
`ollama` raises. This is a limitation of the adapter, not of Ollama.
