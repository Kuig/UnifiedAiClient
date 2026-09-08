from __future__ import annotations

from unified_ai_client.providers.openai_compat import OpenAiCompatProvider


class LlamaCppProvider(OpenAiCompatProvider):
    """llama.cpp server provider.

    Connects to a locally running llama-server using the OpenAI-compatible
    /v1/chat/completions API. Default URL: http://localhost:8080.

    All file handling and reasoning text extraction is inherited from
    OpenAiCompatProvider. llama.cpp does not expose native reasoning tokens;
    reasoning_text will always be an empty string.
    """

    DEFAULT_URL: str = "http://localhost:8080"

    # llama-server routes an input_audio block through miniaudio (mp3, wav,
    # flac), so this is the one local provider that takes audio directly.
    SUPPORTED_FILE_TYPES: frozenset[str] = frozenset({"image", "audio"})

    # Started in multi-model mode, llama-server loads on first inference, so
    # warming up has to send a completion rather than the inherited metadata GET.
    LAZY_MODEL_LOAD: bool = True
