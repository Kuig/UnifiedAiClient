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
