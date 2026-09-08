from __future__ import annotations

from unified_ai_client.providers.openai_compat import OpenAiCompatProvider


class LmStudioProvider(OpenAiCompatProvider):
    """LM Studio local server provider.

    Connects to a locally running LM Studio instance using its
    OpenAI-compatible /v1/chat/completions API. Default URL:
    http://localhost:1234.

    All file handling and reasoning text extraction is inherited from
    OpenAiCompatProvider. LM Studio does not expose native reasoning tokens;
    reasoning_text will always be an empty string.
    """

    DEFAULT_URL: str = "http://localhost:1234"

    # LM Studio loads a model on first inference, so warming up has to send a
    # completion rather than the inherited metadata GET.
    LAZY_MODEL_LOAD: bool = True
