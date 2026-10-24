from __future__ import annotations

from unified_ai_client.providers.openai_compat import OpenAiCompatProvider


class LmStudioProvider(OpenAiCompatProvider):
    """LM Studio local server provider.

    Connects to a locally running LM Studio instance using its
    OpenAI-compatible /v1/chat/completions API. Default URL:
    http://localhost:1234.

    All file handling and reasoning text extraction is inherited from
    OpenAiCompatProvider. reasoning_text is filled when LM Studio is set to
    separate the trace into 'reasoning_content', and empty otherwise. The
    thinking flag is not mapped: LM Studio documents no request-level switch
    for it.
    """

    DEFAULT_URL: str = "http://localhost:1234"

    # LM Studio loads a model on first inference, so warming up has to send a
    # completion rather than the inherited metadata GET.
    LAZY_MODEL_LOAD: bool = True
