from __future__ import annotations

from unified_ai_client.providers.openai_compat import OpenAiCompatProvider


class XAiProvider(OpenAiCompatProvider):
    """xAI API provider.

    Connects to xAI's OpenAI-compatible API endpoint.
    Default URL: https://api.x.ai.
    """

    DEFAULT_URL: str = "https://api.x.ai"

    REQUIRES_API_KEY: bool = True
    SECRETS_KEY: str = "xai_api_key"
    REASONING_PARAM: str | None = "reasoning_effort"
