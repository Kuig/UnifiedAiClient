from __future__ import annotations

from unified_ai_client.providers.openai_compat import OpenAiCompatProvider


class GroqProvider(OpenAiCompatProvider):
    """Groq API provider.

    Connects to Groq's OpenAI-compatible API endpoint.
    Default URL: https://api.groq.com/openai.
    """

    DEFAULT_URL: str = "https://api.groq.com/openai"

    REQUIRES_API_KEY: bool = True
    SECRETS_KEY: str = "groq_api_key"
    REASONING_PARAM: str | None = "reasoning_effort"
