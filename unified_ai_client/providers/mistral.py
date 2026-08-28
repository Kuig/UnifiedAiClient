from __future__ import annotations

from unified_ai_client.providers.openai_compat import OpenAiCompatProvider


class MistralProvider(OpenAiCompatProvider):
    """Mistral API provider.

    Connects to Mistral's API endpoint using the OpenAI-compatible
    /v1/chat/completions API. Default URL: https://api.mistral.ai.
    """

    DEFAULT_URL: str = "https://api.mistral.ai"

    REQUIRES_API_KEY: bool = True
    SECRETS_KEY: str = "mistral_api_key"
    REASONING_PARAM: str | None = "reasoning_effort"
