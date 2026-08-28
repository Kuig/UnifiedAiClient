from __future__ import annotations

from unified_ai_client.providers.openai_compat import OpenAiCompatProvider


class MetaProvider(OpenAiCompatProvider):
    """Meta (Llama API) provider.

    Connects to Llama API's OpenAI-compatible endpoint.
    Default URL: https://api.llama-api.com.
    """

    DEFAULT_URL: str = "https://api.llama-api.com"

    REQUIRES_API_KEY: bool = True
    SECRETS_KEY: str = "meta_api_key"
