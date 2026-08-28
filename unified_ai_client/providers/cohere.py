from __future__ import annotations

from unified_ai_client.providers.openai_compat import OpenAiCompatProvider


class CohereProvider(OpenAiCompatProvider):
    """Cohere API provider.

    Connects to Cohere's OpenAI compatibility endpoint.
    Default URL: https://api.cohere.ai/compatibility (the /v1 prefix is added
    by the endpoint paths, so it must not appear in the base URL).
    """

    DEFAULT_URL: str = "https://api.cohere.ai/compatibility"
